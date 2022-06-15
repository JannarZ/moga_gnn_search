# Main loop for the genetic algorthim to optimize the interface
# Use .yaml file to load the settings

# import general libraries

import sys
import os
import random
import yaml
import pickle
import shutil
import zipfile
from datetime import datetime
from ase.io import read

# import genetic algorithm modules

from deap import base
from deap import creator
from deap import tools

# import the neural network modules
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

# internal modules

from ind_manipulation import ind_creator
from ind_manipulation import ind_creator_amap
from ind_manipulation import ind_creator_dist_control
from ind_manipulation import cross_over_1pt
from ind_manipulation import assign_struct
from ind_manipulation import assign_graph
from ind_manipulation import assign_graph_ind
from ind_manipulation import structure_mutation
from ind_manipulation import atom_num_mutation
from gnn_energy_calculator import energy_calculate_gnn
from ga_functions_gnn import replace
from ga_functions_gnn import restart
from ga_functions_gnn import random_select
from ga_functions_gnn import print_gen
from ga_functions_gnn import write_hof
from ga_functions_gnn import write_offspring
from ga_functions_gnn import write_pool
from ga_functions_gnn import check_redundancy
from ga_functions_gnn import check_constrain

# load the settings from yaml input file
try:
    print('Reading ga parameters from input file...')
    with open('ga_input.yaml', 'r') as f_input:
        input_settings = yaml.safe_load(f_input)
except FileNotFoundError:
    print('Input file: ga_input.yaml does not exist. Exiting...')
    sys.exit()

# assign parameters to variables
try:
    # general settings
    run_title = input_settings['general_settings']['run_title']
    read_restart_file = input_settings['general_settings']['read_restart_file']
    write_restart_every = input_settings['general_settings']['write_restart_every']
    index = input_settings['general_settings']['index']
    cutoff = input_settings['general_settings']['cutoff']
    model_file = input_settings['general_settings']['model_file']
    num_member_hof = input_settings['general_settings']['num_member_hof']
    max_generation = input_settings['general_settings']['max_generation']
    population_size = input_settings['general_settings']['population_size']
    left_side_atom_num = input_settings['general_settings']['left_side_atom_num']
    right_side_atom_num = input_settings['general_settings']['right_side_atom_num']
    left_side_file_name = input_settings['general_settings']['left_side_file_name']
    right_side_file_name = input_settings['general_settings']['right_side_file_name']
    sand_box_path = input_settings['general_settings']['sand_box_path']
    single_run_time_limit = input_settings['general_settings']['single_run_time_limit']
    num_promoted = input_settings['general_settings']['num_promoted']

    # Interface structure settings
    cell_height = input_settings['interface_settings']['cell_height']
    interface_len = input_settings['interface_settings']['interface_len']
    left_e_per_atom = input_settings['interface_settings']['left_e_per_atom']
    right_e_per_atom = input_settings['interface_settings']['right_e_per_atom']
    min_width = input_settings['interface_settings']['min_width']
    max_width = input_settings['interface_settings']['max_width']
    max_height = input_settings['interface_settings']['max_height']
    min_atom_num = input_settings['interface_settings']['min_atom_num']
    max_atom_num = input_settings['interface_settings']['max_atom_num']
    max_coord_num = input_settings['interface_settings']['max_coord_num']
    inter_atom_limit = input_settings['interface_settings']['inter_atom_limit']
    atom_type_list = input_settings['interface_settings']['atom_type_list']

    # Variation parameters
    cut_loc_mu = input_settings['variational_settings']['cut_loc_mu']
    cut_loc_sigma = input_settings['variational_settings']['cut_loc_sigma']
    structure_mutation_fraction = input_settings['variational_settings']['structure_mutation_fraction']
    structure_mutation_sigma = input_settings['variational_settings']['structure_mutation_sigma']
    atom_num_mutation_mean = input_settings['variational_settings']['atom_num_mutation_mean']
    atom_num_mutation_sigma = input_settings['variational_settings']['atom_num_mutation_sigma']
    probability_crossover = input_settings['variational_settings']['probability_crossover']
    probability_structure_mutation = input_settings['variational_settings']['probability_structure_mutation']
    probability_atom_num_mutation = input_settings['variational_settings']['probability_atom_num_mutation']
    probability_random_replace = input_settings['variational_settings']['probability_random_replace']

    # File setting
    write_restart_file = input_settings['output_settings']['write_restart_file']
    setting_file = input_settings['output_settings']['setting_file']
    offspring_file = input_settings['output_settings']['offspring_file']
    pool_summary = input_settings['output_settings']['pool_summary']
    best_objective_file = input_settings['output_settings']['best_objective_file']
    hof_file = input_settings['output_settings']['hof_file']
except KeyError as e:
    print('Parameter ' + str(e) + ' does not exist in input file! Exiting...')
    sys.exit()

# check if the summation of variational parameters equals to 1
total_prob = probability_crossover + \
             probability_structure_mutation + \
             probability_atom_num_mutation + \
             probability_random_replace
if (total_prob - 1) > 0.000001:
    print('Total variation probability not equal to one! Exiting...')
    sys.exit()

hof = tools.ParetoFront(similar=check_redundancy)
toolbox = base.Toolbox()

# define a energy hof to store the structure with the lowest energy ever seen
# normal HallOfFame obj in DEAP should sort structures base on first fitness
energy_hof = tools.HallOfFame(num_member_hof, similar=check_redundancy)


# define the GIN class for GNN energy calculation
class GIN(torch.nn.Module):
    def __init__(self, input_dim, conv_hidden_dim, linear_hidden_dim,
                 node_output_dim, graph_output_dim, conv_num_layers,
                 dropout, task='graph'):
        # implement this function that initializes the layers for node embeding

        super(GIN, self).__init__()

        # probability of an element to be zeroed
        self.dropout = dropout

        # determine the output dimension from task
        self.task = task
        if self.task == 'graph':
            output_dim = graph_output_dim
        elif self.task == 'node':
            output_dim = node_output_dim

        # so layers can be changed later
        self.conv_num_layers = conv_num_layers
        self.convs = torch.nn.ModuleList()

        # fisrt layer have different input chanel
        self.convs.append(self.build_conv(input_dim, conv_hidden_dim))

        for i in range(conv_num_layers - 1):
            self.convs.append(self.build_conv(conv_hidden_dim, conv_hidden_dim))

        # batch normal layers
        self.bns = torch.nn.ModuleList()
        for i in range(conv_num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(conv_hidden_dim))

        # post message passing layers after convolution
        self.post_mp = torch.nn.Sequential(torch.nn.Linear(conv_hidden_dim, linear_hidden_dim),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(self.dropout),
                                           torch.nn.Linear(linear_hidden_dim, output_dim),)

    def build_conv(self, in_dim, hidden_dim):
        return GINConv(torch.nn.Sequential(torch.nn.Linear(in_dim, hidden_dim),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_dim, hidden_dim)))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        # implement this function that takes the feature tensor x,
        # edge_index tensor and returns the output tensor

        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.conv_num_layers - 1):
            x = self.convs[i](x.float(), edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        x = global_add_pool(x, batch)
        emb = x
        x = self.post_mp(x)

        return x, emb


# generate the atom object for side structures & load the GNN model
model = torch.load(model_file, map_location='cpu')
left_atom_obj = read(left_side_file_name, format='vasp')
right_atom_obj = read(right_side_file_name, format='vasp')


# initialize genetic algorithm
def setup():
    creator.create('FitnessMin', base.Fitness, weights=((-1.0, 1.0)))
    creator.create('Individual', list, fitness=creator.FitnessMin, index=int, formation_e=float, history='Initial')

    toolbox.register('ind_creator', ind_creator_amap,
                     interface_len=interface_len,
                     min_width=min_width,
                     max_width=max_width,
                     max_height=max_height,
                     cell_height=cell_height,
                     atom_type_list=atom_type_list,
                     left_atom_obj=left_atom_obj,
                     right_atom_obj=right_atom_obj,
                     inter_atom_limit=inter_atom_limit,
                     filter_range=3.5,
                     loop_limit=1000)
    toolbox.register('evaluate', energy_calculate_gnn,
                     model=model,
                     left_e_per_atom=left_e_per_atom,
                     right_e_per_atom=right_e_per_atom,
                     left_atom_obj=left_atom_obj,
                     right_atom_obj=right_atom_obj,
                     interface_len=interface_len,
                     time_limit=single_run_time_limit,
                     k=5)
    toolbox.register('crossover', cross_over_1pt, cut_loc_mu=cut_loc_mu, cut_loc_sigma=cut_loc_sigma)
    toolbox.register('replace_by_random', replace,
                     interface_len=interface_len,
                     min_width=min_width,
                     max_width=max_width,
                     max_height=max_height,
                     cell_height=cell_height,
                     min_atom_num=min_atom_num,
                     max_atom_num=max_atom_num,
                     atom_type_list=atom_type_list)
    toolbox.register('assign_index', assign_index)
    toolbox.register('assign_struct', assign_struct,
                     left_atom_obj=left_atom_obj,
                     right_atom_obj=right_atom_obj,
                     pbc=[True, True, False],
                     cutoff=cutoff)
    toolbox.register('random_sel', random_select)
    toolbox.register('sort_best', tools.selNSGA2)
    toolbox.register('structure_mutation', structure_mutation,
                     frac_atom=structure_mutation_fraction,
                     max_height=max_height,
                     std=structure_mutation_sigma)
    toolbox.register('atom_num_mutation', atom_num_mutation,
                     sigma=atom_num_mutation_sigma,
                     min_atom_num=min_atom_num,
                     max_atom_num=max_atom_num,
                     max_height=max_height,
                     atom_type_list=atom_type_list)
    toolbox.register('is_same', check_redundancy)
    toolbox.register('satisfy_constrain', check_constrain,
                     left_atom_obj=left_atom_obj,
                     right_atom_obj=right_atom_obj,
                     min_num=min_atom_num,
                     max_num=max_atom_num,
                     max_coord_num=max_coord_num,
                     cutoff=cutoff,
                     max_height=max_height,
                     inter_atom_limit=inter_atom_limit)
    toolbox.register('map', map)

    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.ind_creator)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)


def run():
    global hof, energy_hof, read_restart_file, index

    # reset index count
    with open(setting_file, 'a+') as f_obj:
        f_obj.write('run_title = ' + run_title + '\n' +
                    'read_restart_file = ' + str(read_restart_file) + '\n' +
                    'write_restart_file = ' + write_restart_file + '\n' +
                    'write_restart_every = ' + str(write_restart_every) + '\n' +
                    'index =' + str(index) + '\n' +
                    'num_member_hof = ' + str(num_member_hof) + '\n' +
                    'max_generation = ' + str(max_generation) + '\n' +
                    'population_size = ' + str(population_size) + '\n' +
                    'cell_height = ' + str(cell_height) + '\n' +
                    'left_side_atom_num = ' + str(left_side_atom_num) + '\n' +
                    'right_side_atom_num = ' + str(right_side_atom_num) + '\n' +
                    'sand_box_path = ' + sand_box_path + '\n' +
                    'single_run_time_limit = ' + str(single_run_time_limit) + '\n' +
                    'num_promoted = ' + str(num_promoted) + '\n' +

                    # interface structure settings
                    'interface_len = ' + str(interface_len) + '\n' +
                    'left_e_per_atom = ' + str(left_e_per_atom) + '\n' +
                    'right_e_per_atom = ' + str(right_e_per_atom) + '\n' +
                    'min_width = ' + str(min_width) + '\n' +
                    'max_width = ' + str(max_width) + '\n' +
                    'max_height = ' + str(max_height) + '\n' +
                    'min_atom_num = ' + str(min_atom_num) + '\n' +
                    'max_atom_num = ' + str(max_atom_num) + '\n' +
                    'atom_type_list = ' + str(atom_type_list) + '\n' +

                    # variation parameters
                    'cut_loc_mu = ' + str(cut_loc_mu) + '\n' +
                    'cut_loc_sigma = ' + str(cut_loc_sigma) + '\n' +
                    'structure_mutation_fraction = ' + str(structure_mutation_fraction) + '\n' +
                    'structure_mutation_sigma = ' + str(structure_mutation_sigma) + '\n' +
                    'atom_num_mutation_mean = ' + str(atom_num_mutation_mean) + '\n' +
                    'atom_num_mutation_sigma = ' + str(atom_num_mutation_sigma) + '\n' +
                    'probability_crossover = ' + str(probability_crossover) + '\n' +
                    'probability_structure_mutation = ' + str(probability_structure_mutation) + '\n' +
                    'probability_atom_num_mutation = ' + str(probability_atom_num_mutation) + '\n' +
                    'probability_random_replace = ' + str(probability_random_replace) + '\n' +

                    # file setting
                    'setting_file = ' + setting_file + '\n' +
                    'offspring_file = ' + offspring_file + '\n' +
                    'pool_summary = ' + pool_summary + '\n' +
                    'best_objective_file = ' + best_objective_file + '\n'
                    )

    # reset offspring file
    with open(offspring_file, 'a+') as f_obj:
        f_obj.write('Offspring fitness and history')
        f_obj.write('\n')

    # reset objective file
    with open(best_objective_file, "a+") as f:
        f.write("#Evolution of the best objective with generation")
        f.write('\n')
        f.write("#Generation \t Objective")
        f.write('\n')
        f.write('start time: %s' % (datetime.now().strftime("%d %B, %Y at %H:%M:%S.")))
        f.write('\n')

    # reset the pool summary
    with open(pool_summary, 'a+') as f_obj:
        f_obj.write('# Pool Summary')
        f_obj.write('\n')

    # write the initial lines of hof_file
    with open(hof_file, 'a+') as f_obj:
        f_obj.write('Keep record of the best structures')
        f_obj.write('\n')

    # restart an evolution or start a new one
    if read_restart_file is True:
        try:
            gen, population, pre_index = restart(write_restart_file)
            index += pre_index
        except:
            print("cannot restart from %s, quiting..." % write_restart_file)
            read_restart_file = False
            sys.exit()
    else:
        gen = 0

        # delete the zip file with same name, so don't need to manually do it for repeat jobs
        zip_file_name = os.path.join(sand_box_path, run_title + '.zip')
        try:
            os.remove(zip_file_name)
            print(zip_file_name + ' removed!')
        except FileNotFoundError:
            print('No contradict zip file detected!')

        # if all the individuals have -100 fitness then regenerate the population
        print('Start time: %s' % (datetime.now().strftime("%d %B, %Y at %H:%M:%S.")))
        all_ind_not_satisfy = True
        trail_population = 1
        while all_ind_not_satisfy:
            # create new population
            print('Start trail initial population: ' + str(trail_population))
            population = toolbox.population(n=population_size)

            # assign the corresponding graph to each individual
            for ind in population:
                toolbox.assign_struct(ind)

            # assign index and fitness for all the individuals in the population & and check constrain
            for ind in population:
                toolbox.assign_index(ind)
                toolbox.evaluate(ind, population, hof)
                # check if individual satisfy the constrains
                if not toolbox.satisfy_constrain(ind):
                    original_fit = ind.fitness.values[0]
                    ind.fitness.values = (100, -100)
                    histroy = f'Set individual {ind.index} original fitness: ' \
                              f'{original_fit} to arbitrary large since not satisfy constrains'
                    ind.history += histroy
                else:
                    print('Individual: ' + str(ind.index) + ' satisfy the constrain!')

                # if the fitness is not -100 then enter the main loop with this good structure
                if ind.fitness.values[0] != 100:
                    all_ind_not_satisfy = False
                    print('Individual: ' + str(ind.index) + ' is a good one!')

            # add one to trail population
            trail_population += 1

            # delete the trail population folders, if none of them satisfy constrain
            if all_ind_not_satisfy:
                for ind in population:
                    shutil.rmtree(str(ind.index))

        # print officially start info
        print("Create a population of size %d" % population_size)
        print("Run %d generations" % max_generation)
        print('GA start time: %s' % (datetime.now().strftime("%d %B, %Y at %H:%M:%S.")))

    # begin evolution
    while gen < max_generation:

        population = toolbox.sort_best(population, population_size)
        write_pool(population, gen, pool_summary)
        print("-- Generation %i -- %s" % (gen, datetime.now().strftime("%d %B, %Y at %H:%M:%S.")))
        sys.stdout.flush()

        if gen >= max_generation:
            print('Reached required generation number, exiting...')
            sys.exit()

        # select next generation
        offspring = toolbox.random_sel(population, len(population) - num_promoted)

        # clone the offspring
        offspring = list(toolbox.map(toolbox.clone, offspring))

        # apply crossover on part of the offsprings
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < probability_crossover:
                # if the parents are same apply some mutation
                if toolbox.is_same(child1, child2):
                    if random.random() < 0.5:
                        toolbox.structure_mutation(child2)
                        toolbox.atom_num_mutation(child1)
                    else:
                        toolbox.structure_mutation(child1)
                        toolbox.atom_num_mutation(child2)
                toolbox.crossover(child1, child2)
                histroy = 'crossover(%d & %d)' % (child1.index, child2.index)
                child1.history = histroy
                child2.history = histroy

                del child1.fitness.values
                del child2.fitness.values
                del child1.struct
                del child2.struct

                toolbox.assign_struct(child1)
                toolbox.assign_struct(child2)

        # apply mutation on other offsprings
        total_mut_prob = probability_structure_mutation + probability_atom_num_mutation + probability_random_replace
        normalize_struct_mutation_prob = probability_structure_mutation / total_mut_prob
        normalize_atom_num_mutation_prob = probability_atom_num_mutation / total_mut_prob
        normalize_random_replace_prob = probability_random_replace / total_mut_prob
        for mutant in offspring:
            prob = normalize_struct_mutation_prob
            rand = random.random()
            if mutant.fitness.valid:
                if rand < prob:
                    toolbox.structure_mutation(mutant)
                    del mutant.fitness.values
                    del mutant.struct
                    toolbox.assign_struct(mutant)
                    history = 'Structure mutation( %d )' % mutant.index
                    mutant.history = history

            prob += normalize_atom_num_mutation_prob
            if mutant.fitness.valid:
                if rand < prob:
                    toolbox.atom_num_mutation(mutant)
                    del mutant.fitness.values
                    del mutant.struct
                    toolbox.assign_struct(mutant)
                    history = 'Atom number mutation (%d)' % mutant.index
                    mutant.history = history

            prob += normalize_random_replace_prob
            if mutant.fitness.valid:
                if rand < prob:
                    toolbox.replace_by_random(mutant)
                    del mutant.fitness.values
                    del mutant.struct
                    toolbox.assign_struct(mutant)
                    history = 'Replaced by random (%d)' % mutant.index
                    mutant.history = history

        # check if there is still some individuals that not changed in variation
        for ind in offspring:
            if ind.fitness.valid:
                toolbox.replace_by_random(ind)
                del ind.fitness.values
                del ind.struct
                del ind.atom_obj
                del ind.rotate_obj
                toolbox.assign_struct(ind)
                histroy = 'Replaced by random individual(%d) AFTER-check' % ind.index
                ind.history = histroy
            
        # redundancy guard
        test_list = list(toolbox.map(toolbox.clone, population))
        for new_ind in offspring:
            for test_ind in test_list:
                if toolbox.is_same(new_ind, test_ind):
                    if random.random() < 0.5:
                        toolbox.structure_mutation(new_ind)
                        history = 'Structure mutate from individual(%d) RED' % new_ind.index
                    else:
                        toolbox.atom_num_mutation(new_ind)
                        history = 'Atom number mutate from individual(%d) RED' % new_ind.index
                    del new_ind.fitness.values
                    del new_ind.struct
                    del new_ind.atom_obj
                    del new_ind.rotate_obj
                    toolbox.assign_struct(new_ind)
                    new_ind.history += history

        # pick out the individuals that changed in variation
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # assign new index & new fitness to those individuals, assign new structure if it's deleted in last step
        # also assign really large fitness to those failed constrains limit
        for bad_ind in invalid_ind:
            try:
                bad_ind.struct
            except AttributeError:
                toolbox.assign_struct(bad_ind)

        for bad_ind in invalid_ind:
            toolbox.assign_index(bad_ind)
            toolbox.evaluate(bad_ind, population, hof)
            # check if individual satisfy the constrains
            try:
                satisfy_constrain = toolbox.satisfy_constrain(bad_ind)
            except ValueError:
                satisfy_constrain = False
            if not satisfy_constrain:
                original_fit = bad_ind.fitness.values[0]
                bad_ind.fitness.values = (100, -100)
                histroy = f'Set individual {bad_ind.index} fitness from {original_fit} ' \
                          f'to arbitrary large since not satisfy constrains'
                bad_ind.history += histroy

        # update the pool
        # write a file that keep record of offspring
        write_offspring(offspring, gen, offspring_file)

        # decide the candiate for next generation
        candidates = list(toolbox.map(toolbox.clone, offspring + population[num_promoted:]))
        candidates = toolbox.sort_best(candidates, len(candidates))

        # update population and hall of fame
        population[num_promoted:] = list(toolbox.map(toolbox.clone, candidates[:(population_size - num_promoted)]))
        hof.update(population)
        energy_hof.update(population)

        # get a list of the index of ind in energy hof
        hof_index = [_.index for _ in hof]
        energy_hof_index = [_.index for _ in energy_hof]

        # print the population information to the screen
        print_gen(population, gen, best_objective_file, energy_hof, hof=hof)

        # write both hof to the hof file
        write_hof(hof_file, gen, hof, energy_hof)

        # move the unselected files to sandbox zip file and delete unnecessary files
        with zipfile.ZipFile(os.path.join(sand_box_path, run_title + '.zip'), 'a', zipfile.ZIP_DEFLATED) as z:
            for ind in candidates[(population_size - num_promoted):]:
                if ind.index not in energy_hof_index + hof_index:
                    for root, dirs, files in os.walk(str(ind.index)):
                        for file in files:
                            z.write(os.path.join(str(ind.index), file))
                    shutil.rmtree(str(ind.index))

        # generation plus one
        gen += 1

        # write the restart pickle file
        if (write_restart_file is not None) and (gen % write_restart_every == 0):
            cp = dict(population=population, generation=gen, index=index, rndstate=random.getstate())
            with open(write_restart_file + '.tmp', "wb") as f:
                pickle.dump(cp, f)
            shutil.move(write_restart_file + '.tmp', write_restart_file)


# function for assign index of each individual
def assign_index(ind):
    global index
    index += 1
    ind.index = index


pwd = os.getcwd()
excute_dir = pwd + '/' + run_title
if not os.path.exists(excute_dir):
    os.mkdir(excute_dir)
os.chdir(excute_dir)
setup()
run()



