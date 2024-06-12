# Main loop for the genetic algorithm to optimize the interface
# Use .yaml file to load the settings

# Import general libraries
import sys
import os
import random
import yaml
import pickle
import shutil
import zipfile
from datetime import datetime
from ase.io import read

# Import genetic algorithm modules
from deap import base, creator, tools

# Import the neural network modules
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

# Internal modules
from ind_manipulation import (
    ind_creator, ind_creator_amap, ind_creator_dist_control, cross_over_1pt, assign_struct,
    assign_graph, assign_graph_ind, structure_mutation, atom_num_mutation
)
from gnn_energy_calculator import energy_calculate_gnn
from ga_functions_gnn import (
    replace, restart, random_select, print_gen, write_hof, write_offspring, write_pool,
    check_redundancy, check_constrain
)

# Load the settings from yaml input file
try:
    print('Reading GA parameters from input file...')
    with open('ga_input.yaml', 'r') as f_input:
        input_settings = yaml.safe_load(f_input)
except FileNotFoundError:
    print('Input file: ga_input.yaml does not exist. Exiting...')
    sys.exit()

# Assign parameters to variables
try:
    # General settings
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

    # File settings
    write_restart_file = input_settings['output_settings']['write_restart_file']
    setting_file = input_settings['output_settings']['setting_file']
    offspring_file = input_settings['output_settings']['offspring_file']
    pool_summary = input_settings['output_settings']['pool_summary']
    best_objective_file = input_settings['output_settings']['best_objective_file']
    hof_file = input_settings['output_settings']['hof_file']
except KeyError as e:
    print(f'Parameter {str(e)} does not exist in input file! Exiting...')
    sys.exit()

# Check if the summation of variational parameters equals 1
total_prob = (
    probability_crossover +
    probability_structure_mutation +
    probability_atom_num_mutation +
    probability_random_replace
)
if abs(total_prob - 1) > 0.000001:
    print('Total variation probability not equal to one! Exiting...')
    sys.exit()

# Initialize the Hall of Fame
hof = tools.ParetoFront(similar=check_redundancy)
toolbox = base.Toolbox()

# Define an energy Hall of Fame to store the structure with the lowest energy ever seen
energy_hof = tools.HallOfFame(num_member_hof, similar=check_redundancy)

# Define the GIN class for GNN energy calculation
class GIN(torch.nn.Module):
    def __init__(self, input_dim, conv_hidden_dim, linear_hidden_dim,
                 node_output_dim, graph_output_dim, conv_num_layers,
                 dropout, task='graph'):
        super(GIN, self).__init__()

        self.dropout = dropout
        self.task = task
        output_dim = graph_output_dim if task == 'graph' else node_output_dim

        self.conv_num_layers = conv_num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(self.build_conv(input_dim, conv_hidden_dim))

        for _ in range(conv_num_layers - 1):
            self.convs.append(self.build_conv(conv_hidden_dim, conv_hidden_dim))

        self.bns = torch.nn.ModuleList()
        for _ in range(conv_num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(conv_hidden_dim))

        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(conv_hidden_dim, linear_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(linear_hidden_dim, output_dim),
        )

    def build_conv(self, in_dim, hidden_dim):
        return GINConv(torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        ))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
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

# Generate the atom objects for side structures & load the GNN model
model = torch.load(model_file, map_location='cpu')
left_atom_obj = read(left_side_file_name, format='vasp')
right_atom_obj = read(right_side_file_name, format='vasp')

# Initialize genetic algorithm
def setup():
    creator.create('FitnessMin', base.Fitness, weights=(-1.0, 1.0))
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

# Function to run the genetic algorithm
def run():
    global hof, energy_hof, read_restart_file, index

    # Reset index count
    with open(setting_file, 'a+') as f_obj:
        f_obj.write(f'run_title = {run_title}\n' +
                    f'read_restart_file = {str(read_restart_file)}\n' +
                    f'write_restart_file = {write_restart_file}\n' +
                    f'write_restart_every = {str(write_restart_every)}\n' +
                    f'index = {str(index)}\n' +
                    f'num_member_hof = {str(num_member_hof)}\n' +
                    f'max_generation = {str(max_generation)}\n' +
                    f'population_size = {str(population_size)}\n' +
                    f'cell_height = {str(cell_height)}\n' +
                    f'left_side_atom_num = {str(left_side_atom_num)}\n' +
                    f'right_side_atom_num = {str(right_side_atom_num)}\n' +
                    f'sand_box_path = {sand_box_path}\n' +
                    f'single_run_time_limit = {str(single_run_time_limit)}\n' +
                    f'num_promoted = {str(num_promoted)}\n' +
                    # Interface structure settings
                    f'interface_len = {str(interface_len)}\n' +
                    f'left_e_per_atom = {str(left_e_per_atom)}\n' +
                    f'right_e_per_atom = {str(right_e_per_atom)}\n' +
                    f'min_width = {str(min_width)}\n' +
                    f'max_width = {str(max_width)}\n' +
                    f'max_height = {str(max_height)}\n' +
                    f'min_atom_num = {str(min_atom_num)}\n' +
                    f'max_atom_num = {str(max_atom_num)}\n' +
                    f'atom_type_list = {str(atom_type_list)}\n' +
                    # Variation parameters
                    f'cut_loc_mu = {str(cut_loc_mu)}\n' +
                    f'cut_loc_sigma = {str(cut_loc_sigma)}\n' +
                    f'structure_mutation_fraction = {str(structure_mutation_fraction)}\n' +
                    f'structure_mutation_sigma = {str(structure_mutation_sigma)}\n' +
                    f'atom_num_mutation_mean = {str(atom_num_mutation_mean)}\n' +
                    f'atom_num_mutation_sigma = {str(atom_num_mutation_sigma)}\n' +
                    f'probability_crossover = {str(probability_crossover)}\n' +
                    f'probability_structure_mutation = {str(probability_structure_mutation)}\n' +
                    f'probability_atom_num_mutation = {str(probability_atom_num_mutation)}\n' +
                    f'probability_random_replace = {str(probability_random_replace)}\n' +
                    # File settings
                    f'setting_file = {setting_file}\n' +
                    f'offspring_file = {offspring_file}\n' +
                    f'pool_summary = {pool_summary}\n' +
                    f'best_objective_file = {best_objective_file}\n'
                    )

    # Reset offspring file
    with open(offspring_file, 'a+') as f_obj:
        f_obj.write('Offspring fitness and history\n')

    # Reset objective file
    with open(best_objective_file, "a+") as f:
        f.write("#Evolution of the best objective with generation\n")
        f.write("#Generation \t Objective\n")
        f.write(f'start time: {datetime.now().strftime("%d %B, %Y at %H:%M:%S.")}\n')

    # Reset the pool summary
    with open(pool_summary, 'a+') as f_obj:
        f_obj.write('# Pool Summary\n')

    # Write the initial lines of hof_file
    with open(hof_file, 'a+') as f_obj:
        f_obj.write('Keep record of the best structures\n')

    # Restart an evolution or start a new one
    if read_restart_file:
        try:
            gen, population, pre_index = restart(write_restart_file)
            index += pre_index
        except Exception:
            print(f"Cannot restart from {write_restart_file}, quitting...")
            read_restart_file = False
            sys.exit()
    else:
        gen = 0

        # Delete the zip file with the same name to avoid conflicts
        zip_file_name = os.path.join(sand_box_path, f'{run_title}.zip')
        try:
            os.remove(zip_file_name)
            print(f'{zip_file_name} removed!')
        except FileNotFoundError:
            print('No conflicting zip file detected!')

        # Create the initial population
        print(f'Start time: {datetime.now().strftime("%d %B, %Y at %H:%M:%S.")}')
        all_ind_not_satisfy = True
        trail_population = 1

        while all_ind_not_satisfy:
            print(f'Start trail initial population: {trail_population}')
            population = toolbox.population(n=population_size)

            # Assign the corresponding graph to each individual
            for ind in population:
                toolbox.assign_struct(ind)

            # Assign index and fitness for all the individuals in the population & check constraints
            for ind in population:
                toolbox.assign_index(ind)
                toolbox.evaluate(ind, population, hof)
                if not toolbox.satisfy_constrain(ind):
                    original_fit = ind.fitness.values[0]
                    ind.fitness.values = (100, -100)
                    history = f'Set individual {ind.index} original fitness: {original_fit} to arbitrary large since not satisfy constraints'
                    ind.history += history
                else:
                    print(f'Individual: {ind.index} satisfy the constraints!')

                if ind.fitness.values[0] != 100:
                    all_ind_not_satisfy = False
                    print(f'Individual: {ind.index} is a good one!')

            trail_population += 1

            # Delete the trail population folders if none of them satisfy constraints
            if all_ind_not_satisfy:
                for ind in population:
                    shutil.rmtree(str(ind.index))

        # Print officially start info
        print(f"Create a population of size {population_size}")
        print(f"Run {max_generation} generations")
        print(f'GA start time: {datetime.now().strftime("%d %B, %Y at %H:%M:%S.")}')

    # Begin evolution
    while gen < max_generation:
        population = toolbox.sort_best(population, population_size)
        write_pool(population, gen, pool_summary)
        print(f"-- Generation {gen} -- {datetime.now().strftime('%d %B, %Y at %H:%M:%S.')}")
        sys.stdout.flush()

        if gen >= max_generation:
            print('Reached required generation number, exiting...')
            sys.exit()

        # Select next generation
        offspring = toolbox.random_sel(population, len(population) - num_promoted)
        offspring = list(toolbox.map(toolbox.clone, offspring))

        # Apply crossover on part of the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < probability_crossover:
                if toolbox.is_same(child1, child2):
                    if random.random() < 0.5:
                        toolbox.structure_mutation(child2)
                        toolbox.atom_num_mutation(child1)
                    else:
                        toolbox.structure_mutation(child1)
                        toolbox.atom_num_mutation(child2)
                toolbox.crossover(child1, child2)
                history = f'crossover({child1.index} & {child2.index})'
                child1.history = history
                child2.history = history

                del child1.fitness.values
                del child2.fitness.values
                del child1.struct
                del child2.struct

                toolbox.assign_struct(child1)
                toolbox.assign_struct(child2)

        # Apply mutation on other offspring
        total_mut_prob = (
            probability_structure_mutation +
            probability_atom_num_mutation +
            probability_random_replace
        )
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
                    history = f'Structure mutation( {mutant.index} )'
                    mutant.history = history

            prob += normalize_atom_num_mutation_prob
            if mutant.fitness.valid:
                if rand < prob:
                    toolbox.atom_num_mutation(mutant)
                    del mutant.fitness.values
                    del mutant.struct
                    toolbox.assign_struct(mutant)
                    history = f'Atom number mutation ({mutant.index})'
                    mutant.history = history

            prob += normalize_random_replace_prob
            if mutant.fitness.valid:
                if rand < prob:
                    toolbox.replace_by_random(mutant)
                    del mutant.fitness.values
                    del mutant.struct
                    toolbox.assign_struct(mutant)
                    history = f'Replaced by random ({mutant.index})'
                    mutant.history = history

        # Check if there are still some individuals that did not change in variation
        for ind in offspring:
            if ind.fitness.valid:
                toolbox.replace_by_random(ind)
                del ind.fitness.values
                del ind.struct
                del ind.atom_obj
                del ind.rotate_obj
                toolbox.assign_struct(ind)
                history = f'Replaced by random individual({ind.index}) AFTER-check'
                ind.history = history

        # Redundancy guard
        test_list = list(toolbox.map(toolbox.clone, population))
        for new_ind in offspring:
            for test_ind in test_list:
                if toolbox.is_same(new_ind, test_ind):
                    if random.random() < 0.5:
                        toolbox.structure_mutation(new_ind)
                        history = f'Structure mutate from individual({new_ind.index}) RED'
                    else:
                        toolbox.atom_num_mutation(new_ind)
                        history = f'Atom number mutate from individual({new_ind.index}) RED'
                    del new_ind.fitness.values
                    del new_ind.struct
                    del new_ind.atom_obj
                    del new_ind.rotate_obj
                    toolbox.assign_struct(new_ind)
                    new_ind.history += history

        # Pick out the individuals that changed in variation
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # Assign new index & new fitness to those individuals, assign new structure if it's deleted in the last step
        # Also assign very large fitness to those that failed constraint limits
        for bad_ind in invalid_ind:
            try:
                bad_ind.struct
            except AttributeError:
                toolbox.assign_struct(bad_ind)

        for bad_ind in invalid_ind:
            toolbox.assign_index(bad_ind)
            toolbox.evaluate(bad_ind, population, hof)
            try:
                satisfy_constrain = toolbox.satisfy_constrain(bad_ind)
            except ValueError:
                satisfy_constrain = False
            if not satisfy_constrain:
                original_fit = bad_ind.fitness.values[0]
                bad_ind.fitness.values = (100, -100)
                history = f'Set individual {bad_ind.index} fitness from {original_fit} to arbitrary large since not satisfy constraints'
                bad_ind.history += history

        # Update the pool
        # Write a file that keeps record of offspring
        write_offspring(offspring, gen, offspring_file)

        # Decide the candidates for the next generation
        candidates = list(toolbox.map(toolbox.clone, offspring + population[num_promoted:]))
        candidates = toolbox.sort_best(candidates, len(candidates))

        # Update population and Hall of Fame
        population[num_promoted:] = list(toolbox.map(toolbox.clone, candidates[:(population_size - num_promoted)]))
        hof.update(population)
        energy_hof.update(population)

        # Get a list of the index of individuals in energy Hall of Fame
        hof_index = [ind.index for ind in hof]
        energy_hof_index = [ind.index for ind in energy_hof]

        # Print the population information to the screen
        print_gen(population, gen, best_objective_file, energy_hof, hof=hof)

        # Write both Hall of Fame to the hof file
        write_hof(hof_file, gen, hof, energy_hof)

        # Move the unselected files to sandbox zip file and delete unnecessary files
        with zipfile.ZipFile(os.path.join(sand_box_path, f'{run_title}.zip'), 'a', zipfile.ZIP_DEFLATED) as z:
            for ind in candidates[(population_size - num_promoted):]:
                if ind.index not in energy_hof_index + hof_index:
                    for root, _, files in os.walk(str(ind.index)):
                        for file in files:
                            z.write(os.path.join(str(ind.index), file))
                    shutil.rmtree(str(ind.index))

        # Increment the generation
        gen += 1

        # Write the restart pickle file
        if (write_restart_file is not None) and (gen % write_restart_every == 0):
            cp = dict(population=population, generation=gen, index=index, rndstate=random.getstate())
            with open(f'{write_restart_file}.tmp', "wb") as f:
                pickle.dump(cp, f)
            shutil.move(f'{write_restart_file}.tmp', write_restart_file)

# Function to assign index to each individual
def assign_index(ind):
    global index
    index += 1
    ind.index = index

pwd = os.getcwd()
execute_dir = os.path.join(pwd, run_title)
if not os.path.exists(execute_dir):
    os.mkdir(execute_dir)
os.chdir(execute_dir)
setup()
run()