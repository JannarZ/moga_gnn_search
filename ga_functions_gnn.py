# some necessary function performing genetic algorithm
# import needed module
import random
import pickle

import networkx as nx
import numpy as np
from copy import deepcopy
from ind_manipulation import attach
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from pymatgen.analysis.structure_matcher import StructureMatcher


# random select individuals from population
def random_select(population, num_sel):
    return random.sample(population, num_sel)


# roulette select
def roulette_select(population, num_sel):
    pass


def restart(restart_file):

    # load restart file
    with open(restart_file, "rb") as f:
        cp = pickle.load(f)
    gen = cp["generation"]
    population = cp["population"]
    random.setstate(cp['rndstate'])
    index = cp['index']
    old_size = len(population)
    print("restart from %s , gen %d (population size %d)" % (restart_file, gen, old_size))
    print('-'*60)
    print("-- Generation %i --" % gen)
    return gen, population, index

# index value must exits for assign_index function
index = 0


# function for assign index of each individual
def assign_index(ind):
    global index
    index += 1
    ind.index = index


# function for writing the hof file
def write_hof(hof_file, gen, hof, energy_hof):
    with open(hof_file, 'a+') as f_obj:
        f_obj.write('Generation: ' + str(gen) + '\n')
        for i, hero in enumerate(hof):
            f_obj.write(f'Index:{hero.index} \t Novelty:{hero.fitness.values[1]:.5f} \t '
                        f'Formation energy:{hero.fitness.values[0]:.5f} \t History:{hero.history}')

        f_obj.write('Energy Hall of Fame: \n')
        for i, hero in enumerate(energy_hof):
            f_obj.write(f'Index: {hero.index} \t Novelty: {hero.fitness.values[1]:.5f} \t '
                        f'Formation energy: {hero.fitness.values[0]:.5f} \t History: {hero.history}\n')


# function that print each generation's info
def print_gen(population, gen, best_objective_file, energy_hof, hof=None):

    # Gather all the fitnesses in one list and print the stats
    energy_objs = [ind.fitness.values[0] for ind in population]
    novelty_objs = [ind.fitness.values[1] for ind in population]
    pop_size = len(population)

    # Calculate the statistical information for formation energy
    energy_mean = sum(energy_objs) / pop_size
    energy_sum2 = sum(x*x for x in energy_objs)
    energy_std = abs(energy_sum2 / pop_size - energy_mean ** 2) ** 0.5

    # Calculate the statistical information for novelty
    novelty_mean = sum(novelty_objs) / pop_size
    novelty_sum2 = sum(x * x for x in novelty_objs)
    novelty_std = abs(novelty_sum2 / pop_size - novelty_mean ** 2) ** 0.5

    # Write the statistical information to objective file and screen
    with open(best_objective_file, "a") as f:
        f.write(f'{gen} \t {min(energy_objs)} \t {max(novelty_objs)}')
        f.write("\n")
    print(f'Energy: Min: {min(energy_objs)} \t Max {max(energy_objs)} \t Avg {energy_mean} \t Std {energy_std} \t')
    print(f'Novelty: Min: {min(novelty_objs)} \t Max {max(novelty_objs)} \t Avg {novelty_mean} \t Std {novelty_std} \t')

    if hof is not None:
        print("============")
        print("Hall Of Fame")
        print("============")
        for i, hero in enumerate(hof):
            print(f'Index:{hero.index} \t Novelty:{hero.fitness.values[1]} \t Formation energy:{hero.fitness.values[0]} \t History:{hero.history}')

    if len(energy_hof) != 0:
        print('===================')
        print('Energy Hall of Fame')
        print('===================')
        for hero in energy_hof:
            print(f'Index:{hero.index} \t Novelty:{hero.fitness.values[1]} \t Formation energy:{hero.fitness.values[0]} \t History:{hero.history}')

    return


# write the offspring info
def write_offspring(offspring, gen, offspring_info_file):
    with open(offspring_info_file, "a") as f:
        f.write("-"*60 + "\n")
        f.write("Generation: %d" % gen)
        f.write("\n")
        for ind in offspring:
            f.write(f'Index:{ind.index} \t Novelty:{ind.fitness.values[1]} \t Formation energy:{ind.fitness.values[0]} \t History:{ind.history}')
            f.write("\n")


# write population info
def write_pool(population, gen, pool_evolution_file):
    with open(pool_evolution_file, "a") as f:
        f.write("-"*60 + "\n")
        f.write("Generation: %d" % gen)
        f.write("\n")
        for ind in population:
            f.write(f'Index:{ind.index} \t Novelty:{ind.fitness.values[1]} \t Formation energy:{ind.fitness.values[0]} \t History:{ind.history}')
            f.write("\n")


# Check if two individuals are same using graph isomorphism check
def check_redundancy(ind1: [[float]], ind2: [[float]]) -> bool:
    try:
        return nx.is_isomorphic(ind1.struct, ind2.struct)
    except (ValueError, AttributeError):
        return False


# replace the individual by a new random structure
def replace(ind, interface_len, min_width, max_width, max_height, cell_height, min_atom_num, max_atom_num,
            atom_type_list):
    ind.clear()
    # decide the cell size
    width = min_width + (max_width - min_width) * random.random()
    atom_num = random.randint(min_atom_num, max_atom_num)
    ind.append([interface_len, width, cell_height])

    # decide the atom range in z direction in frational coordinates
    bottom_limit = (cell_height / 2 - max_height / 2) / cell_height
    upper_limit = (cell_height / 2 + max_height / 2) / cell_height

    for atom in range(atom_num):
        ind.append([random.random(), random.random(), random.uniform(bottom_limit, upper_limit),
                    random.choice(atom_type_list)])
    return ind


# check if the structure's atom number satisfy the atom number constraints
def check_constrain(ind, left_atom_obj, right_atom_obj, min_num, max_num, max_height, max_coord_num, cutoff,
                    inter_atom_limit, pbc=[True, False, False]):
    # check the atom number constrain
    atom_num = len(ind) - 1
    if min_num <= atom_num <= max_num:
        # check the height constrains
        temp_copy = deepcopy(ind)
        cell_height = temp_copy[0][2]
        del temp_copy[0]
        ind_matrix = np.array(temp_copy)
        frac_max_height = np.amax(ind_matrix, axis=0)[2]
        frac_min_height = np.amin(ind_matrix, axis=0)[2]
        height_diff = cell_height * (frac_max_height - frac_min_height)
        if height_diff > max_height:
            print('height')
            return False
        else:
            # check the coordination number for all atoms
            graph, atom_obj, _ = ind.struct, ind.atom_obj, ind.rotate_obj
            atom_obj.set_pbc(pbc)  # only along the GB or interface direction, pbc considered
            nl = neighbor_list('i', atom_obj, cutoff)
            coord_num = np.bincount(nl)
            max_coord = coord_num.max()  # get the largest coordination number
            if max_coord > max_coord_num:
                print('coord')
                return False

            if not nx.is_connected(graph):
                print('not connected')
                return False
            #if nx.has_bridges(graph):
                #print('has bridge')
                #return False
            # finally check the inter-atom distances
            # get the distance matrix
            dist_mat = atom_obj.get_all_distances(mic=True)  # minimum-image convention used
            atomic_num = atom_obj.get_atomic_numbers()

            # check if the distance smaller than the covalent bond length
            atom_num = dist_mat.shape[0]
            iu = np.triu_indices(atom_num, k=1)  # the dist_mat is an symmetric matrix, diagonal are zeros

            # determine whether use the default covalent radius from ase
            # or user defined inter-atomic distance limitation
            try:
                if 'covalent radius' in inter_atom_limit:
                    use_covalent_dist = True
                else:
                    raise TypeError('The inter atom limit parameter should either be covalent radius or an integer!')
            except TypeError:
                use_covalent_dist = False

            for i in range(len(iu[0])):
                if use_covalent_dist:
                    dist_limit = covalent_radii[atomic_num[iu[0][i]]] + \
                                    covalent_radii[atomic_num[iu[1][i]]]
                else:
                    dist_limit = inter_atom_limit
                real_dist = dist_mat[iu[0][i], iu[1][i]]
                if real_dist < dist_limit:
                    print(real_dist, dist_limit)
                    #print('distance')
                    return False
            return True
    else:
        return False


if __name__ == '__main__':
    from ase.io import read, write
    from ind_manipulation import ind_creator_dist_control

    a = read(r'/Users/randy/multi_obj_ga/bp1212_multiobj_periodic/19740/all.atom', format='lammps-dump-text')
    nl = neighbor_list('i', a, 2.6)
    coord_num = np.bincount(nl)
    max_coord = coord_num.max()  # get the largest coordination number
    print(max_coord)

    #left_atom_obj = read(r'/Users/randy/gnn_dataset/12left', format='vasp')
    #right_atom_obj = read(r'/Users/randy/gnn_dataset/12right', format='vasp')
    #ind = ind_creator_dist_control(8.761141, 4.5, 8, 2.5, 16, 8, 15, [15, 15], left_atom_obj, right_atom_obj)

    #s, atom_obj = check_constrain(ind, left_atom_obj, right_atom_obj, 8, 15, 2.5, 4, 2.6, pbc=[True, False, False])
    #print(s)

    #atom_obj.write(r'/Users/randy/GA_interface/bp1212_gnn_check_coord_25h/test', format='vasp')
