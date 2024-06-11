# Some necessary functions for performing a genetic algorithm

# Import needed modules
import random
import pickle
import networkx as nx
import numpy as np
from copy import deepcopy
from ind_manipulation import attach
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list


def random_select(population: list, num_sel: int) -> list:
    """
    Randomly select individuals from the population.

    Parameters:
    population (list): The population to select from.
    num_sel (int): The number of individuals to select.

    Returns:
    list: A list of randomly selected individuals.
    """
    return random.sample(population, num_sel)

def roulette_select(population: list, num_sel: int) -> list:
    """
    Select individuals from the population using roulette wheel selection.

    Parameters:
    population (list): The population to select from.
    num_sel (int): The number of individuals to select.

    Returns:
    list: A list of selected individuals.
    """
    pass # Need implementation 


def restart(restart_file: str) -> tuple:
    """
    Restart the genetic algorithm from a saved checkpoint.

    Parameters:
    restart_file (str): The path to the restart file.

    Returns:
    tuple: A tuple containing the generation number, the population, and the index.
    """
    # Load the restart file
    with open(restart_file, "rb") as f:
        cp = pickle.load(f)
    
    gen = cp["generation"]
    population = cp["population"]
    random.setstate(cp['rndstate'])
    index = cp['index']
    old_size = len(population)

    print(f"Restart from {restart_file}, gen {gen} (population size {old_size})")
    print('-' * 60)
    print(f"-- Generation {gen} --")
    
    return gen, population, index


# Index value must exist for assign_index function
index = 0

def assign_index(ind: object) -> None:
    """
    Assign an index to the individual.

    Parameters:
    ind (object): The individual to assign an index to.

    Returns:
    None
    """
    global index
    index += 1
    ind.index = index

def write_hof(hof_file: str, gen: int, hof: list, energy_hof: list) -> None:
    """
    Write the Hall of Fame (HOF) information to a file.

    Parameters:
    hof_file (str): The path to the HOF file.
    gen (int): The current generation number.
    hof (list): The list of Hall of Fame individuals.
    energy_hof (list): The list of energy Hall of Fame individuals.

    Returns:
    None
    """
    with open(hof_file, 'a+') as f_obj:
        f_obj.write(f'Generation: {gen}\n')
        
        for hero in hof:
            f_obj.write(f'Index: {hero.index} \t Novelty: {hero.fitness.values[1]:.5f} \t '
                        f'Formation energy: {hero.fitness.values[0]:.5f} \t History: {hero.history}\n')
        
        f_obj.write('Energy Hall of Fame:\n')
        
        for hero in energy_hof:
            f_obj.write(f'Index: {hero.index} \t Novelty: {hero.fitness.values[1]:.5f} \t '
                        f'Formation energy: {hero.fitness.values[0]:.5f} \t History: {hero.history}\n')

def print_gen(population: list, gen: int, best_objective_file: str, energy_hof: list, hof: list = None) -> None:
    """
    Print each generation's information and write the best objectives to a file.

    Parameters:
    population (list): The population of individuals.
    gen (int): The current generation number.
    best_objective_file (str): The file to write the best objectives.
    energy_hof (list): The list of energy Hall of Fame individuals.
    hof (list): The list of Hall of Fame individuals. Default is None.

    Returns:
    None
    """
    # Gather all the fitnesses in one list and print the stats
    energy_objs = [ind.fitness.values[0] for ind in population]
    novelty_objs = [ind.fitness.values[1] for ind in population]
    pop_size = len(population)

    # Calculate the statistical information for formation energy
    energy_mean = sum(energy_objs) / pop_size
    energy_sum2 = sum(x * x for x in energy_objs)
    energy_std = abs(energy_sum2 / pop_size - energy_mean ** 2) ** 0.5

    # Calculate the statistical information for novelty
    novelty_mean = sum(novelty_objs) / pop_size
    novelty_sum2 = sum(x * x for x in novelty_objs)
    novelty_std = abs(novelty_sum2 / pop_size - novelty_mean ** 2) ** 0.5

    # Write the statistical information to the objective file and screen
    with open(best_objective_file, "a") as f:
        f.write(f'{gen} \t {min(energy_objs)} \t {max(novelty_objs)}\n')
    
    print(f'Energy: Min: {min(energy_objs)} \t Max: {max(energy_objs)} \t Avg: {energy_mean} \t Std: {energy_std}')
    print(f'Novelty: Min: {min(novelty_objs)} \t Max: {max(novelty_objs)} \t Avg: {novelty_mean} \t Std: {novelty_std}')

    if hof is not None:
        print("============")
        print("Hall Of Fame")
        print("============")
        for hero in hof:
            print(f'Index: {hero.index} \t Novelty: {hero.fitness.values[1]} \t Formation energy: {hero.fitness.values[0]} \t History: {hero.history}')

    if len(energy_hof) != 0:
        print('===================')
        print('Energy Hall of Fame')
        print('===================')
        for hero in energy_hof:
            print(f'Index: {hero.index} \t Novelty: {hero.fitness.values[1]} \t Formation energy: {hero.fitness.values[0]} \t History: {hero.history}')

def write_offspring(offspring: list, gen: int, offspring_info_file: str) -> None:
    """
    Write the offspring information to a file.

    Parameters:
    offspring (list): The list of offspring individuals.
    gen (int): The current generation number.
    offspring_info_file (str): The file to write the offspring information.

    Returns:
    None
    """
    with open(offspring_info_file, "a") as f:
        f.write("-" * 60 + "\n")
        f.write(f"Generation: {gen}\n")
        for ind in offspring:
            f.write(f'Index: {ind.index} \t Novelty: {ind.fitness.values[1]} \t '
                    f'Formation energy: {ind.fitness.values[0]} \t History: {ind.history}\n')

def write_pool(population: list, gen: int, pool_evolution_file: str) -> None:
    """
    Write the population information to a file.

    Parameters:
    population (list): The population of individuals.
    gen (int): The current generation number.
    pool_evolution_file (str): The file to write the population information.

    Returns:
    None
    """
    with open(pool_evolution_file, "a") as f:
        f.write("-" * 60 + "\n")
        f.write(f"Generation: {gen}\n")
        for ind in population:
            f.write(f'Index: {ind.index} \t Novelty: {ind.fitness.values[1]} \t '
                    f'Formation energy: {ind.fitness.values[0]} \t History: {ind.history}\n')

def check_redundancy(ind1: list, ind2: list) -> bool:
    """
    Check if two individuals are the same using graph isomorphism.

    Parameters:
    ind1 (list): The first individual to check.
    ind2 (list): The second individual to check.

    Returns:
    bool: True if the individuals are isomorphic (same), False otherwise.
    """
    try:
        return nx.is_isomorphic(ind1.struct, ind2.struct)
    except (ValueError, AttributeError):
        return False

def replace(ind: list, interface_len: float, min_width: float, max_width: float, max_height: float, cell_height: float, min_atom_num: int, max_atom_num: int, atom_type_list: list) -> list:
    """
    Replace the individual with a new random structure.

    Parameters:
    ind (list): The individual to be replaced.
    interface_len (float): The length of the interface.
    min_width (float): The minimum width of the interface.
    max_width (float): The maximum width of the interface.
    max_height (float): The maximum height of the interface.
    cell_height (float): The height of the cell.
    min_atom_num (int): The minimum number of atoms.
    max_atom_num (int): The maximum number of atoms.
    atom_type_list (list): The list of possible atom types.

    Returns:
    list: The new individual with a random structure.
    """
    ind.clear()

    # Decide the cell size
    width = min_width + (max_width - min_width) * random.random()
    atom_num = random.randint(min_atom_num, max_atom_num)
    ind.append([interface_len, width, cell_height])

    # Decide the atom range in z direction in fractional coordinates
    bottom_limit = (cell_height / 2 - max_height / 2) / cell_height
    upper_limit = (cell_height / 2 + max_height / 2) / cell_height

    for _ in range(atom_num):
        ind.append([random.random(), random.random(), random.uniform(bottom_limit, upper_limit),
                    random.choice(atom_type_list)])

    return ind

def check_constrain(ind: list, left_atom_obj: Atoms, right_atom_obj: Atoms, min_num: int, max_num: int, max_height: float, max_coord_num: int, cutoff: float, inter_atom_limit: float, pbc: list = [True, False, False]) -> bool:
    """
    Check if the structure's atom number satisfies the atom number constraints.

    Parameters:
    ind (list): The individual to check.
    left_atom_obj (Atoms): Left atom object.
    right_atom_obj (Atoms): Right atom object.
    min_num (int): Minimum number of atoms.
    max_num (int): Maximum number of atoms.
    max_height (float): Maximum allowable height difference.
    max_coord_num (int): Maximum coordination number.
    cutoff (float): Cutoff distance for neighbor list.
    inter_atom_limit (float): Inter-atomic distance limit.
    pbc (list): Periodic boundary conditions. Default is [True, False, False].

    Returns:
    bool: True if the individual satisfies the constraints, False otherwise.
    """
    # Check the atom number constraint
    atom_num = len(ind) - 1
    if min_num <= atom_num <= max_num:
        # Check the height constraints
        temp_copy = deepcopy(ind)
        cell_height = temp_copy[0][2]
        del temp_copy[0]
        ind_matrix = np.array(temp_copy)
        frac_max_height = np.amax(ind_matrix, axis=0)[2]
        frac_min_height = np.amin(ind_matrix, axis=0)[2]
        height_diff = cell_height * (frac_max_height - frac_min_height)
        if height_diff > max_height:
            return False
        else:
            # Check the coordination number for all atoms
            graph, atom_obj, _ = ind.struct, ind.atom_obj, ind.rotate_obj
            atom_obj.set_pbc(pbc)  # Only along the GB or interface direction, PBC considered
            nl = neighbor_list('i', atom_obj, cutoff)
            coord_num = np.bincount(nl)
            max_coord = coord_num.max()  # Get the largest coordination number
            if max_coord > max_coord_num:
                return False

            if not nx.is_connected(graph):
                return False

            # Finally check the inter-atom distances
            # Get the distance matrix
            dist_mat = atom_obj.get_all_distances(mic=True)  # Minimum-image convention used
            atomic_num = atom_obj.get_atomic_numbers()

            # Check if the distance is smaller than the covalent bond length
            atom_num = dist_mat.shape[0]
            iu = np.triu_indices(atom_num, k=1)  # The dist_mat is a symmetric matrix, diagonals are zeros

            # Determine whether to use the default covalent radius from ASE or user-defined inter-atomic distance limitation
            try:
                if 'covalent radius' in inter_atom_limit:
                    use_covalent_dist = True
                else:
                    raise TypeError('The inter-atom limit parameter should either be covalent radius or an integer!')
            except TypeError:
                use_covalent_dist = False

            for i in range(len(iu[0])):
                if use_covalent_dist:
                    dist_limit = covalent_radii[atomic_num[iu[0][i]]] + covalent_radii[atomic_num[iu[1][i]]]
                else:
                    dist_limit = inter_atom_limit
                real_dist = dist_mat[iu[0][i], iu[1][i]]
                if real_dist < dist_limit:
                    return False
            return True
    else:
        return False


if __name__ == '__main__':
    pass

