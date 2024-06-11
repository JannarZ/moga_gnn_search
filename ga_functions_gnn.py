# Necessary functions for performing a genetic algorithm
import random
import pickle
from typing import List, Tuple, Any
import networkx as nx
import numpy as np
from copy import deepcopy
from ind_manipulation import attach
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

# Randomly select individuals from the population
def random_select(population: List[Any], num_sel: int) -> List[Any]:
    """
    Randomly select a specified number of individuals from the population.
    
    :param population: List of individuals in the population.
    :param num_sel: Number of individuals to select.
    :return: List of selected individuals.
    """
    return random.sample(population, num_sel)

# Roulette wheel selection (to be implemented)
def roulette_select(population: List[Any], num_sel: int) -> List[Any]:
    """
    Select individuals from the population using the roulette wheel method.
    
    :param population: List of individuals in the population.
    :param num_sel: Number of individuals to select.
    :return: List of selected individuals.
    """
    pass  # Implementation needed

def restart(restart_file: str) -> Tuple[int, List[Any], int]:
    """
    Restart the genetic algorithm from a saved checkpoint.
    
    :param restart_file: Path to the restart file.
    :return: Generation number, population, and index.
    """
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

# Global index value for assigning indices to individuals
index = 0

def assign_index(ind: Any) -> None:
    """
    Assign a unique index to an individual.
    
    :param ind: Individual to assign the index to.
    """
    global index
    index += 1
    ind.index = index

def write_hof(hof_file: str, gen: int, hof: List[Any], energy_hof: List[Any]) -> None:
    """
    Write the Hall of Fame (HOF) to a file.
    
    :param hof_file: Path to the HOF file.
    :param gen: Current generation number.
    :param hof: List of Hall of Fame individuals.
    :param energy_hof: List of energy Hall of Fame individuals.
    """
    with open(hof_file, 'a+') as f_obj:
        f_obj.write(f'Generation: {gen}\n')
        for hero in hof:
            f_obj.write(
                f'Index:{hero.index} \t Novelty:{hero.fitness.values[1]:.5f} \t '
                f'Formation energy:{hero.fitness.values[0]:.5f} \t History:{hero.history}\n'
            )
        f_obj.write('Energy Hall of Fame:\n')
        for hero in energy_hof:
            f_obj.write(
                f'Index: {hero.index} \t Novelty: {hero.fitness.values[1]:.5f} \t '
                f'Formation energy: {hero.fitness.values[0]:.5f} \t History: {hero.history}\n'
            )

def print_gen(population: List[Any], gen: int, best_objective_file: str, energy_hof: List[Any], hof: List[Any] = None) -> None:
    """
    Print information about each generation.
    
    :param population: Current population.
    :param gen: Current generation number.
    :param best_objective_file: Path to the file for storing the best objectives.
    :param energy_hof: List of energy Hall of Fame individuals.
    :param hof: List of Hall of Fame individuals (optional).
    """
    # Gather all the fitnesses in one list
    fitnesses = [ind.fitness.values for ind in population]
    length = len(population)
    mean = sum(fitnesses) / length
    sum2 = sum(x**2 for x in fitnesses)
    std = abs(sum2 / length - mean**2)**0.5
    
    # Print statistics for the current generation
    print(f"-- Generation {gen} --")
    print(f"  Min {min(fitnesses):.5f}")
    print(f"  Max {max(fitnesses):.5f}")
    print(f"  Avg {mean:.5f}")
    print(f"  Std {std:.5f}")
    
    # Save the best individual in the best objective file
    with open(best_objective_file, 'a+') as f:
        best = max(population, key=lambda ind: ind.fitness.values[0])
        f.write(f'{gen}\t{best.fitness.values[0]}\n')

# Add more functions here as needed for the genetic algorithm process
