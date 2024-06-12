import os
import shutil
import subprocess
import numpy as np
import networkx as nx
from ase import Atoms
from ase.io import read
from os.path import dirname, join
from copy import deepcopy

# Import the neural network packages
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def select_k(spectrum: np.ndarray, minimum_energy: float = 0.9) -> int:
    """
    Determine the number of eigenvalues k from the eigenvalue spectrum.

    Parameters:
    spectrum (np.ndarray): The eigenvalue spectrum.
    minimum_energy (float): The minimum cumulative energy fraction to retain. Default is 0.9.

    Returns:
    int: The number of eigenvalues k that retains the desired minimum energy fraction.
    """
    running_total = 0.0
    total = sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    for i in range(len(spectrum)):
        running_total += spectrum[i]
        if running_total / total >= minimum_energy:
            return i + 1
    return len(spectrum)

def eigen_sim(ind_1: list, ind_2: list) -> float:
    """
    Calculate the similarity between two graphs using the eigenvector similarity.

    Parameters:
    ind_1 (list): The first individual containing the graph structure.
    ind_2 (list): The second individual containing the graph structure.

    Returns:
    float: The similarity score between the two graphs.
    """
    laplacian1 = nx.spectrum.laplacian_spectrum(ind_1.struct)
    laplacian2 = nx.spectrum.laplacian_spectrum(ind_2.struct)

    k1 = select_k(laplacian1)
    k2 = select_k(laplacian2)

    k = min(k1, k2)

    return sum((laplacian1[:k] - laplacian2[:k]) ** 2)

def struc2graph(file: str, cutoff: float, format: str = None, pbc: list = [True, False, False], atom_force: bool = False, get_energy: bool = False) -> nx.Graph:
    """
    Convert a structure file to a graph object.

    Parameters:
    file (str): Path to the structure file.
    cutoff (float): Cutoff distance for creating edges in the graph.
    format (str): Format of the structure file. Default is None.
    pbc (list): Periodic boundary conditions. Default is [True, False, False].
    atom_force (bool): Whether to read atom forces from a file. Default is False.
    get_energy (bool): Whether to read total energy from OUTCAR file. Default is False.

    Returns:
    nx.Graph: A graph object representing the structure.
    """
    # Get the total energy, which is the target for NN to train
    if get_energy:
        outcar_path = join(dirname(file), 'OUTCAR')
        with open(outcar_path, 'r') as f:
            out_lines = f.readlines()
        for line in out_lines:
            if 'energy(sigma->0) =' in line:
                tot_e = float(line.split()[-1])
                break
    else:
        tot_e = None

    # Create the graph object
    graph = nx.Graph(pbc=pbc, y=tot_e)

    # Get the positions for all the atoms in the file
    # ASE can auto detect the file format
    atom_obj = read(file, format=format) if format else read(file)

    # If read atom force, try to read atom force from force_on_atom file
    if atom_force:
        try:
            with open(join(dirname(file), 'force_on_atom_detail'), 'r') as f:
                f_lines = f.readlines()
        except FileNotFoundError:
            print('Cannot read force_on_atom_detail file in directory.')
            f_lines = []

    # Make cell a one-dimensional vector
    cell = np.array([atom_obj.get_cell()[i][i] for i in range(3)])
    ori_atom_pos = atom_obj.get_positions()

    # Add the cell to the graph object
    graph.graph['cell'] = cell

    # Calculate the pairs within the bond length cutoff for the original cell
    atom_num = len(ori_atom_pos)
    for i in range(atom_num):
        feature_vector = np.append(cell, ori_atom_pos[i])
        if atom_force and f_lines:
            force_vector = [float(f_lines[i + 3].split()[j]) for j in range(12)]
            graph.add_node(i, x=np.append(feature_vector, force_vector))
        else:
            graph.add_node(i, x=feature_vector)

        for j in range(atom_num):
            distance = np.linalg.norm(ori_atom_pos[i] - ori_atom_pos[j])
            if distance <= cutoff and distance != 0:
                graph.add_edge(i, j, distance=distance)

    # Consider the atoms in the neighbor cell in x direction
    if pbc[0]:
        x_adj_atom_pos = deepcopy(ori_atom_pos)
        x_adj_atom_pos[:, 0] += cell[0]
        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(ori_atom_pos[i] - x_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    graph.add_edge(i, j, distance=distance)

    # Consider the atoms in the neighbor cell in y direction
    if pbc[1]:
        y_adj_atom_pos = deepcopy(ori_atom_pos)
        y_adj_atom_pos[:, 1] += cell[1]
        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(ori_atom_pos[i] - y_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    graph.add_edge(i, j, distance=distance)

    # Consider the atoms in the neighbor cell in z direction
    if pbc[2]:
        z_adj_atom_pos = deepcopy(ori_atom_pos)
        z_adj_atom_pos[:, 2] += cell[2]
        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(ori_atom_pos[i] - z_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    graph.add_edge(i, j, distance=distance)

    return graph

def ged(ind_1: list, ind_2: list, cutoff: float = 2.6, pbc: list = [True, False, False]) -> int:
    """
    Calculate the novelty (graph edit distance) between two individuals.

    Parameters:
    ind_1 (list): The first individual.
    ind_2 (list): The second individual.
    cutoff (float): Cutoff distance for creating edges in the graph. Default is 2.6.
    pbc (list): Periodic boundary conditions. Default is [True, False, False].

    Returns:
    int: The graph edit distance between the two individuals.
    """
    # Create the graph objects for two individuals
    g_1 = nx.Graph()
    g_2 = nx.Graph()

    # Process the first individual
    ind_1_cell = np.array(ind_1[0])
    ind_1_positions = np.delete(np.array(ind_1[1:]), -1, 1) * ind_1_cell
    atom_num_1 = len(ind_1_positions)

    for i in range(atom_num_1):
        g_1.add_node(i)
        for j in range(atom_num_1):
            distance = np.linalg.norm(ind_1_positions[i] - ind_1_positions[j])
            if distance <= cutoff and distance != 0:
                g_1.add_edge(i, j)

    # Consider the atoms in neighbor cell (x direction) for the first individual
    if pbc[0]:
        x_adj_atom_pos = deepcopy(ind_1_positions)
        x_adj_atom_pos[:, 0] += ind_1_cell[0]
        for i in range(atom_num_1):
            for j in range(atom_num_1):
                distance = np.linalg.norm(ind_1_positions[i] - x_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_1.add_edge(i, j)

    # Consider the atoms in neighbor cell (y direction) for the first individual
    if pbc[1]:
        y_adj_atom_pos = deepcopy(ind_1_positions)
        y_adj_atom_pos[:, 1] += ind_1_cell[1]
        for i in range(atom_num_1):
            for j in range(atom_num_1):
                distance = np.linalg.norm(ind_1_positions[i] - y_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_1.add_edge(i, j)

    # Consider the atoms in neighbor cell (z direction) for the first individual
    if pbc[2]:
        z_adj_atom_pos = deepcopy(ind_1_positions)
        z_adj_atom_pos[:, 2] += ind_1_cell[2]
        for i in range(atom_num_1):
            for j in range(atom_num_1):
                distance = np.linalg.norm(ind_1_positions[i] - z_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_1.add_edge(i, j)

    # Process the second individual
    ind_2_cell = np.array(ind_2[0])
    ind_2_positions = np.delete(np.array(ind_2[1:]), -1, 1) * ind_2_cell
    atom_num_2 = len(ind_2_positions)

    for i in range(atom_num_2):
        g_2.add_node(i)
        for j in range(atom_num_2):
            distance = np.linalg.norm(ind_2_positions[i] - ind_2_positions[j])
            if distance <= cutoff and distance != 0:
                g_2.add_edge(i, j)

    # Consider the atoms in neighbor cell (x direction) for the second individual
    if pbc[0]:
        x_adj_atom_pos = deepcopy(ind_2_positions)
        x_adj_atom_pos[:, 0] += ind_2_cell[0]
        for i in range(atom_num_2):
            for j in range(atom_num_2):
                distance = np.linalg.norm(ind_2_positions[i] - x_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_2.add_edge(i, j)

    # Consider the atoms in neighbor cell (y direction) for the second individual
    if pbc[1]:
        y_adj_atom_pos = deepcopy(ind_2_positions)
        y_adj_atom_pos[:, 1] += ind_2_cell[1]
        for i in range(atom_num_2):
            for j in range(atom_num_2):
                distance = np.linalg.norm(ind_2_positions[i] - y_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_2.add_edge(i, j)

    # Consider the atoms in neighbor cell (z direction) for the second individual
    if pbc[2]:
        z_adj_atom_pos = deepcopy(ind_2_positions)
        z_adj_atom_pos[:, 2] += ind_2_cell[2]
        for i in range(atom_num_2):
            for j in range(atom_num_2):
                distance = np.linalg.norm(ind_2_positions[i] - z_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_2.add_edge(i, j)

    return nx.graph_edit_distance(g_1, g_2)

def energy_calculate_gnn(ind: object, pop: list, hof: list, model: object, left_e_per_atom: float, right_e_per_atom: float, left_atom_obj: Atoms, right_atom_obj: Atoms, interface_len: float, k: int, time_limit: float) -> None:
    """
    Calculate the energy of the complete graph using a Graph Neural Network (GNN).

    Parameters:
    ind (object): The individual whose energy is being calculated.
    pop (list): The population of individuals.
    hof (list): The Hall of Fame individuals.
    model (object): The trained GNN model.
    left_e_per_atom (float): Energy per atom for the left material.
    right_e_per_atom (float): Energy per atom for the right material.
    left_atom_obj (Atoms): ASE Atoms object for the left material.
    right_atom_obj (Atoms): ASE Atoms object for the right material.
    interface_len (float): Length of the interface.
    k (int): Number of nearest neighbors to consider for novelty calculation.
    time_limit (float): Time limit for calculations.

    Returns:
    None
    """
    # Get the left and right atom numbers from the individual
    left_atomic_num = left_atom_obj.get_atomic_numbers()[0]
    right_atomic_num = right_atom_obj.get_atomic_numbers()[0]

    # Generate the PyTorch geometric object
    nx_graph, atom_obj, rotate_obj = ind.struct, ind.atom_obj, ind.rotate_obj

    # If the middle part is null, set fitness to 100
    if nx_graph is None:
        # Define left and right atom number for the case of attach failed
        right_atom_num = right_atom_obj.get_global_number_of_atoms()
        left_atom_num = left_atom_obj.get_global_number_of_atoms()

        # Save the structure to the disk
        pwd = os.getcwd()
        job_dir = join(pwd, str(ind.index))
        os.mkdir(job_dir)
        with open(join(job_dir, 'log.gnn'), 'w') as f:
            f.write('Middle part has no atom!\n')
            f.write(f'Left atom type: {left_atomic_num}\n')
            f.write(f'Right atom type: {right_atomic_num}\n')
            f.write(f'Left atom number: {left_atom_num}\n')
            f.write(f'Right atom number: {right_atom_num}\n')
            f.write(f'Left energy per atom: {left_e_per_atom}\n')
            f.write(f'Right energy per atom: {right_e_per_atom}\n')

        # Assign total energy and fitness to the individual
        ind.formation_e = None
        ind.fitness.values = (100, -100)
        print(f'Setting novelty of individual {ind.index} to {ind.fitness} due to null middle part!')

        return

    # Get the atom numbers for left and right side atoms
    left_atom_num = np.count_nonzero(atom_obj.get_atomic_numbers() == left_atomic_num)
    right_atom_num = np.count_nonzero(atom_obj.get_atomic_numbers() == right_atomic_num)

    # Create the Atoms object for the individual
    ind_cell = np.array(ind[0])
    ind_position = ind_cell * np.delete(np.array(ind[1:]), -1, axis=1)
    ind_atomic_numbers = np.array(ind[1:])[:, 2]
    ind_atom_obj = Atoms(ind_atomic_numbers, positions=ind_position, cell=ind_cell)

    # Create the folder for storing the structures
    pwd = os.getcwd()
    job_dir = join(pwd, str(ind.index))
    os.mkdir(job_dir)

    # Write the Lammps input structure to the folder
    atom_obj.write(join(job_dir, 'in.atom'), format='lammps-data')
    ind_atom_obj.write(join(job_dir, 'middle'), format='lammps-data')
    rotate_obj.write(join(job_dir, 'rotate'), format='lammps-data')

    # Use GNN to get the energy of the system
    pygeo_graph = from_networkx(nx_graph)

    # Use DataLoader to generate the batch with batch_size = 1
    nominal_dataset = DataLoader([pygeo_graph], batch_size=1, shuffle=False)

    # Use the model to evaluate total energy
    tot_e = None
    for data in nominal_dataset:
        tot_e, _ = model(data)
    tot_e = tot_e.item()

    # Calculate the formation energy
    if tot_e != 10000:
        if left_atomic_num == right_atomic_num:
            formation_e = (tot_e - left_atom_num * left_e_per_atom) / (2 * interface_len)
        else:
            formation_e = (tot_e - left_atom_num * left_e_per_atom - right_atom_num * right_e_per_atom) / (2 * interface_len)
    else:
        formation_e = 100

    # Calculate the fitness according to the novelty (graph edit distance)
    # Select some of the best structures from hof
    selected_hof = hof[:len(pop)]
    eigen_sim_list = []
    for individual in pop:
        try:
            eigen_sim_list.append(eigen_sim(ind, individual))
        except nx.exception.NetworkXError:
            eigen_sim_list.append(abs(len(ind) - len(individual)))  # Use the difference of atom number as similarity when no atoms in structure
        except AttributeError:
            eigen_sim_list.append(100)
    for individual in selected_hof:
        try:
            eigen_sim_list.append(eigen_sim(ind, individual))
        except nx.exception.NetworkXError:
            eigen_sim_list.append(abs(len(ind) - len(individual)))
        except AttributeError:
            eigen_sim_list.append(100)

    eigen_sim_list.sort()  # Sort the similarity
    eigen_sim_list = eigen_sim_list[:k]  # Pick out k nearest neighbors

    fitness = sum(eigen_sim_list) / len(eigen_sim_list)

    # Save the structure to the disk and write a log file
    with open(join(job_dir, 'log.gnn'), 'w') as f:
        f.write(f'Total energy of structure: {tot_e}\n')
        f.write(f'Left atom type: {left_atomic_num}\n')
        f.write(f'Right atom type: {right_atomic_num}\n')
        f.write(f'Left atom number: {left_atom_num}\n')
        f.write(f'Right atom number: {right_atom_num}\n')
        f.write(f'Interface length: {interface_len}\n')
        f.write(f'Left energy per atom: {left_e_per_atom}\n')
        f.write(f'Right energy per atom: {right_e_per_atom}\n')

    # Assign total energy and the fitness to the individual
    ind.formation_e = formation_e
    ind.fitness.values = (formation_e, fitness)
    print(f'Setting novelty and formation energy of individual {ind.index} to {ind.fitness.values[1]} & {ind.fitness.values[0]}')

def rotate_180(input_file: str, output_file: str, xcell_line: int, atom_line: int) -> float:
    """
    Rotate the atomic positions in the input file by 180 degrees and write to the output file.

    Parameters:
    input_file (str): The path to the input file containing atomic positions.
    output_file (str): The path to the output file to write the rotated positions.
    xcell_line (int): The line number of the x-cell dimension in the input file.
    atom_line (int): The starting line number of the atomic positions in the input file.

    Returns:
    float: The y-cell dimension.
    """
    # Read input file cell and atom info
    with open(input_file, 'r') as f:
        input_lines = f.readlines()

    # Get the middle point and cell dimensions
    x_cell = float(input_lines[xcell_line - 1].split()[1])
    y_cell = float(input_lines[xcell_line].split()[1])
    z_cell = float(input_lines[xcell_line + 1].split()[1])
    mid_x = x_cell / 2
    mid_y = y_cell / 2
    mid_z = z_cell / 2

    # Get the atom position matrix
    matrix = []
    atom_lines = input_lines[atom_line - 1:]
    for line in atom_lines:
        if line.strip():
            atom_coords = line.split()[:-3] + [float(coord) for coord in line.split()[-3:]]
            matrix.append(atom_coords)

    # Calculate the atom positions after the rotation
    for atom in matrix:
        original_x, original_y, original_z = atom[-3], atom[-2], atom[-1]
        rotate_x = 2 * mid_x - original_x
        rotate_y = 2 * mid_y - original_y
        rotate_z = 2 * mid_z - original_z
        atom[-3], atom[-2], atom[-1] = rotate_x, rotate_y, rotate_z

    # Write the rotated atom positions to the output file
    pre_lines = input_lines[:atom_line - 1]
    with open(output_file, 'w') as f:
        for line in pre_lines:
            f.write(line)
        for atom in matrix:
            f.write(' '.join(map(str, atom)) + '\n')

    return y_cell

def get_tot_e_static(output_path: str) -> float:
    """
    Get the total energy from a static LAMMPS run.

    Parameters:
    output_path (str): The path to the LAMMPS output log file.

    Returns:
    float: The total energy read from the log file.
    """
    # Read the log.lammps file as a list of strings
    with open(output_path, 'r') as f:
        lines = f.readlines()

    # Initialize total energy variable
    tot_energy = None

    # Keywords to identify the line with energy values
    match_strings = ['Step', 'Temp', 'E_pair', 'E_mol', 'TotEng']

    # Find the line with the keywords and extract the total energy from the next line
    for i in range(len(lines)):
        if all(match in lines[i] for match in match_strings):
            tot_energy = float(lines[i + 1].split()[4])
            break

    # Check if total energy was found
    if tot_energy is None:
        raise ValueError("Total energy not found in the log file.")

    return tot_energy
