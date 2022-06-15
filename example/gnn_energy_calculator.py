# functions for precting energy using GNN

import os
import shutil
import subprocess
import numpy as np
import networkx as nx
from ase import Atoms
from ase.io import read
from os.path import dirname, join
from copy import deepcopy

# import the neural network packages
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# function that determine the number k from eigenvalue spectrum
def select_k(spectrum: np.array, minimum_energy: float = 0.9) -> int:
    running_total = 0.0
    total = sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    for i in range(len(spectrum)):
        running_total += spectrum[i]
        if running_total / total >= minimum_energy:
            return i + 1
    return len(spectrum)


# function that calculate the similarity between two graphs
# using the eigenvector similarity
def eigen_sim(ind_1: [[float]], ind_2: [[float]]) -> float:

    laplacian1 = nx.spectrum.laplacian_spectrum(ind_1.struct)
    laplacian2 = nx.spectrum.laplacian_spectrum(ind_2.struct)

    k1 = select_k(laplacian1)
    k2 = select_k(laplacian2)

    k = min(k1, k2)

    return sum((laplacian1[:k] - laplacian2[:k]) ** 2)


# function that transfer structure file to graph object
def struc2graph(file, cutoff, format=None, pbc=[True, False, False], atom_force=False, get_energy=False):
    # get the total energy, which is the target for NN to train
    if get_energy:
        outcar_path = join(dirname(file), 'OUTCAR')
        with open(outcar_path, 'r') as f:
            out_lines = f.readlines()
        for line in out_lines:
            if 'energy(sigma->0) =' in line:
                tot_e = float(line.split()[-1])
    else:
        tot_e = None

    # create the graph object
    graph = nx.Graph(pbc=pbc, y=tot_e)

    # get the positions for all the atoms in the file
    # ASE can auto detect the file format
    if format is not None:
        atom_obj = read(file, format=format)
    else:
        atom_obj = read(file)

    # if read atom force, try to read atom force from force_on_atom file
    if atom_force:
        try:
            with open(join(dirname(file), 'force_on_atom_detail'), 'r') as f:
                f_lines = f.readlines()
        except FileNotFoundError:
            print('Can\'t read force_on_atom_detail file in directory.')
    else:
        pass

    # make cell a one dimensional vector
    cell = []
    cell.append(atom_obj.get_cell()[0][0])
    cell.append(atom_obj.get_cell()[1][1])
    cell.append(atom_obj.get_cell()[2][2])
    cell = np.array(cell)
    ori_atom_pos = atom_obj.get_positions()

    # add the cell to the graph object
    graph.graph['cell'] = cell

    # calculate the pairs that within the bond length cutoff for the original cell
    atom_num = len(ori_atom_pos)
    for i in range(atom_num):
        feature_vector = np.append(cell, ori_atom_pos[i])
        if atom_force:
            force_vector = [float(f_lines[i + 3].split()[0]),
                            float(f_lines[i + 3].split()[1]),
                            float(f_lines[i + 3].split()[2]),
                            float(f_lines[i + 3].split()[3]),
                            float(f_lines[i + 3].split()[4]),
                            float(f_lines[i + 3].split()[5]),
                            float(f_lines[i + 3].split()[6]),
                            float(f_lines[i + 3].split()[7]),
                            float(f_lines[i + 3].split()[8]),
                            float(f_lines[i + 3].split()[9]),
                            float(f_lines[i + 3].split()[10]),
                            float(f_lines[i + 3].split()[11]),
                            ]

            graph.add_node(i, x=np.append(feature_vector, force_vector))
        else:
            graph.add_node(i, x=feature_vector)

        for j in range(atom_num):
            distance = np.sqrt((ori_atom_pos[i][0] - ori_atom_pos[j][0]) ** 2 +
                               (ori_atom_pos[i][1] - ori_atom_pos[j][1]) ** 2 +
                               (ori_atom_pos[i][2] - ori_atom_pos[j][2]) ** 2)
            if distance <= cutoff and distance != 0:
                graph.add_edge(i, j, distance=distance)

    # consider the atoms in neighbor cell
    if pbc[0]:
        x_adj_atom_pos = deepcopy(ori_atom_pos)
        x_adj_atom_pos[:, 0] = x_adj_atom_pos[:, 0] + cell[0]

        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.sqrt((ori_atom_pos[i][0] - x_adj_atom_pos[j][0]) ** 2 +
                                   (ori_atom_pos[i][1] - x_adj_atom_pos[j][1]) ** 2 +
                                   (ori_atom_pos[i][2] - x_adj_atom_pos[j][2]) ** 2)
                if distance <= cutoff and distance != 0:
                    graph.add_edge(i, j, distance=distance)

    # consider the atoms in neighbor cell y direction
    if pbc[1]:
        y_adj_atom_pos = deepcopy(ori_atom_pos)
        y_adj_atom_pos[:, 0] = y_adj_atom_pos[:, 0] + cell[1]

        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.sqrt((ori_atom_pos[i][0] - y_adj_atom_pos[j][0]) ** 2 +
                                      (ori_atom_pos[i][1] - y_adj_atom_pos[j][1]) ** 2 +
                                      (ori_atom_pos[i][2] - y_adj_atom_pos[j][2]) ** 2)
                if distance <= cutoff and distance != 0:
                    graph.add_edge(i, j, distance=distance)

    # consider the atoms in z direction neighbor cell
    if pbc[2]:
        z_adj_atom_pos = deepcopy(ori_atom_pos)
        z_adj_atom_pos[:, 0] = z_adj_atom_pos[:, 0] + cell[1]

        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.sqrt((ori_atom_pos[i][0] - z_adj_atom_pos[j][0]) ** 2 +
                                   (ori_atom_pos[i][1] - z_adj_atom_pos[j][1]) ** 2 +
                                   (ori_atom_pos[i][2] - z_adj_atom_pos[j][2]) ** 2)
                if distance <= cutoff and distance != 0:
                    graph.add_edge(i, j, distance=distance)

    return graph


# Function that calculate the novelty (graph edit distance) from two individuals (middle part)
def ged(ind_1: [[float]], ind_2: [[float]], cutoff: float = 2.6, pbc: [bool] = [True, False, False]) -> int:
    # create the graph objects for two individuals
    g_1 = nx.Graph()
    g_2 = nx.Graph()

    ind_1_cell = np.array(ind_1[0])
    ind_1_positions = np.delete(np.array(ind_1[1:]), -1, 1) * ind_1_cell

    ind_2_cell = np.array(ind_1[0])
    ind_2_positions = np.delete(np.array(ind_2[1:]), -1, 1) * ind_2_cell

    # calculate the pairs that within the bond length cutoff for the original cell for individual 1
    atom_num = len(ind_1_positions)
    for i in range(atom_num):
        g_1.add_node(i)

        for j in range(atom_num):
            distance = np.linalg.norm(ind_1_positions[i] - ind_1_positions[j])
            if distance <= cutoff and distance != 0:
                g_1.add_edge(i, j)

    # consider the atoms in neighbor cell
    if pbc[0]:
        x_adj_atom_pos = deepcopy(ind_1_positions)
        x_adj_atom_pos[:, 0] = x_adj_atom_pos[:, 0] + ind_1_cell[0]

        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(ind_1_positions[i] - x_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_1.add_edge(i, j)

    # consider the atoms in neighbor cell y direction
    if pbc[1]:
        y_adj_atom_pos = deepcopy(ind_1_positions)
        y_adj_atom_pos[:, 0] = y_adj_atom_pos[:, 0] + ind_1_cell[1]

        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(ind_1_positions[i] - y_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_1.add_edge(i, j)

    # consider the atoms in z direction neighbor cell
    if pbc[2]:
        z_adj_atom_pos = deepcopy(ind_1_positions)
        z_adj_atom_pos[:, 0] = z_adj_atom_pos[:, 0] + ind_1_cell[1]

        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(ind_1_positions[i] - z_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_1.add_edge(i, j)

    # calculate the pairs that within the bond length cutoff for the original cell for individual 2
    atom_num = len(ind_2_positions)
    for i in range(atom_num):
        g_2.add_node(i)

        for j in range(atom_num):
            distance = np.linalg.norm(ind_2_positions[i] - ind_2_positions[j])
            if distance <= cutoff and distance != 0:
                g_2.add_edge(i, j)

    # consider the atoms in neighbor cell
    if pbc[0]:
        x_adj_atom_pos = deepcopy(ind_2_positions)
        x_adj_atom_pos[:, 0] = x_adj_atom_pos[:, 0] + ind_2_cell[0]

        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(ind_2_positions[i] - x_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_2.add_edge(i, j)

    # consider the atoms in neighbor cell y direction
    if pbc[1]:
        y_adj_atom_pos = deepcopy(ind_2_positions)
        y_adj_atom_pos[:, 0] = y_adj_atom_pos[:, 0] + ind_2_cell[1]

        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(ind_2_positions[i] - y_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_2.add_edge(i, j)

    # consider the atoms in z direction neighbor cell
    if pbc[2]:
        z_adj_atom_pos = deepcopy(ind_2_positions)
        z_adj_atom_pos[:, 0] = z_adj_atom_pos[:, 0] + ind_2_cell[1]

        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(ind_2_positions[i] - z_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_2.add_edge(i, j)

    return nx.graph_edit_distance(g_1, g_2)


# function for calculate the energy of the complete graph
def energy_calculate_gnn(ind, pop, hof, model, left_e_per_atom, right_e_per_atom, left_atom_obj, right_atom_obj, interface_len, k, time_limit):

    # get the left and right atom number from the ind
    left_atomic_num = left_atom_obj.get_atomic_numbers()[0]
    right_atomic_num = right_atom_obj.get_atomic_numbers()[0]

    # generate the pytorch geometric object
    nx_graph, atom_obj, rotate_obj = ind.struct, ind.atom_obj, ind.rotate_obj

    # if encounters the null middle part, set fitness to 100
    if nx_graph is None:
        # Define left and right atom number for the case of attach failed
        right_atom_num = right_atom_obj.get_global_number_of_atoms()
        left_atom_num = left_atom_obj.get_global_number_of_atoms()

        # save the structure to the disk
        pwd = os.getcwd()
        job_dir = join(pwd, str(ind.index))
        os.mkdir(job_dir)
        with open(join(job_dir, 'log.gnn'), 'w') as f:
            f.write('middle part have no atom!\n')
            f.write(f'left atom type: {left_atomic_num}\n')
            f.write(f'right atom type: {right_atomic_num}\n')
            f.write(f'left atom number: {left_atom_num}\n')
            f.write(f'right atom number: {right_atom_num}\n')
            f.write(f'left energy per atom: {left_e_per_atom}\n')
            f.write(f'right energy per atom: {right_e_per_atom}\n')

        # assign total energy and the fitness to the ind
        ind.formation_e = None
        ind.fitness.values = (100, -100)
        print('Setting novelty of individual {} to {} , due to null middle part!'.format(ind.index, ind.fitness))

        return

    # get the atom number for left and right side atom
    left_atom_num = np.count_nonzero(atom_obj.get_atomic_numbers() == left_atomic_num)
    right_atom_num = np.count_nonzero(atom_obj.get_atomic_numbers() == right_atomic_num)

    # create the Atoms object for ind
    ind_cell = np.array(ind[0])
    ind_position = ind_cell * np.delete(np.array(ind[1:]), -1, axis=1) 
    ind_atomic_numbers = np.array(ind[1:])[:, 2]
    ind_atom_obj = Atoms(ind_atomic_numbers, positions=ind_position, cell=ind_cell)

    # create the folder for storing the structures
    pwd = os.getcwd()
    job_dir = join(pwd, str(ind.index))
    os.mkdir(job_dir)

    # write the Lammps input structure to the folder
    atom_obj.write(join(job_dir, 'in.atom'), format='lammps-data')
    ind_atom_obj.write(join(job_dir, 'middle'), format='lammps-data')
    rotate_obj.write(join(job_dir, 'rotate'), format='lammps-data')

    # using GNN get the energy of the system
    pygeo_graph = from_networkx(nx_graph)

    # use dataloader to generate the batch with batch_size = 1
    nominal_dataset = DataLoader([pygeo_graph], batch_size=1, shuffle=False)

    # use model to evaluate total energy
    tot_e = None
    for data in nominal_dataset:
        tot_e, emb = model(data)
    tot_e = tot_e.item()

    # calculate the formation energy
    if tot_e != 10000:
        if left_atomic_num == right_atomic_num:
            formation_e = (tot_e - left_atom_num * left_e_per_atom) / (2 * interface_len)
        else:
            formation_e = (tot_e - left_atom_num * left_e_per_atom - right_atom_num * right_e_per_atom) / (2 * interface_len)
    else:
        formation_e = 100

    # calculate the fitness according to the novelty (graph edit distance)
    # select first some of the best structures from hof
    selected_hof = hof[:len(pop)]
    eigen_sim_list = []
    for _ in pop:
        try:
            eigen_sim_list.append(eigen_sim(ind, _))
        except nx.exception.NetworkXError:
            eigen_sim_list.append(abs(len(ind) - len(_)))  # Use the difference of atom number as similarity, when no atoms in structure
        except AttributeError:
            eigen_sim_list.append(100)
    for _ in selected_hof:
        try:
            eigen_sim_list.append(eigen_sim(ind, _))
        except nx.exception.NetworkXError:
            eigen_sim_list.append(abs(len(ind) - len(_)))
        except AttributeError:
            eigen_sim_list.append(100)

    eigen_sim_list.sort()  # sort the similarity
    eigen_sim_list = eigen_sim_list[:k]  # pick out k nearest neighbors

    fitness = sum(eigen_sim_list) / len(eigen_sim_list)

    # save the structure to the disk and write a log file
    with open(join(job_dir, 'log.gnn'), 'w') as f:
        f.write(f'total energy of structure: {tot_e}\n')
        f.write(f'left atom type: {left_atomic_num}\n')
        f.write(f'right atom type: {right_atomic_num}\n')
        f.write(f'left atom number: {left_atom_num}\n')
        f.write(f'right atom number: {right_atom_num}\n')
        f.write(f'interface length: {interface_len}\n')
        f.write(f'left energy per atom: {left_e_per_atom}\n')
        f.write(f'right energy per atom: {right_e_per_atom}\n')

    # assign total energy and the fitness to the ind
    ind.formation_e = formation_e
    ind.fitness.values = (formation_e, fitness)
    print(f'Setting novelty and formation energy of individual {ind.index} to {ind.fitness.values[1]} & {ind.fitness.values[0]}')


# function for rotate in.data 180 degrees
def rotate_180(input_file, output_file, xcell_line, atom_line):
    # read input file cell and atom info
    with open(input_file, 'r') as f:
        input_lines = f.readlines()

    # get the middle point and x,y cell
    x_cell = float(input_lines[xcell_line-1].split()[1])
    y_cell = float(input_lines[xcell_line].split()[1])
    z_cell = float(input_lines[xcell_line+1].split()[1])
    mid_x = x_cell/2
    mid_y = y_cell/2
    mid_z = z_cell/2

    # get the atom position matrix
    matrix = []
    atom_lines = input_lines[atom_line-1:]
    for line in atom_lines:
        atom_coords = []
        if line != '\n':
            for i in range(len(line.split()) - 3):
                atom_coords.append(line.split()[i])
            atom_coords.append(float(line.split()[-3]))
            atom_coords.append(float(line.split()[-2]))
            atom_coords.append(float(line.split()[-1]))
            matrix.append(atom_coords)

    # calculate the atom position after the rotation
    for atom in matrix:
        original_x = atom[-3]
        original_y = atom[-2]
        original_z = atom[-1]
        delta_x = mid_x - original_x
        delta_y = mid_y - original_y
        delta_z = mid_z - original_z
        rotate_x = original_x + 2*delta_x
        rotate_y = original_y + 2*delta_y
        rotate_z = original_z + 2*delta_z
        atom[-3] = rotate_x
        atom[-2] = rotate_y
        atom[-1] = rotate_z

    # write the rotated atom position to the output file
    # copy lines before atom position
    pre_lines = input_lines[:atom_line-1]
    with open(output_file, 'w') as f:
        for line in pre_lines:
            f.write(line)
        for atom in matrix:
            for item in atom:
                if isinstance(item, str):
                    f.write(item + ' ')
                else:
                    f.write(str(item) + ' ')
            f.write('\n')

    return y_cell


# get the total energy from static lammps run
def get_tot_e_static(output_path):
    """
        :param output_path: the path to the LAMMPS output log file
        :return: return the total energy that read from the log file
    """
    # read the log.lammps file as a list of strings
    with open(output_path, 'r') as f:
        lines = f.readlines()

    # get the last line with the keywords (where the final energy is)
    match_strings = ['Step', 'Temp', 'E_pair', 'E_mol', 'TotEng']
    for i in range(len(lines)):
        if all(match in lines[i] for match in match_strings):
            tot_energy = float(lines[i + 1].split()[4])
    return tot_energy

