# Functions for create and manipulate the individual

import random
import numpy as np
import networkx as nx
from ase import Atoms
from ase.io import write
from copy import deepcopy
from ase.data import covalent_radii
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure


# TODO: finish all the creation and variation funcitons


def ind_creator(interface_len, min_width, max_width, max_height, cell_height, min_atom_num, max_atom_num, atom_type_list):

    """
    :param interface_len: length of interface
    :param min_width:
    :param max_width:
    :param max_height:
    :param cell_height: total cell height including vacuum
    :param min_atom_num:
    :param max_atom_num:
    :param atom_type_list: atom type list that include the atomic number of each type
    :return: a list with first element be the size of the cell, rest of the them are the atom position
    """

    ind = []
    # decide the cell size and the planar atom density
    width = min_width + (max_width - min_width) * random.random()

    # use the approximate planary atom density to decide how many atoms
    # this helps avoid the case that in first generation all individuals are not good after relaxation
    max_density = max_atom_num/(interface_len*max_width)
    min_density = min_atom_num/(interface_len*min_width)
    planar_atom_density = (max_density + min_density)/2
    atom_num = int(planar_atom_density*width*interface_len)
    ind.append([interface_len, width, cell_height])

    # decide the atom range in z direction in frational coordinates
    bottom_limit = (cell_height/2 - max_height/2)/cell_height
    upper_limit = (cell_height/2 + max_height/2)/cell_height

    for atom in range(atom_num):
        ind.append([random.random(), random.random(), random.uniform(bottom_limit, upper_limit),
                    random.choice(atom_type_list)])
    return ind


# function that create the ind object that makes the inter-atom distance larger than certain value
def ind_creator_dist_control(interface_len, min_width, max_width, max_height, cell_height, min_atom_num, max_atom_num,
                             atom_type_list, left_atom_obj, right_atom_obj, inter_atom_limit, filter_range=3.5):
    # decide the cell size and the planar atom density
    width = min_width + (max_width - min_width) * random.random()

    # use the approximate planary atom density to decide how many atoms
    # this helps avoid the case that in first generation all individuals are not good after relaxation
    max_density = max_atom_num / (interface_len * max_width)
    min_density = min_atom_num / (interface_len * min_width)
    planar_atom_density = (max_density + min_density) / 2
    atom_num = int(planar_atom_density * width * interface_len)

    # decide the atom range in z direction in cartesian coordinates
    bottom_limit = cell_height / 2 - max_height / 2
    upper_limit = cell_height / 2 + max_height / 2

    # decide the cell for calculate the cartesian coordinates later
    mid_cell = np.array([interface_len, width, cell_height])

    # make cell a one dimensional vector
    left_cell = [left_atom_obj.get_cell()[0][0], left_atom_obj.get_cell()[1][1], left_atom_obj.get_cell()[2][2]]
    left_cell = np.array(left_cell)
    left_ori_atom_pos = left_atom_obj.get_positions()

    right_cell = [right_atom_obj.get_cell()[0][0], right_atom_obj.get_cell()[1][1], right_atom_obj.get_cell()[2][2]]
    right_ori_atom_pos = right_atom_obj.get_positions()

    # decide the atom coordinates for the right side
    right_attatched_pos = right_ori_atom_pos + np.tile(np.array([0, left_cell[1] + mid_cell[1], 0]),
                                                       (right_ori_atom_pos.shape[0], 1))

    # from the coordinates array generate the array that contains atomic number
    # number at the end of each line is the atomic number corresponding to that atom coordiantes
    left_coords = np.c_[left_ori_atom_pos, left_atom_obj.get_atomic_numbers()]
    right_coords = np.c_[right_attatched_pos, right_atom_obj.get_atomic_numbers()]

    # generate the list for the ind & append the cell to the first row
    ind = [mid_cell.tolist()]

    # build an array include the atoms close to the middle part for distance checking
    filter_coords = []
    for atom in left_coords:
        if left_cell[1] - filter_range < atom[1] <= left_cell[1]:
            filter_coords.append(atom)

    for atom in right_coords:
        if left_cell[1] + mid_cell[1] < atom[1] <= left_cell[1] + mid_cell[1] + filter_range:
            filter_coords.append(atom)

    # build the atom coordinate array that contains all the coordinates for the neighbors of middle atom
    # here only the periodic image at the x direction is considered
    x_up_neighbor = []
    x_down_neighbor = []

    # determine whether use the covalent radius from ASE or use the user defined inter atom distance limiation
    try:
        if 'covalent radius' in inter_atom_limit:
            use_covalent_dist = True
        else:
            raise TypeError('The inter atom limit parameter should either be covalent radius or an integer!')
    except TypeError:
        use_covalent_dist = False

    # add more atoms satisfy the distance requirement
    for atom in range(atom_num - 1):
        satisfy_condition = False
        s = 0
        while not satisfy_condition:
            s += 1
            # random generate an atom coordinate
            random_coords = np.array([random.uniform(0, left_cell[0]),
                                      random.uniform(left_cell[1], left_cell[1] + width),
                                      random.uniform(bottom_limit, upper_limit),
                                      random.choice(atom_type_list)])

            # check the distance between existing atoms (including the x direction neighbor)
            # and the random generated new atom
            if len(x_up_neighbor) != 0 and len(x_down_neighbor) != 0:
                dist_control_atoms = np.vstack((filter_coords, x_up_neighbor))
                dist_control_atoms = np.vstack((dist_control_atoms, x_down_neighbor))
            else:
                dist_control_atoms = filter_coords

            for atom_line in dist_control_atoms:

                # determine the minimum distance between atoms
                if use_covalent_dist:
                    min_covalent_len = covalent_radii[int(atom_line[-1])] + covalent_radii[int(random_coords[-1])]
                else:
                    min_covalent_len = inter_atom_limit

                # calculate the distance and decide whether it is larger than the minimum covalent bond length
                dist = np.linalg.norm(random_coords[:3] - atom_line[:3])
                if dist >= min_covalent_len:
                    satisfy_condition = True
                else:
                    # if there's one atom that has distance less than this new atom
                    # break the for loop and generate a new one
                    satisfy_condition = False
                    break

        # append the atom that satisfy the constrain to the ind
        # the coordinates need to be modified so only the middle part remain
        # use deepcopy so the coordinates won't be changed later for the same object
        # wtf python?! if not use deep copy the variable name target to the original object
        filter_coords.append(deepcopy(random_coords))
        coords_before_norm = deepcopy(random_coords)
        coords_before_norm[1] -= left_cell[1]
        coords_before_norm = coords_before_norm / np.append(mid_cell, 1)  # the last number is now atomic number
        ind.append(coords_before_norm.tolist())

        # append the neighbor image atom coordinates to the neighbor list
        up_neighbor_img = deepcopy(random_coords)
        up_neighbor_img[0] += interface_len
        x_up_neighbor.append(up_neighbor_img)
        down_neighbor_img = deepcopy(random_coords)
        down_neighbor_img[0] -= interface_len
        x_down_neighbor.append(down_neighbor_img)

    return ind


# function that create the ind object that makes the inter-atom distance larger than certain value
def ind_creator_ama(interface_len, min_width, max_width, max_height, cell_height, atom_type_list, left_atom_obj,
                    right_atom_obj, inter_atom_limit, filter_range=3.5, loop_limit=10000):
    # decide the cell size and the planar atom density
    width = min_width + (max_width - min_width) * random.random()

    # use the approximate planary atom density to decide how many atoms
    # this helps avoid the case that in first generation all individuals are not good after relaxation
    # max_density = max_atom_num / (interface_len * max_width)
    # min_density = min_atom_num / (interface_len * min_width)
    # planar_atom_density = (max_density + min_density) / 2
    # atom_num = int(planar_atom_density * width * interface_len)

    # decide the atom range in z direction in cartesian coordinates
    bottom_limit = cell_height / 2 - max_height / 2
    upper_limit = cell_height / 2 + max_height / 2

    # decide the cell for calculate the cartesian coordinates later
    mid_cell = np.array([interface_len, width, cell_height])

    # make cell a one dimensional vector
    left_cell = [left_atom_obj.get_cell()[0][0], left_atom_obj.get_cell()[1][1], left_atom_obj.get_cell()[2][2]]
    left_cell = np.array(left_cell)
    left_ori_atom_pos = left_atom_obj.get_positions()

    right_cell = [right_atom_obj.get_cell()[0][0], right_atom_obj.get_cell()[1][1], right_atom_obj.get_cell()[2][2]]
    right_ori_atom_pos = right_atom_obj.get_positions()

    # decide the atom coordinates for the right side
    right_attatched_pos = right_ori_atom_pos + np.tile(np.array([0, left_cell[1] + mid_cell[1], 0]),
                                                       (right_ori_atom_pos.shape[0], 1))

    # from the coordinates array generate the array that contains atomic number
    # number at the end of each line is the atomic number corresponding to that atom coordiantes
    left_coords = np.c_[left_ori_atom_pos, left_atom_obj.get_atomic_numbers()]
    right_coords = np.c_[right_attatched_pos, right_atom_obj.get_atomic_numbers()]

    # generate the list for the ind & append the cell to the first row
    ind = [mid_cell.tolist()]

    # build an array include the atoms close to the middle part for distance checking
    filter_coords = []
    for atom in left_coords:
        if left_cell[1] - filter_range < atom[1] <= left_cell[1]:
            filter_coords.append(atom)

    for atom in right_coords:
        if left_cell[1] + mid_cell[1] < atom[1] <= left_cell[1] + mid_cell[1] + filter_range:
            filter_coords.append(atom)

    # build the atom coordinate array that contains all the coordinates for the neighbors of middle atom
    # here only the periodic image at the x direction is considered
    x_up_neighbor = []
    x_down_neighbor = []

    # determine whether use the covalent radius from ASE or use the user defined inter atom distance limiation
    try:
        if 'covalent radius' in inter_atom_limit:
            use_covalent_dist = True
        else:
            raise TypeError('The inter atom limit parameter should either be covalent radius or an integer!')
    except TypeError:
        use_covalent_dist = False

    # Add more atoms satisfy the distance requirement
    # Add as many atoms as possible to the middle part
    # If after loop_limit iteration, still not find new atom, then give up
    continue_find = True
    while continue_find:
        loop_num = 0
        satisfy_condition = False
        while not satisfy_condition:
            # random generate an atom coordinate
            random_coords = np.array([random.uniform(0, left_cell[0]),
                                      random.uniform(left_cell[1], left_cell[1] + width),
                                      random.uniform(bottom_limit, upper_limit),
                                      random.choice(atom_type_list)])

            # check the distance between existing atoms (including the x direction neighbor)
            # and the random generated new atom
            if len(x_up_neighbor) != 0 and len(x_down_neighbor) != 0:
                dist_control_atoms = np.vstack((filter_coords, x_up_neighbor))
                dist_control_atoms = np.vstack((dist_control_atoms, x_down_neighbor))
            else:
                dist_control_atoms = filter_coords

            for atom_line in dist_control_atoms:

                # determine the minimum distance between atoms
                if use_covalent_dist:
                    min_covalent_len = covalent_radii[int(atom_line[-1])] + covalent_radii[int(random_coords[-1])]
                else:
                    min_covalent_len = inter_atom_limit

                # calculate the distance and decide whether it is larger than the minimum covalent bond length
                dist = np.linalg.norm(random_coords[:3] - atom_line[:3])
                if dist >= min_covalent_len:
                    satisfy_condition = True
                else:
                    # if there's one atom that has distance less than this new atom
                    # break the for loop and generate a new one
                    satisfy_condition = False
                    loop_num += 1
                    break

            # if loop number reach the limitation set the continue_find to False
            if loop_num > loop_limit:
                continue_find = False
                break

        # Append the atom that satisfy the constrain to the ind
        # The coordinates need to be modified so only the middle part remain
        # Use deepcopy so the coordinates won't be changed later for the same object
        # If the loop is interupted due to the maxmum loop number reached, don't append this very last atom
        # Wtf python?! if not use deep copy the variable name target to the original object
        if continue_find:
            filter_coords.append(deepcopy(random_coords))
            coords_before_norm = deepcopy(random_coords)
            coords_before_norm[1] -= left_cell[1]
            coords_before_norm = coords_before_norm / np.append(mid_cell, 1)  # the last number is now atomic number
            ind.append(coords_before_norm.tolist())

            # append the neighbor image atom coordinates to the neighbor list
            up_neighbor_img = deepcopy(random_coords)
            up_neighbor_img[0] += interface_len
            x_up_neighbor.append(up_neighbor_img)
            down_neighbor_img = deepcopy(random_coords)
            down_neighbor_img[0] -= interface_len
            x_down_neighbor.append(down_neighbor_img)

    return ind


# Function that generate the ind object by stuffing as much as possible atoms
# without break the interatomic limitation
def ind_creator_amap(interface_len, min_width, max_width, max_height, cell_height, atom_type_list, left_atom_obj,
                    right_atom_obj, inter_atom_limit, filter_range=3.5, loop_limit=10000):
    # decide the cell size and the planar atom density
    width = min_width + (max_width - min_width) * random.random()

    # use the approximate planary atom density to decide how many atoms
    # this helps avoid the case that in first generation all individuals are not good after relaxation
    # max_density = max_atom_num / (interface_len * max_width)
    # min_density = min_atom_num / (interface_len * min_width)
    # planar_atom_density = (max_density + min_density) / 2
    # atom_num = int(planar_atom_density * width * interface_len)

    # decide the atom range in z direction in cartesian coordinates
    bottom_limit = cell_height / 2 - max_height / 2
    upper_limit = cell_height / 2 + max_height / 2

    # decide the cell for calculate the cartesian coordinates later
    mid_cell = np.array([interface_len, width, cell_height])

    # make cell a one dimensional vector
    left_cell = [left_atom_obj.get_cell()[0][0], left_atom_obj.get_cell()[1][1], left_atom_obj.get_cell()[2][2]]
    left_cell = np.array(left_cell)
    left_ori_atom_pos = left_atom_obj.get_positions()

    right_cell = [right_atom_obj.get_cell()[0][0], right_atom_obj.get_cell()[1][1], right_atom_obj.get_cell()[2][2]]
    right_cell = np.array(right_cell)
    right_ori_atom_pos = right_atom_obj.get_positions()

    # decide the atom coordinates for the right side
    left_attatched_pos = left_ori_atom_pos + np.tile(np.array([0, right_cell[1] + mid_cell[1], 0]),
                                                       (left_ori_atom_pos.shape[0], 1))

    # from the coordinates array generate the array that contains atomic number
    # number at the end of each line is the atomic number corresponding to that atom coordiantes
    right_coords = np.c_[right_ori_atom_pos, right_atom_obj.get_atomic_numbers()]
    left_coords = np.c_[left_attatched_pos, left_atom_obj.get_atomic_numbers()]

    # generate the list for the ind & append the cell to the first row
    ind = [mid_cell.tolist()]

    # build an array include the atoms close to the middle part for distance checking
    filter_coords = []
    for atom in left_coords:
        if right_cell[1] + mid_cell[1] < atom[1] <= right_cell[1] + mid_cell[1] + filter_range:
            filter_coords.append(atom)

    for atom in right_coords:
        if right_cell[1] - filter_range < atom[1] <= right_cell[1]:
            filter_coords.append(atom)
    #print(filter_coords)

    # build the atom coordinate array that contains all the coordinates for the neighbors of middle atom
    # here only the periodic image at the x direction is considered
    x_up_neighbor = []
    x_down_neighbor = []

    # determine whether use the covalent radius from ASE or use the user defined inter atom distance limiation
    try:
        if 'covalent radius' in inter_atom_limit:
            use_covalent_dist = True
        else:
            raise TypeError('The inter atom limit parameter should either be covalent radius or an integer!')
    except TypeError:
        use_covalent_dist = False

    # Add more atoms satisfy the distance requirement
    # Add as many atoms as possible to the middle part
    # If after loop_limit iteration, still not find new atom, then give up
    continue_find = True
    while continue_find:
        loop_num = 0
        satisfy_condition = False
        while not satisfy_condition:
            # random generate an atom coordinate
            random_coords = np.array([random.uniform(0, left_cell[0]),
                                      random.uniform(left_cell[1], left_cell[1] + width),
                                      random.uniform(bottom_limit, upper_limit),
                                      random.choice(atom_type_list)])

            # check the distance between existing atoms (including the x direction neighbor)
            # and the random generated new atom
            if len(x_up_neighbor) != 0 and len(x_down_neighbor) != 0:
                dist_control_atoms = np.vstack((filter_coords, x_up_neighbor))
                dist_control_atoms = np.vstack((dist_control_atoms, x_down_neighbor))
            else:
                dist_control_atoms = filter_coords

            for atom_line in dist_control_atoms:

                # determine the minimum distance between atoms
                if use_covalent_dist:
                    min_covalent_len = covalent_radii[int(atom_line[-1])] + covalent_radii[int(random_coords[-1])]
                else:
                    min_covalent_len = inter_atom_limit

                # calculate the distance and decide whether it is larger than the minimum covalent bond length
                dist = np.linalg.norm(random_coords[:3] - atom_line[:3])
                if dist >= min_covalent_len:
                    satisfy_condition = True
                else:
                    # if there's one atom that has distance less than this new atom
                    # break the for loop and generate a new one
                    satisfy_condition = False
                    loop_num += 1
                    break

            # if loop number reach the limitation set the continue_find to False
            if loop_num > loop_limit:
                continue_find = False
                break

        # Append the atom that satisfy the constrain to the ind
        # The coordinates need to be modified so only the middle part remain
        # Use deepcopy so the coordinates won't be changed later for the same object
        # If the loop is interupted due to the maxmum loop number reached, don't append this very last atom
        # Wtf python?! if not use deep copy the variable name target to the original object
        if continue_find:
            filter_coords.append(deepcopy(random_coords))
            coords_before_norm = deepcopy(random_coords)
            coords_before_norm[1] -= left_cell[1]
            coords_before_norm = coords_before_norm / np.append(mid_cell, 1)  # the last number is now atomic number
            ind.append(coords_before_norm.tolist())

            # append the neighbor image atom coordinates to the neighbor list
            up_neighbor_img = deepcopy(random_coords)
            up_neighbor_img[0] += interface_len
            x_up_neighbor.append(up_neighbor_img)
            down_neighbor_img = deepcopy(random_coords)
            down_neighbor_img[0] -= interface_len
            x_down_neighbor.append(down_neighbor_img)

    return ind



# function to attach the middle part from GA to the two sides
def attach(ind, left_atom_obj, right_atom_obj, cutoff=2.6, pbc=[True, True, False], return_graph=True,
           periodic_struc=True):
    # make cell a one dimensional vector
    left_cell = [left_atom_obj.get_cell()[0][0], left_atom_obj.get_cell()[1][1], left_atom_obj.get_cell()[2][2]]
    left_cell = np.array(left_cell)
    left_ori_atom_pos = left_atom_obj.get_positions()

    right_cell = [right_atom_obj.get_cell()[0][0], right_atom_obj.get_cell()[1][1], right_atom_obj.get_cell()[2][2]]
    right_cell = np.array(right_cell)
    right_ori_atom_pos = right_atom_obj.get_positions()

    # get the cell size for tow side graphs
    middle_cell = np.array(ind[0])
    try:
        mid_old_reduced_atom_pos = np.delete(np.array(ind[1:]), -1, 1)
    except np.AxisError:
        if return_graph:
            return None, None, None
        else:
            return None, None

    mid_ori_atom_pos = mid_old_reduced_atom_pos * np.tile(middle_cell, (len(ind) - 1, 1))

    # change the atom coordinates to fit the complete cell
    mid_attatched_pos = mid_ori_atom_pos + np.tile(np.array([0, right_cell[1], 0]), (mid_ori_atom_pos.shape[0], 1))
    left_attatched_pos = left_ori_atom_pos + np.tile(np.array([0, right_cell[1] + middle_cell[1], 0]), (left_ori_atom_pos.shape[0], 1))

    # Generate the rotated atom objects
    if periodic_struc:
        # Find the center-point to rotate according to
        mid_x = middle_cell[0] / 2
        mid_y = middle_cell[1] / 2
        mid_z = middle_cell[2] / 2
        center_vec = np.array([mid_x, mid_y, mid_z])

        # Calculate the translation vector for each atom
        trans_vec = center_vec - mid_ori_atom_pos

        # Translate all atoms two times of the translation vector so all of it center symmetric with original
        rotated_mid_ori_pos = mid_ori_atom_pos + 2 * trans_vec

        # Generate the atom positions that need to be attatched
        rotated_mid_attatched_pos = rotated_mid_ori_pos + np.tile(np.array([0, left_cell[1] + middle_cell[1] +
                                                                            right_cell[1], 0]),
                                                                  (mid_ori_atom_pos.shape[0], 1))

        # generate the complete list of atoms
        complete_pos = np.vstack((right_ori_atom_pos, mid_attatched_pos, left_attatched_pos, rotated_mid_attatched_pos))

        mid_atomic_num = np.array(ind[1:])[:, -1]
        rotate_obj = Atoms(mid_atomic_num, positions=rotated_mid_ori_pos, cell=middle_cell, pbc=pbc)
    else:
        # generate the complete list of atoms
        complete_pos = np.vstack((right_ori_atom_pos, mid_attatched_pos, left_attatched_pos))
        rotate_obj = None

    # calculate the pairs that within the bond length cutoff for the original cell
    cell = np.array([left_cell[0], left_cell[1] + 2 * middle_cell[1] + right_cell[1], left_cell[2]])
    atom_num = len(complete_pos)

    # generate a atoms object using ASE
    left_atomic_num = left_atom_obj.get_atomic_numbers()
    right_atomic_num = right_atom_obj.get_atomic_numbers()
    mid_atomic_num = np.array(ind[1:])[:, -1]  # first line of ind is the cell
    atomic_num_array = np.append(right_atomic_num, mid_atomic_num)  # seems append method can't take more than 2 one time
    atomic_num_array = np.append(atomic_num_array, left_atomic_num)
    if periodic_struc:
        atomic_num_array = np.append(atomic_num_array, mid_atomic_num)
    complete_obj = Atoms(atomic_num_array, positions=complete_pos, cell=cell, pbc=pbc)

    # use the parameter 'return_graph' to determine if return also the graph obj
    if return_graph:

        # create the complete graph object and add the edges from two sides to the graph
        complete_graph = nx.Graph(cell=cell)

        for i in range(atom_num):
            feature_vector = np.append(cell, complete_pos[i] / cell)
            complete_graph.add_node(i, x=feature_vector)

            for j in range(atom_num):
                distance = np.linalg.norm(complete_pos[i] - complete_pos[j])
                if distance <= cutoff and distance != 0:
                    complete_graph.add_edge(i, j, distance=distance)

        # consider the atoms in neighbor cell
        if pbc[0]:
            x_adj_atom_pos = deepcopy(complete_pos)
            x_adj_atom_pos[:, 0] = x_adj_atom_pos[:, 0] + cell[0]
            for i in range(atom_num):
                for j in range(atom_num):
                    distance = np.linalg.norm(complete_pos[i] - x_adj_atom_pos[j])
                    if distance <= cutoff and distance != 0:
                        complete_graph.add_edge(i, j, distance=distance)

        # consider the atoms in neighbor cell y direction
        if pbc[1]:
            y_adj_atom_pos = deepcopy(complete_pos)
            y_adj_atom_pos[:, 0] = y_adj_atom_pos[:, 0] + cell[1]
            for i in range(atom_num):
                for j in range(atom_num):
                    distance = np.linalg.norm(complete_pos[i] - y_adj_atom_pos[j])
                    if distance <= cutoff and distance != 0:
                        complete_graph.add_edge(i, j, distance=distance)

        # consider the atoms in z direction neighbor cell
        if pbc[2]:
            z_adj_atom_pos = deepcopy(complete_pos)
            z_adj_atom_pos[:, 0] = z_adj_atom_pos[:, 0] + cell[1]
            for i in range(atom_num):
                for j in range(atom_num):
                    distance = np.linalg.norm(complete_pos[i] - z_adj_atom_pos[j])
                    if distance <= cutoff and distance != 0:
                        complete_graph.add_edge(i, j, distance=distance)

        return complete_graph, complete_obj, rotate_obj

    else:
        return complete_obj, rotate_obj


def cross_over_1pt(parent_1, parent_2, cut_loc_mu, cut_loc_sigma):

    """
    :param parent_1: first parent
    :param parent_2: second parent
    :param cut_loc_mu: mean value of the cut location (fractional) coordinate, for a Guassian distribution
    :param cut_loc_sigma: sigma of the Gaussian distribution
    :return: two child individuals
    """

    # TODO: add multiple points cross over or line slice

    # either slice the x or y direction
    dimension_index = random.randint(0, 1)

    # take the lengths of two parents
    length_1 = len(parent_1)
    length_2 = len(parent_2)

    # randomly decide the new cell size
    interface_len = random.uniform(parent_1[0][0], parent_2[0][0])
    width = random.uniform(parent_1[0][1], parent_2[0][1])
    height = random.uniform(parent_1[0][2], parent_2[0][2])
    '''rand = random.random()
    if rand < (1/3):
        interface_len = (parent_1[0][0] + parent_2[0][0])/2
        width = (parent_1[0][1] + parent_2[0][1])/2
        height = (parent_1[0][2] + parent_2[0][2])/2
    elif (1/3) < rand < (2/3):
        interface_len = max(parent_1[0][0], parent_2[0][0])
        width = max(parent_1[0][1], parent_2[0][1])
        height = max(parent_1[0][2], parent_2[0][2])
    else:
        interface_len = min(parent_1[0][0], parent_2[0][0])
        width = min(parent_1[0][1], parent_2[0][1])
        height = min(parent_1[0][2], parent_2[0][2])'''

    # make temporary copy of parents, and empty the original partents lists
    copy_1 = deepcopy(parent_1)
    copy_2 = deepcopy(parent_2)
    parent_1.clear()
    parent_2.clear()

    # put the cell size into the child
    parent_1.append([interface_len, width, height])
    parent_2.append([interface_len, width, height])

    # decide the cut point and make copy of the original
    cut_point = random.gauss(cut_loc_mu, cut_loc_sigma)
    while cut_point > 1 or cut_point < 0:
        cut_point = random.gauss(cut_loc_mu, cut_loc_sigma)

    # put the sliced parts in the offspring
    for atom_num in range(1, length_1):
        atom = copy_1[atom_num]
        if atom[dimension_index] < cut_point:
            parent_1.append(atom)
        else:
            parent_2.append(atom)

    for atom_num in range(1, length_2):
        atom = copy_2[atom_num]
        if atom[dimension_index] <= cut_point:
            parent_2.append(atom)
        else:
            parent_1.append(atom)

    return parent_1, parent_2


# do the structure mutation of individual, perturb given fractional number of atom with a Gaussian distribution
def structure_mutation(ind, frac_atom, max_height, std):
    # determine how many atoms mutated according to the frac_atom parameter
    atom_num = len(ind)
    #print(atom_num)
    mutate_atom_num = random.randint(1, len(ind)-1)
    #print(mutate_atom_num)

    # get the cell size and limit for atom height in frac coordinates
    interface_len = ind[0][0]
    width = ind[0][1]
    cell_height = ind[0][2]
    bottom_limit = (cell_height / 2 - max_height / 2) / cell_height
    upper_limit = (cell_height / 2 + max_height / 2) / cell_height

    # copy and clear the original individual
    temp_copy = deepcopy(ind)
    ind.clear()

    # decide a list of line number that make the change
    full_list = list(range(1, atom_num))
    if atom_num == mutate_atom_num:
        line_num = full_list
    else:
        line_num = random.sample(full_list, mutate_atom_num)
    # print(line_num)

    # change the coordiante
    for line in line_num:
        # consider 3 dimentional muatation, compute the fractional distance
        interface_mutation_dis = random.gauss(0, std)/interface_len
        width_mutation_dis = random.gauss(0, std)/width
        height_mutation_dis = random.gauss(0, std)/cell_height

        # if fractional coords larger than boundary (0, 1) then set the coordinate to 0 or 1
        # note that if the boundary in Lammps input file is not periodic for one direction
        # then the coordinate for an atom along that diraction should not equal to boundary value
        if interface_mutation_dis + temp_copy[line][0] < 0:
            # make it not equal to boundary coordinate just be safe
            temp_copy[line][0] = 0.000001
        elif 0 <= interface_mutation_dis + temp_copy[line][0] <= 1:
            temp_copy[line][0] += interface_mutation_dis
        else:
            temp_copy[line][0] = 0.999999

        if width_mutation_dis + temp_copy[line][1] < 0:
            temp_copy[line][1] = 0.000001
        elif 0 <= width_mutation_dis + temp_copy[line][1] <= 1:
            temp_copy[line][1] += width_mutation_dis
        else:
            temp_copy[line][1] = 0.999999

        if height_mutation_dis + temp_copy[line][2] < bottom_limit:
            temp_copy[line][2] = bottom_limit
        elif bottom_limit <= height_mutation_dis + temp_copy[line][2] <= upper_limit:
            temp_copy[line][2] += height_mutation_dis
        else:
            temp_copy[line][2] = upper_limit

    # copy changed individual into original list
    for atom in range(atom_num):
        ind.append(temp_copy[atom])

    return ind


# function that do the atom number mutation
def atom_num_mutation(ind, sigma, min_atom_num, max_atom_num, max_height, atom_type_list, mean_value=0):
    # get the length of the individual and a list that contain the label of each line (exclude first line)
    ind_len = len(ind)
    full_list = list(range(1, ind_len))

    # decide the atom change number by using a integer Gauss distribution
    # use a while loop to avoid the case that normal distribution give 0 or make
    # TODO: find out a way that don't use while loop
    atom_change_num = 0
    while not (atom_change_num != 0 and (ind_len - 1 + atom_change_num) > 0):
    #while not (atom_change_num != 0 and min_atom_num <= (ind_len - 1 + atom_change_num) <= max_atom_num):
        atom_change_num = round(random.gauss(mean_value, sigma))

    # remove or add atoms depends on value of atom_change_num
    if atom_change_num < 0:
        del_num = abs(atom_change_num)
        while del_num > 0:
            ind_len = len(ind)
            ind.pop(random.randint(1, ind_len - 1))
            del_num -= 1
        return ind
    else:
        # get the cell size and limit for atom height in fractional coordinates
        cell_height = ind[0][2]
        bottom_limit = (cell_height / 2 - max_height / 2) / cell_height
        upper_limit = (cell_height / 2 + max_height / 2) / cell_height
        for i in range(atom_change_num):
            ind.append([random.random(), random.random(), random.uniform(bottom_limit, upper_limit),
                        random.choice(atom_type_list)])
        return ind


# Assign graph and atom object to the ind
def assign_struct(ind, left_atom_obj, right_atom_obj, cutoff, pbc):

    nx_graph, atom_obj, rotate_obj = attach(ind, left_atom_obj, right_atom_obj, cutoff=cutoff, pbc=pbc,
                                                return_graph=True, periodic_struc=True)

    ind.struct = nx_graph
    ind.atom_obj = atom_obj
    ind.rotate_obj = rotate_obj


# Function that assign a graph object to the ind
def assign_graph(ind, left_atom_obj, right_atom_obj, cutoff=2.6, pbc=[True, False, False]):
    # make cell a one dimensional vector
    left_cell = [left_atom_obj.get_cell()[0][0], left_atom_obj.get_cell()[1][1], left_atom_obj.get_cell()[2][2]]
    left_cell = np.array(left_cell)
    left_ori_atom_pos = left_atom_obj.get_positions()

    right_cell = [right_atom_obj.get_cell()[0][0], right_atom_obj.get_cell()[1][1], right_atom_obj.get_cell()[2][2]]
    right_cell = np.array(right_cell)
    right_ori_atom_pos = right_atom_obj.get_positions()

    # get the cell size for tow side graphs
    middle_cell = np.array(ind[0])
    mid_old_reduced_atom_pos = np.delete(np.array(ind[1:]), -1, 1)
    mid_ori_atom_pos = mid_old_reduced_atom_pos * np.tile(middle_cell, (len(ind) - 1, 1))

    # change the atom coordinates to fit the complete cell
    mid_attatched_pos = mid_ori_atom_pos + np.tile(np.array([0, left_cell[1], 0]), (mid_ori_atom_pos.shape[0], 1))
    right_attatched_pos = right_ori_atom_pos + np.tile(np.array([0, left_cell[1] + middle_cell[1], 0]),
                                                       (right_ori_atom_pos.shape[0], 1))

    # generate the complete list of atoms
    complete_pos = np.vstack((left_ori_atom_pos, mid_attatched_pos, right_attatched_pos))

    # calculate the pairs that within the bond length cutoff for the original cell
    cell = np.array([left_cell[0], left_cell[1] + middle_cell[1] + right_cell[1], left_cell[2]])
    atom_num = len(complete_pos)

    # create the complete graph object and add the edges from two sides to the graph
    complete_graph = nx.Graph(cell=cell)
    for i in range(atom_num):
        feature_vector = np.append(cell, complete_pos[i] / cell)
        complete_graph.add_node(i, x=feature_vector)
        for j in range(atom_num):
            distance = np.linalg.norm(complete_pos[i] - complete_pos[j])
            if distance <= cutoff and distance != 0:
                complete_graph.add_edge(i, j, distance=distance)

    # consider the atoms in neighbor cell
    if pbc[0]:
        x_adj_atom_pos = deepcopy(complete_pos)
        x_adj_atom_pos[:, 0] = x_adj_atom_pos[:, 0] + cell[0]
        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(complete_pos[i] - x_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    complete_graph.add_edge(i, j, distance=distance)

    # consider the atoms in neighbor cell y direction
    if pbc[1]:
        y_adj_atom_pos = deepcopy(complete_pos)
        y_adj_atom_pos[:, 0] = y_adj_atom_pos[:, 0] + cell[1]
        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(complete_pos[i] - y_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    complete_graph.add_edge(i, j, distance=distance)

    # consider the atoms in z direction neighbor cell
    if pbc[2]:
        z_adj_atom_pos = deepcopy(complete_pos)
        z_adj_atom_pos[:, 0] = z_adj_atom_pos[:, 0] + cell[1]
        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(complete_pos[i] - z_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    complete_graph.add_edge(i, j, distance=distance)

    ind.struct = complete_graph
    return ind.struct


# Function that assign graph for each individual
def assign_graph_ind(ind_1, cutoff=2.6, pbc=[True, False, False]):
    # Create the graph objects for ind
    g_1 = nx.Graph()

    ind_1_cell = np.array(ind_1[0])
    try:
        ind_1_positions = np.delete(np.array(ind_1[1:]), -1, 1) * ind_1_cell
    except np.AxisError:
        ind_1.struct = g_1
        return ind_1.struct


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

    ind_1.struct = g_1
    return ind_1.struct
    

# function that merge atoms in the ind that are too close to each other
def merge_atom(ind, tol=0.01):
    ind.struct.merge_sites(tol)


# write the ind to a file for debug
def write_ind(ind, path):
    cell = np.array(ind[0])
    pos = np.array(ind[1:])[:, 0:-1] * cell
    symbol = np.array(ind[1:])[:, -1]
    complete_obj = Atoms(symbol, positions=pos, cell=cell, pbc=[1, 1, 1])
    complete_obj.write(path, format='vasp')


if __name__ == '__main__':
    from ase.io import read
    left_atom_obj = read(r'/Users/randy/gnn_dataset/12left', format='vasp')
    right_atom_obj = read(r'/Users/randy/gnn_dataset/12right', format='vasp')
    attach(ind, left_atom_obj, right_atom_obj, return_graph=False)
    #ind = ind_creator_ama(8.761141, 4.5, 8, 1.6, 16, [15, 15], left_atom_obj, right_atom_obj, 2.6)
    #write_ind(ind, '/Users/randy/GA_interface/bp1212_gnn_dist_control_5000/ind_test')
    #atom_obj = attach(ind, left_atom_obj, right_atom_obj, return_graph=False)
    #atom_obj.write('/Users/randy/GA_interface/bp1212_gnn_dist_control_5000/whole_test', format='vasp')
    #from ga_functions_gnn import check_constrain
    #s = check_constrain(ind, left_atom_obj, right_atom_obj, 0, 100, 1.6, 4, 2.6, 2.2, pbc=[True, False, False])
    #print(s)



