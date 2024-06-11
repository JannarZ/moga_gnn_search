# Functions for create and manipulate the individual

import random
import numpy as np
import networkx as nx
from ase import Atoms
from ase.io import write
from copy import deepcopy
from ase.data import covalent_radii

def ind_creator(interface_len: float, min_width: float, max_width: float, max_height: float, cell_height: float, min_atom_num: int, max_atom_num: int, atom_type_list: list) -> list:
    """
    Create an individual (GB structure) for the genetic algorithm.

    Parameters:
    interface_len (float): Length of the interface.
    min_width (float): Minimum width of the interface.
    max_width (float): Maximum width of the interface.
    max_height (float): Maximum height of the interface.
    cell_height (float): Height of the cell.
    min_atom_num (int): Minimum number of atoms.
    max_atom_num (int): Maximum number of atoms.
    atom_type_list (list): List of possible atom types.

    Returns:
    list: Generated individual with atomic positions and types.
    """
    ind = []
    # Decide the cell size and the planar atom density
    width = min_width + (max_width - min_width) * random.random()

    # Use the approximate planar atom density to decide how many atoms
    # This helps avoid the case that in the first generation all individuals are not good after relaxation
    max_density = max_atom_num / (interface_len * max_width)
    min_density = min_atom_num / (interface_len * min_width)
    planar_atom_density = (max_density + min_density) / 2
    atom_num = int(planar_atom_density * width * interface_len)
    ind.append([interface_len, width, cell_height])

    # Decide the atom range in the z direction in fractional coordinates
    bottom_limit = (cell_height / 2 - max_height / 2) / cell_height
    upper_limit = (cell_height / 2 + max_height / 2) / cell_height

    for atom in range(atom_num):
        ind.append([random.random(), random.random(), random.uniform(bottom_limit, upper_limit),
                    random.choice(atom_type_list)])
    return ind

def ind_creator_dist_control(interface_len: float, min_width: float, max_width: float, max_height: float, cell_height: float, min_atom_num: int, max_atom_num: int, atom_type_list: list, left_atom_obj: Atoms, right_atom_obj: Atoms, inter_atom_limit: float, filter_range: float = 3.5) -> list:
    """
    Create an individual (GB structure) with controlled inter-atomic distances.

    Parameters:
    interface_len (float): Length of the interface.
    min_width (float): Minimum width of the interface.
    max_width (float): Maximum width of the interface.
    max_height (float): Maximum height of the interface.
    cell_height (float): Height of the cell.
    min_atom_num (int): Minimum number of atoms.
    max_atom_num (int): Maximum number of atoms.
    atom_type_list (list): List of possible atom types.
    left_atom_obj (Atoms): Left atom object.
    right_atom_obj (Atoms): Right atom object.
    inter_atom_limit (float): Maximum allowed inter-atomic distance.
    filter_range (float): Range for filtering. Default is 3.5.

    Returns:
    list: Generated individual with controlled inter-atomic distances.
    """
    # Decide the cell size and the planar atom density
    width = min_width + (max_width - min_width) * random.random()

    # Use the approximate planar atom density to decide how many atoms
    # This helps avoid the case that in the first generation all individuals are not good after relaxation
    max_density = max_atom_num / (interface_len * max_width)
    min_density = min_atom_num / (interface_len * min_width)
    planar_atom_density = (max_density + min_density) / 2
    atom_num = int(planar_atom_density * width * interface_len)

    # Decide the atom range in the z direction in Cartesian coordinates
    bottom_limit = cell_height / 2 - max_height / 2
    upper_limit = cell_height / 2 + max_height / 2

    # Decide the cell for calculating the Cartesian coordinates later
    mid_cell = np.array([interface_len, width, cell_height])

    # Make cell a one-dimensional vector
    left_cell = [left_atom_obj.get_cell()[0][0], left_atom_obj.get_cell()[1][1], left_atom_obj.get_cell()[2][2]]
    left_cell = np.array(left_cell)
    left_ori_atom_pos = left_atom_obj.get_positions()

    right_cell = [right_atom_obj.get_cell()[0][0], right_atom_obj.get_cell()[1][1], right_atom_obj.get_cell()[2][2]]
    right_ori_atom_pos = right_atom_obj.get_positions()

    # Decide the atom coordinates for the right side
    right_attatched_pos = right_ori_atom_pos + np.tile(np.array([0, left_cell[1] + mid_cell[1], 0]),
                                                       (right_ori_atom_pos.shape[0], 1))

    # From the coordinates array generate the array that contains atomic numbers
    # Number at the end of each line is the atomic number corresponding to that atom coordinates
    left_coords = np.c_[left_ori_atom_pos, left_atom_obj.get_atomic_numbers()]
    right_coords = np.c_[right_attatched_pos, right_atom_obj.get_atomic_numbers()]

    # Generate the list for the individual & append the cell to the first row
    ind = [mid_cell.tolist()]

    # Build an array including the atoms close to the middle part for distance checking
    filter_coords = []
    for atom in left_coords:
        if left_cell[1] - filter_range < atom[1] <= left_cell[1]:
            filter_coords.append(atom)

    for atom in right_coords:
        if left_cell[1] + mid_cell[1] < atom[1] <= left_cell[1] + mid_cell[1] + filter_range:
            filter_coords.append(atom)

    # Build the atom coordinate array that contains all the coordinates for the neighbors of the middle atom
    # Here only the periodic image in the x direction is considered
    x_up_neighbor = []
    x_down_neighbor = []

    # Determine whether to use the covalent radius from ASE or use the user-defined inter-atom distance limitation
    try:
        if 'covalent radius' in inter_atom_limit:
            use_covalent_dist = True
        else:
            raise TypeError('The inter-atom limit parameter should either be covalent radius or an integer!')
    except TypeError:
        use_covalent_dist = False

    # Add more atoms that satisfy the distance requirement
    for _ in range(atom_num - 1):
        satisfy_condition = False
        while not satisfy_condition:
            # Randomly generate an atom coordinate
            random_coords = np.array([random.uniform(0, left_cell[0]),
                                      random.uniform(left_cell[1], left_cell[1] + width),
                                      random.uniform(bottom_limit, upper_limit),
                                      random.choice(atom_type_list)])

            # Check the distance between existing atoms (including the x direction neighbor)
            # and the randomly generated new atom
            if x_up_neighbor and x_down_neighbor:
                dist_control_atoms = np.vstack((filter_coords, x_up_neighbor, x_down_neighbor))
            else:
                dist_control_atoms = filter_coords

            for atom_line in dist_control_atoms:
                # Determine the minimum distance between atoms
                if use_covalent_dist:
                    min_covalent_len = covalent_radii[int(atom_line[-1])] + covalent_radii[int(random_coords[-1])]
                else:
                    min_covalent_len = inter_atom_limit

                # Calculate the distance and decide whether it is larger than the minimum covalent bond length
                dist = np.linalg.norm(random_coords[:3] - atom_line[:3])
                if dist < min_covalent_len:
                    break
            else:
                satisfy_condition = True

        # Append the atom that satisfies the constraint to the individual
        filter_coords.append(deepcopy(random_coords))
        coords_before_norm = deepcopy(random_coords)
        coords_before_norm[1] -= left_cell[1]
        coords_before_norm = coords_before_norm / np.append(mid_cell, 1)  # The last number is now atomic number
        ind.append(coords_before_norm.tolist())

        # Append the neighbor image atom coordinates to the neighbor list
        up_neighbor_img = deepcopy(random_coords)
        up_neighbor_img[0] += interface_len
        x_up_neighbor.append(up_neighbor_img)
        down_neighbor_img = deepcopy(random_coords)
        down_neighbor_img[0] -= interface_len
        x_down_neighbor.append(down_neighbor_img)

    return ind

def ind_creator_ama(interface_len: float, min_width: float, max_width: float, max_height: float, cell_height: float, atom_type_list: list, left_atom_obj: Atoms, right_atom_obj: Atoms, inter_atom_limit: float, filter_range: float = 3.5, loop_limit: int = 10000) -> list:
    """
    Create an individual (GB structure) with as many atoms as possible while maintaining inter-atomic distances larger than a specified value.

    Parameters:
    interface_len (float): Length of the interface.
    min_width (float): Minimum width of the interface.
    max_width (float): Maximum width of the interface.
    max_height (float): Maximum height of the interface.
    cell_height (float): Height of the cell.
    atom_type_list (list): List of possible atom types.
    left_atom_obj (Atoms): Left atom object.
    right_atom_obj (Atoms): Right atom object.
    inter_atom_limit (float): Maximum allowed inter-atomic distance.
    filter_range (float): Range for filtering. Default is 3.5.
    loop_limit (int): Maximum number of iterations to find a new atom. Default is 10000.

    Returns:
    list: Generated individual with as many atoms as possible while maintaining specified inter-atomic distances.
    """
    # Decide the cell size and the planar atom density
    width = min_width + (max_width - min_width) * random.random()

    # Decide the atom range in the z direction in Cartesian coordinates
    bottom_limit = cell_height / 2 - max_height / 2
    upper_limit = cell_height / 2 + max_height / 2

    # Decide the cell for calculating the Cartesian coordinates later
    mid_cell = np.array([interface_len, width, cell_height])

    # Make cell a one-dimensional vector
    left_cell = [left_atom_obj.get_cell()[0][0], left_atom_obj.get_cell()[1][1], left_atom_obj.get_cell()[2][2]]
    left_cell = np.array(left_cell)
    left_ori_atom_pos = left_atom_obj.get_positions()

    right_cell = [right_atom_obj.get_cell()[0][0], right_atom_obj.get_cell()[1][1], right_atom_obj.get_cell()[2][2]]
    right_ori_atom_pos = right_atom_obj.get_positions()

    # Decide the atom coordinates for the right side
    right_attatched_pos = right_ori_atom_pos + np.tile(np.array([0, left_cell[1] + mid_cell[1], 0]),
                                                       (right_ori_atom_pos.shape[0], 1))

    # From the coordinates array generate the array that contains atomic numbers
    # Number at the end of each line is the atomic number corresponding to that atom coordinates
    left_coords = np.c_[left_ori_atom_pos, left_atom_obj.get_atomic_numbers()]
    right_coords = np.c_[right_attatched_pos, right_atom_obj.get_atomic_numbers()]

    # Generate the list for the individual & append the cell to the first row
    ind = [mid_cell.tolist()]

    # Build an array including the atoms close to the middle part for distance checking
    filter_coords = []
    for atom in left_coords:
        if left_cell[1] - filter_range < atom[1] <= left_cell[1]:
            filter_coords.append(atom)

    for atom in right_coords:
        if left_cell[1] + mid_cell[1] < atom[1] <= left_cell[1] + mid_cell[1] + filter_range:
            filter_coords.append(atom)

    # Build the atom coordinate array that contains all the coordinates for the neighbors of the middle atom
    # Here only the periodic image in the x direction is considered
    x_up_neighbor = []
    x_down_neighbor = []

    # Determine whether to use the covalent radius from ASE or use the user-defined inter-atom distance limitation
    try:
        if 'covalent radius' in inter_atom_limit:
            use_covalent_dist = True
        else:
            raise TypeError('The inter-atom limit parameter should either be covalent radius or an integer!')
    except TypeError:
        use_covalent_dist = False

    # Add more atoms that satisfy the distance requirement
    # Add as many atoms as possible to the middle part
    # If after loop_limit iterations, still not find a new atom, then give up
    continue_find = True
    while continue_find:
        loop_num = 0
        satisfy_condition = False
        while not satisfy_condition:
            # Randomly generate an atom coordinate
            random_coords = np.array([random.uniform(0, left_cell[0]),
                                      random.uniform(left_cell[1], left_cell[1] + width),
                                      random.uniform(bottom_limit, upper_limit),
                                      random.choice(atom_type_list)])

            # Check the distance between existing atoms (including the x direction neighbor)
            # and the randomly generated new atom
            if x_up_neighbor and x_down_neighbor:
                dist_control_atoms = np.vstack((filter_coords, x_up_neighbor, x_down_neighbor))
            else:
                dist_control_atoms = filter_coords

            for atom_line in dist_control_atoms:
                # Determine the minimum distance between atoms
                if use_covalent_dist:
                    min_covalent_len = covalent_radii[int(atom_line[-1])] + covalent_radii[int(random_coords[-1])]
                else:
                    min_covalent_len = inter_atom_limit

                # Calculate the distance and decide whether it is larger than the minimum covalent bond length
                dist = np.linalg.norm(random_coords[:3] - atom_line[:3])
                if dist >= min_covalent_len:
                    satisfy_condition = True
                else:
                    # If there's one atom that has a distance less than this new atom
                    # Break the for loop and generate a new one
                    satisfy_condition = False
                    loop_num += 1
                    break

            # If loop number reaches the limitation, set continue_find to False
            if loop_num > loop_limit:
                continue_find = False
                break

        # Append the atom that satisfies the constraint to the individual
        # The coordinates need to be modified so only the middle part remains
        # Use deepcopy so the coordinates won't be changed later for the same object
        if continue_find:
            filter_coords.append(deepcopy(random_coords))
            coords_before_norm = deepcopy(random_coords)
            coords_before_norm[1] -= left_cell[1]
            coords_before_norm = coords_before_norm / np.append(mid_cell, 1)  # The last number is now atomic number
            ind.append(coords_before_norm.tolist())

            # Append the neighbor image atom coordinates to the neighbor list
            up_neighbor_img = deepcopy(random_coords)
            up_neighbor_img[0] += interface_len
            x_up_neighbor.append(up_neighbor_img)
            down_neighbor_img = deepcopy(random_coords)
            down_neighbor_img[0] -= interface_len
            x_down_neighbor.append(down_neighbor_img)

    return ind

def ind_creator_amap(interface_len: float, min_width: float, max_width: float, max_height: float, cell_height: float, atom_type_list: list, left_atom_obj: Atoms, right_atom_obj: Atoms, inter_atom_limit: float, filter_range: float = 3.5, loop_limit: int = 10000) -> list:
    """
    Create an individual (GB structure) by stuffing as many atoms as possible without breaking the inter-atomic limitation.

    Parameters:
    interface_len (float): Length of the interface.
    min_width (float): Minimum width of the interface.
    max_width (float): Maximum width of the interface.
    max_height (float): Maximum height of the interface.
    cell_height (float): Height of the cell.
    atom_type_list (list): List of possible atom types.
    left_atom_obj (Atoms): Left atom object.
    right_atom_obj (Atoms): Right atom object.
    inter_atom_limit (float): Maximum allowed inter-atomic distance.
    filter_range (float): Range for filtering. Default is 3.5.
    loop_limit (int): Maximum number of iterations to find a new atom. Default is 10000.

    Returns:
    list: Generated individual with as many atoms as possible while maintaining specified inter-atomic distances.
    """
    # Decide the cell size and the planar atom density
    width = min_width + (max_width - min_width) * random.random()

    # Decide the atom range in the z direction in Cartesian coordinates
    bottom_limit = cell_height / 2 - max_height / 2
    upper_limit = cell_height / 2 + max_height / 2

    # Decide the cell for calculating the Cartesian coordinates later
    mid_cell = np.array([interface_len, width, cell_height])

    # Make cell a one-dimensional vector
    left_cell = [left_atom_obj.get_cell()[0][0], left_atom_obj.get_cell()[1][1], left_atom_obj.get_cell()[2][2]]
    left_cell = np.array(left_cell)
    left_ori_atom_pos = left_atom_obj.get_positions()

    right_cell = [right_atom_obj.get_cell()[0][0], right_atom_obj.get_cell()[1][1], right_atom_obj.get_cell()[2][2]]
    right_cell = np.array(right_cell)
    right_ori_atom_pos = right_atom_obj.get_positions()

    # Decide the atom coordinates for the left side
    left_attached_pos = left_ori_atom_pos + np.tile(np.array([0, right_cell[1] + mid_cell[1], 0]),
                                                    (left_ori_atom_pos.shape[0], 1))

    # From the coordinates array generate the array that contains atomic numbers
    # Number at the end of each line is the atomic number corresponding to that atom coordinates
    right_coords = np.c_[right_ori_atom_pos, right_atom_obj.get_atomic_numbers()]
    left_coords = np.c_[left_attached_pos, left_atom_obj.get_atomic_numbers()]

    # Generate the list for the individual & append the cell to the first row
    ind = [mid_cell.tolist()]

    # Build an array including the atoms close to the middle part for distance checking
    filter_coords = []
    for atom in left_coords:
        if right_cell[1] + mid_cell[1] < atom[1] <= right_cell[1] + mid_cell[1] + filter_range:
            filter_coords.append(atom)

    for atom in right_coords:
        if right_cell[1] - filter_range < atom[1] <= right_cell[1]:
            filter_coords.append(atom)

    # Build the atom coordinate array that contains all the coordinates for the neighbors of the middle atom
    # Here only the periodic image in the x direction is considered
    x_up_neighbor = []
    x_down_neighbor = []

    # Determine whether to use the covalent radius from ASE or use the user-defined inter-atom distance limitation
    try:
        if 'covalent radius' in inter_atom_limit:
            use_covalent_dist = True
        else:
            raise TypeError('The inter-atom limit parameter should either be covalent radius or an integer!')
    except TypeError:
        use_covalent_dist = False

    # Add more atoms that satisfy the distance requirement
    # Add as many atoms as possible to the middle part
    # If after loop_limit iterations, still not find a new atom, then give up
    continue_find = True
    while continue_find:
        loop_num = 0
        satisfy_condition = False
        while not satisfy_condition:
            # Randomly generate an atom coordinate
            random_coords = np.array([random.uniform(0, left_cell[0]),
                                      random.uniform(left_cell[1], left_cell[1] + width),
                                      random.uniform(bottom_limit, upper_limit),
                                      random.choice(atom_type_list)])

            # Check the distance between existing atoms (including the x direction neighbor)
            # and the randomly generated new atom
            if x_up_neighbor and x_down_neighbor:
                dist_control_atoms = np.vstack((filter_coords, x_up_neighbor, x_down_neighbor))
            else:
                dist_control_atoms = filter_coords

            for atom_line in dist_control_atoms:
                # Determine the minimum distance between atoms
                if use_covalent_dist:
                    min_covalent_len = covalent_radii[int(atom_line[-1])] + covalent_radii[int(random_coords[-1])]
                else:
                    min_covalent_len = inter_atom_limit

                # Calculate the distance and decide whether it is larger than the minimum covalent bond length
                dist = np.linalg.norm(random_coords[:3] - atom_line[:3])
                if dist >= min_covalent_len:
                    satisfy_condition = True
                else:
                    # If there's one atom that has a distance less than this new atom
                    # Break the for loop and generate a new one
                    satisfy_condition = False
                    loop_num += 1
                    break

            # If loop number reaches the limitation, set continue_find to False
            if loop_num > loop_limit:
                continue_find = False
                break

        # Append the atom that satisfies the constraint to the individual
        # The coordinates need to be modified so only the middle part remains
        # Use deepcopy so the coordinates won't be changed later for the same object
        if continue_find:
            filter_coords.append(deepcopy(random_coords))
            coords_before_norm = deepcopy(random_coords)
            coords_before_norm[1] -= left_cell[1]
            coords_before_norm = coords_before_norm / np.append(mid_cell, 1)  # The last number is now atomic number
            ind.append(coords_before_norm.tolist())

            # Append the neighbor image atom coordinates to the neighbor list
            up_neighbor_img = deepcopy(random_coords)
            up_neighbor_img[0] += interface_len
            x_up_neighbor.append(up_neighbor_img)
            down_neighbor_img = deepcopy(random_coords)
            down_neighbor_img[0] -= interface_len
            x_down_neighbor.append(down_neighbor_img)

    return ind

def attach(ind: list, left_atom_obj: Atoms, right_atom_obj: Atoms, cutoff: float = 2.6, pbc: list = [True, True, False], return_graph: bool = True, periodic_struc: bool = True) -> tuple:
    """
    Attach the middle part from GA to the two sides.

    Parameters:
    ind (list): The individual to attach.
    left_atom_obj (Atoms): Left atom object.
    right_atom_obj (Atoms): Right atom object.
    cutoff (float): Bond length cutoff. Default is 2.6.
    pbc (list): Periodic boundary conditions. Default is [True, True, False].
    return_graph (bool): Whether to return the graph object. Default is True.
    periodic_struc (bool): Whether the structure is periodic. Default is True.

    Returns:
    tuple: A tuple containing the complete graph (optional), the complete atoms object, and the rotated atoms object (optional).
    """
    # Make cell a one-dimensional vector
    left_cell = np.array([left_atom_obj.get_cell()[0][0], left_atom_obj.get_cell()[1][1], left_atom_obj.get_cell()[2][2]])
    left_ori_atom_pos = left_atom_obj.get_positions()

    right_cell = np.array([right_atom_obj.get_cell()[0][0], right_atom_obj.get_cell()[1][1], right_atom_obj.get_cell()[2][2]])
    right_ori_atom_pos = right_atom_obj.get_positions()

    # Get the cell size for the two side graphs
    middle_cell = np.array(ind[0])
    try:
        mid_old_reduced_atom_pos = np.delete(np.array(ind[1:]), -1, 1)
    except np.AxisError:
        if return_graph:
            return None, None, None
        else:
            return None, None

    mid_ori_atom_pos = mid_old_reduced_atom_pos * np.tile(middle_cell, (len(ind) - 1, 1))

    # Change the atom coordinates to fit the complete cell
    mid_attached_pos = mid_ori_atom_pos + np.tile(np.array([0, right_cell[1], 0]), (mid_ori_atom_pos.shape[0], 1))
    left_attached_pos = left_ori_atom_pos + np.tile(np.array([0, right_cell[1] + middle_cell[1], 0]), (left_ori_atom_pos.shape[0], 1))

    # Generate the rotated atom objects
    if periodic_struc:
        # Find the center point to rotate according to
        center_vec = middle_cell / 2

        # Calculate the translation vector for each atom
        trans_vec = center_vec - mid_ori_atom_pos

        # Translate all atoms two times the translation vector to make them center symmetric with the original
        rotated_mid_ori_pos = mid_ori_atom_pos + 2 * trans_vec

        # Generate the atom positions that need to be attached
        rotated_mid_attached_pos = rotated_mid_ori_pos + np.tile(np.array([0, left_cell[1] + middle_cell[1] + right_cell[1], 0]), (mid_ori_atom_pos.shape[0], 1))

        # Generate the complete list of atoms
        complete_pos = np.vstack((right_ori_atom_pos, mid_attached_pos, left_attached_pos, rotated_mid_attached_pos))

        mid_atomic_num = np.array(ind[1:])[:, -1]
        rotate_obj = Atoms(mid_atomic_num, positions=rotated_mid_ori_pos, cell=middle_cell, pbc=pbc)
    else:
        # Generate the complete list of atoms
        complete_pos = np.vstack((right_ori_atom_pos, mid_attached_pos, left_attached_pos))
        rotate_obj = None

    # Calculate the pairs that are within the bond length cutoff for the original cell
    cell = np.array([left_cell[0], left_cell[1] + 2 * middle_cell[1] + right_cell[1], left_cell[2]])
    atom_num = len(complete_pos)

    # Generate an atoms object using ASE
    left_atomic_num = left_atom_obj.get_atomic_numbers()
    right_atomic_num = right_atom_obj.get_atomic_numbers()
    mid_atomic_num = np.array(ind[1:])[:, -1]
    atomic_num_array = np.concatenate((right_atomic_num, mid_atomic_num, left_atomic_num))
    if periodic_struc:
        atomic_num_array = np.concatenate((atomic_num_array, mid_atomic_num))
    complete_obj = Atoms(atomic_num_array, positions=complete_pos, cell=cell, pbc=pbc)

    # Use the parameter 'return_graph' to determine if return also the graph object
    if return_graph:
        # Create the complete graph object and add the edges from two sides to the graph
        complete_graph = nx.Graph(cell=cell)

        for i in range(atom_num):
            feature_vector = np.append(cell, complete_pos[i] / cell)
            complete_graph.add_node(i, x=feature_vector)

            for j in range(atom_num):
                distance = np.linalg.norm(complete_pos[i] - complete_pos[j])
                if distance <= cutoff and distance != 0:
                    complete_graph.add_edge(i, j, distance=distance)

        # Consider the atoms in neighbor cell
        if pbc[0]:
            x_adj_atom_pos = deepcopy(complete_pos)
            x_adj_atom_pos[:, 0] = x_adj_atom_pos[:, 0] + cell[0]
            for i in range(atom_num):
                for j in range(atom_num):
                    distance = np.linalg.norm(complete_pos[i] - x_adj_atom_pos[j])
                    if distance <= cutoff and distance != 0:
                        complete_graph.add_edge(i, j, distance=distance)

        # Consider the atoms in neighbor cell y direction
        if pbc[1]:
            y_adj_atom_pos = deepcopy(complete_pos)
            y_adj_atom_pos[:, 1] = y_adj_atom_pos[:, 1] + cell[1]
            for i in range(atom_num):
                for j in range(atom_num):
                    distance = np.linalg.norm(complete_pos[i] - y_adj_atom_pos[j])
                    if distance <= cutoff and distance != 0:
                        complete_graph.add_edge(i, j, distance=distance)

        # Consider the atoms in z direction neighbor cell
        if pbc[2]:
            z_adj_atom_pos = deepcopy(complete_pos)
            z_adj_atom_pos[:, 2] = z_adj_atom_pos[:, 2] + cell[2]
            for i in range(atom_num):
                for j in range(atom_num):
                    distance = np.linalg.norm(complete_pos[i] - z_adj_atom_pos[j])
                    if distance <= cutoff and distance != 0:
                        complete_graph.add_edge(i, j, distance=distance)

        return complete_graph, complete_obj, rotate_obj

    else:
        return complete_obj, rotate_obj

def cross_over_1pt(parent_1: list, parent_2: list, cut_loc_mu: float, cut_loc_sigma: float) -> tuple:
    """
    Perform one-point crossover between two parent individuals to produce two children.

    Parameters:
    parent_1 (list): First parent individual.
    parent_2 (list): Second parent individual.
    cut_loc_mu (float): Mean value of the cut location (fractional) coordinate for a Gaussian distribution.
    cut_loc_sigma (float): Sigma of the Gaussian distribution.

    Returns:
    tuple: Two child individuals resulting from the crossover.
    """
    # Either slice the x or y direction
    dimension_index = random.randint(0, 1)

    # Take the lengths of two parents
    length_1 = len(parent_1)
    length_2 = len(parent_2)

    # Randomly decide the new cell size
    interface_len = random.uniform(parent_1[0][0], parent_2[0][0])
    width = random.uniform(parent_1[0][1], parent_2[0][1])
    height = random.uniform(parent_1[0][2], parent_2[0][2])

    # Make temporary copy of parents and clear the original parents lists
    copy_1 = deepcopy(parent_1)
    copy_2 = deepcopy(parent_2)
    parent_1.clear()
    parent_2.clear()

    # Put the cell size into the children
    parent_1.append([interface_len, width, height])
    parent_2.append([interface_len, width, height])

    # Decide the cut point and make a copy of the original
    cut_point = random.gauss(cut_loc_mu, cut_loc_sigma)
    while cut_point > 1 or cut_point < 0:
        cut_point = random.gauss(cut_loc_mu, cut_loc_sigma)

    # Put the sliced parts in the offspring
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

def structure_mutation(ind: list, frac_atom: float, max_height: float, std: float) -> list:
    """
    Perform structure mutation on an individual by perturbing a given fractional number of atoms with a Gaussian distribution.

    Parameters:
    ind (list): The individual to mutate.
    frac_atom (float): Fraction of atoms to mutate.
    max_height (float): Maximum height of the cell.
    std (float): Standard deviation for the Gaussian distribution.

    Returns:
    list: The mutated individual.
    """
    # Determine how many atoms to mutate according to the frac_atom parameter
    atom_num = len(ind)
    mutate_atom_num = random.randint(1, atom_num - 1)

    # Get the cell size and limit for atom height in fractional coordinates
    interface_len = ind[0][0]
    width = ind[0][1]
    cell_height = ind[0][2]
    bottom_limit = (cell_height / 2 - max_height / 2) / cell_height
    upper_limit = (cell_height / 2 + max_height / 2) / cell_height

    # Copy and clear the original individual
    temp_copy = deepcopy(ind)
    ind.clear()

    # Decide a list of line numbers to make the change
    full_list = list(range(1, atom_num))
    if atom_num == mutate_atom_num:
        line_num = full_list
    else:
        line_num = random.sample(full_list, mutate_atom_num)

    # Change the coordinates
    for line in line_num:
        # Consider 3-dimensional mutation, compute the fractional distance
        interface_mutation_dis = random.gauss(0, std) / interface_len
        width_mutation_dis = random.gauss(0, std) / width
        height_mutation_dis = random.gauss(0, std) / cell_height

        # Update the interface coordinate
        if interface_mutation_dis + temp_copy[line][0] < 0:
            temp_copy[line][0] = 0.000001  # Avoid boundary coordinate
        elif 0 <= interface_mutation_dis + temp_copy[line][0] <= 1:
            temp_copy[line][0] += interface_mutation_dis
        else:
            temp_copy[line][0] = 0.999999

        # Update the width coordinate
        if width_mutation_dis + temp_copy[line][1] < 0:
            temp_copy[line][1] = 0.000001
        elif 0 <= width_mutation_dis + temp_copy[line][1] <= 1:
            temp_copy[line][1] += width_mutation_dis
        else:
            temp_copy[line][1] = 0.999999

        # Update the height coordinate
        if height_mutation_dis + temp_copy[line][2] < bottom_limit:
            temp_copy[line][2] = bottom_limit
        elif bottom_limit <= height_mutation_dis + temp_copy[line][2] <= upper_limit:
            temp_copy[line][2] += height_mutation_dis
        else:
            temp_copy[line][2] = upper_limit

    # Copy changed individual back into the original list
    for atom in range(atom_num):
        ind.append(temp_copy[atom])

    return ind

def atom_num_mutation(ind: list, sigma: float, min_atom_num: int, max_atom_num: int, max_height: float, atom_type_list: list, mean_value: float = 0) -> list:
    """
    Perform atom number mutation on an individual.

    Parameters:
    ind (list): The individual to mutate.
    sigma (float): Standard deviation for the Gaussian distribution.
    min_atom_num (int): Minimum number of atoms allowed in the individual.
    max_atom_num (int): Maximum number of atoms allowed in the individual.
    max_height (float): Maximum height of the cell.
    atom_type_list (list): List of possible atom types.
    mean_value (float): Mean value for the Gaussian distribution. Default is 0.

    Returns:
    list: The mutated individual.
    """
    # Get the length of the individual and a list containing the labels of each line (excluding the first line)
    ind_len = len(ind)
    full_list = list(range(1, ind_len))

    # Decide the atom change number using an integer Gaussian distribution
    atom_change_num = 0
    while not (atom_change_num != 0 and min_atom_num <= (ind_len - 1 + atom_change_num) <= max_atom_num):
        atom_change_num = round(random.gauss(mean_value, sigma))

    # Remove or add atoms depending on the value of atom_change_num
    if atom_change_num < 0:
        del_num = abs(atom_change_num)
        while del_num > 0:
            ind_len = len(ind)
            ind.pop(random.randint(1, ind_len - 1))
            del_num -= 1
        return ind
    else:
        # Get the cell size and limit for atom height in fractional coordinates
        cell_height = ind[0][2]
        bottom_limit = (cell_height / 2 - max_height / 2) / cell_height
        upper_limit = (cell_height / 2 + max_height / 2) / cell_height
        for _ in range(atom_change_num):
            ind.append([random.random(), random.random(), random.uniform(bottom_limit, upper_limit),
                        random.choice(atom_type_list)])
        return ind

def assign_struct(ind: list, left_atom_obj: Atoms, right_atom_obj: Atoms, cutoff: float, pbc: list) -> None:
    """
    Assign graph and atom object to the individual.

    Parameters:
    ind (list): The individual to assign structure to.
    left_atom_obj (Atoms): Left atom object.
    right_atom_obj (Atoms): Right atom object.
    cutoff (float): Bond length cutoff.
    pbc (list): Periodic boundary conditions.

    Returns:
    None
    """
    # Attach the middle part from GA to the two sides
    nx_graph, atom_obj, rotate_obj = attach(ind, left_atom_obj, right_atom_obj, cutoff=cutoff, pbc=pbc,
                                            return_graph=True, periodic_struc=True)

    # Assign the graph, atom object, and rotated atom object to the individual
    ind.struct = nx_graph
    ind.atom_obj = atom_obj
    ind.rotate_obj = rotate_obj

def assign_graph(ind: list, left_atom_obj: Atoms, right_atom_obj: Atoms, cutoff: float = 2.6, pbc: list = [True, False, False]) -> nx.Graph:
    """
    Assign a graph object to the individual.

    Parameters:
    ind (list): The individual to assign a graph to.
    left_atom_obj (Atoms): Left atom object.
    right_atom_obj (Atoms): Right atom object.
    cutoff (float): Bond length cutoff. Default is 2.6.
    pbc (list): Periodic boundary conditions. Default is [True, False, False].

    Returns:
    nx.Graph: The complete graph object representing the individual.
    """
    # Make cell a one-dimensional vector
    left_cell = np.array([left_atom_obj.get_cell()[0][0], left_atom_obj.get_cell()[1][1], left_atom_obj.get_cell()[2][2]])
    left_ori_atom_pos = left_atom_obj.get_positions()

    right_cell = np.array([right_atom_obj.get_cell()[0][0], right_atom_obj.get_cell()[1][1], right_atom_obj.get_cell()[2][2]])
    right_ori_atom_pos = right_atom_obj.get_positions()

    # Get the cell size for the two side graphs
    middle_cell = np.array(ind[0])
    mid_old_reduced_atom_pos = np.delete(np.array(ind[1:]), -1, 1)
    mid_ori_atom_pos = mid_old_reduced_atom_pos * np.tile(middle_cell, (len(ind) - 1, 1))

    # Change the atom coordinates to fit the complete cell
    mid_attached_pos = mid_ori_atom_pos + np.tile(np.array([0, left_cell[1], 0]), (mid_ori_atom_pos.shape[0], 1))
    right_attached_pos = right_ori_atom_pos + np.tile(np.array([0, left_cell[1] + middle_cell[1], 0]), (right_ori_atom_pos.shape[0], 1))

    # Generate the complete list of atoms
    complete_pos = np.vstack((left_ori_atom_pos, mid_attached_pos, right_attached_pos))

    # Calculate the pairs that are within the bond length cutoff for the original cell
    cell = np.array([left_cell[0], left_cell[1] + middle_cell[1] + right_cell[1], left_cell[2]])
    atom_num = len(complete_pos)

    # Create the complete graph object and add the edges from two sides to the graph
    complete_graph = nx.Graph(cell=cell)
    for i in range(atom_num):
        feature_vector = np.append(cell, complete_pos[i] / cell)
        complete_graph.add_node(i, x=feature_vector)
        for j in range(atom_num):
            distance = np.linalg.norm(complete_pos[i] - complete_pos[j])
            if distance <= cutoff and distance != 0:
                complete_graph.add_edge(i, j, distance=distance)

    # Consider the atoms in neighbor cell (x direction)
    if pbc[0]:
        x_adj_atom_pos = deepcopy(complete_pos)
        x_adj_atom_pos[:, 0] += cell[0]
        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(complete_pos[i] - x_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    complete_graph.add_edge(i, j, distance=distance)

    # Consider the atoms in neighbor cell (y direction)
    if pbc[1]:
        y_adj_atom_pos = deepcopy(complete_pos)
        y_adj_atom_pos[:, 1] += cell[1]
        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(complete_pos[i] - y_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    complete_graph.add_edge(i, j, distance=distance)

    # Consider the atoms in neighbor cell (z direction)
    if pbc[2]:
        z_adj_atom_pos = deepcopy(complete_pos)
        z_adj_atom_pos[:, 2] += cell[2]
        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(complete_pos[i] - z_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    complete_graph.add_edge(i, j, distance=distance)

    ind.struct = complete_graph
    return ind.struct

def assign_graph_ind(ind_1: list, cutoff: float = 2.6, pbc: list = [True, False, False]) -> nx.Graph:
    """
    Assign a graph object to the individual.

    Parameters:
    ind_1 (list): The individual to assign a graph to.
    cutoff (float): Bond length cutoff. Default is 2.6.
    pbc (list): Periodic boundary conditions. Default is [True, False, False].

    Returns:
    nx.Graph: The complete graph object representing the individual.
    """
    # Create the graph object for the individual
    g_1 = nx.Graph()

    # Get the cell size and atom positions for the individual
    ind_1_cell = np.array(ind_1[0])
    try:
        ind_1_positions = np.delete(np.array(ind_1[1:]), -1, 1) * ind_1_cell
    except np.AxisError:
        ind_1.struct = g_1
        return ind_1.struct

    # Calculate the pairs that are within the bond length cutoff for the original cell
    atom_num = len(ind_1_positions)
    for i in range(atom_num):
        g_1.add_node(i)
        for j in range(atom_num):
            distance = np.linalg.norm(ind_1_positions[i] - ind_1_positions[j])
            if distance <= cutoff and distance != 0:
                g_1.add_edge(i, j)

    # Consider the atoms in neighbor cell (x direction)
    if pbc[0]:
        x_adj_atom_pos = deepcopy(ind_1_positions)
        x_adj_atom_pos[:, 0] += ind_1_cell[0]
        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(ind_1_positions[i] - x_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_1.add_edge(i, j)

    # Consider the atoms in neighbor cell (y direction)
    if pbc[1]:
        y_adj_atom_pos = deepcopy(ind_1_positions)
        y_adj_atom_pos[:, 1] += ind_1_cell[1]
        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(ind_1_positions[i] - y_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_1.add_edge(i, j)

    # Consider the atoms in neighbor cell (z direction)
    if pbc[2]:
        z_adj_atom_pos = deepcopy(ind_1_positions)
        z_adj_atom_pos[:, 2] += ind_1_cell[2]
        for i in range(atom_num):
            for j in range(atom_num):
                distance = np.linalg.norm(ind_1_positions[i] - z_adj_atom_pos[j])
                if distance <= cutoff and distance != 0:
                    g_1.add_edge(i, j)

    ind_1.struct = g_1
    return ind_1.struct

def merge_atom(ind: object, tol: float = 0.01) -> None:
    """
    Merge atoms in the individual that are too close to each other.

    Parameters:
    ind (object): The individual whose atoms need to be merged.
    tol (float): Tolerance for merging atoms. Default is 0.01.

    Returns:
    None
    """
    ind.struct.merge_sites(tol)

def write_ind(ind: list, path: str) -> None:
    """
    Write the individual to a file for debugging.

    Parameters:
    ind (list): The individual to write to the file.
    path (str): The file path where the individual will be written.

    Returns:
    None
    """
    # Extract cell dimensions and atomic positions from the individual
    cell = np.array(ind[0])
    pos = np.array(ind[1:])[:, 0:-1] * cell
    symbol = np.array(ind[1:])[:, -1]

    # Create an Atoms object with the extracted data
    complete_obj = Atoms(symbol, positions=pos, cell=cell, pbc=[1, 1, 1])

    # Write the Atoms object to the specified file path in VASP format
    complete_obj.write(path, format='vasp')

if __name__ == '__main__':
    pass
