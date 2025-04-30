"""
This file will create the mapping scheme for dNAM with 8x10 origami
"""
import random
from collections import defaultdict
import argparse
import numpy as np
import origami_design as od
from datetime import datetime
import os

# Origami Dimensions
ROW = 8
COLUMN = 10

def read_n_parse_args():
    """
    Read argument from command line
    :return: arguments
    """
    parser = argparse.ArgumentParser(description="create parity mapping based on the number of parity bits.")
    parser.add_argument("-pn", "--parity_number", help="Number of parity for encoding", default=24)
    parser.add_argument("-pc", "--parity_coverage", help="Number of positions required for each parity", default=8)
    args = parser.parse_args()
    return args

def mirror(point, axis):
    x, y = point
    if axis == 'x':
        return (x, COLUMN - 1 - y)
    elif axis == 'y':
        return (ROW - 1 - x, y)
    elif axis == 'xy':
        return (ROW - 1 - x, COLUMN - 1 - y)
    return point

def find_mirrors(parity_position: tuple):
    mirrors = []
    for a in ['x', 'y', 'xy']:
        mirrored_point = mirror(parity_position, a)
        mirrors.append(mirrored_point)
    return mirrors

def define_parity_positions(number_of_parity: int):
    if number_of_parity == 24:
        positions = od.parity_mapping_24()
    elif number_of_parity == 16:
        positions = od.parity_mapping_16()
    elif number_of_parity == 40:
        positions = od.parity_mapping_40()
    
    return positions

def find_unique_positions(parity_positions):
    new_positions = parity_positions
    for parity_index in new_positions:
        mirrors = find_mirrors(parity_index)
        for mirror in mirrors:
            new_positions.remove(mirror)
    return new_positions

def create_matrix():
    all_positions = []
    for i in range(ROW):
        for j in range(COLUMN):
            all_positions.append((i, j))
    return all_positions

def get_except_parity(all_positions, args):
    parity_pos = define_parity_positions(int(args.parity_number))
    for pos in parity_pos:
        all_positions.remove(pos)
    
    return all_positions

# Helper function to check if a point or its mirrors are in selected set
def is_valid(point, selected):
    mirrors = [
        mirror(point, 'x'),
        mirror(point, 'y'),
        mirror(point, 'xy')
    ]
    return all(m not in selected for m in mirrors) 

# Now pick 8 non-mirrored points for each parity position
def generate_seeds_4_unique(unique_parity_positions, all_except_parity, parity_coverage):
    parity_to_positions = {}

    for parity in unique_parity_positions:
        available = all_except_parity.copy()
        selected = set()
        
        while len(selected) < parity_coverage and available:
            point = random.choice(available)
            if is_valid(point, selected):
                selected.add(point)
                # Also remove its mirrors from availability to avoid picking them
                mirrors = [
                    mirror(point, 'x'),
                    mirror(point, 'y'),
                    mirror(point, 'xy')
                ]
                for m in mirrors:
                    if m in available:
                        available.remove(m)
                if point in available:
                    available.remove(point)
            else:
                available.remove(point)
        
        parity_to_positions[parity] = list(selected)
    return parity_to_positions

def generate_mirrors_across_axis(seed_of_unique_positions, axis):
    all_positions = seed_of_unique_positions
    new_set = {}
    for pos, values in all_positions.items():
        mirrors = []
        pos_mirror = mirror(pos, axis)
        for _, (x, y) in enumerate(values):
            mirrors.append(mirror((x, y), axis))
        new_set[pos_mirror] = mirrors
    return new_set

def generate_all(seed_of_unique_positions):
    x_mirrors = generate_mirrors_across_axis(seed_of_unique_positions, "x")
    y_mirrors = generate_mirrors_across_axis(seed_of_unique_positions, "y")
    xy_mirrors = generate_mirrors_across_axis(seed_of_unique_positions, "xy")

    copy_of_unique_positions = seed_of_unique_positions.copy()
    copy_of_unique_positions.update(x_mirrors)
    copy_of_unique_positions.update(y_mirrors)
    copy_of_unique_positions.update(xy_mirrors)
    return copy_of_unique_positions


def create_file_name(parity_number):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"parity_mapping_{parity_number}_{timestamp}.txt"
    return file_name

def write_in_txt(data, file_name):
    # Write the formatted dictionary content to the file
    with open(file_name, "w") as file:
        file.write("{\n")
        for key, values in data.items():
            file.write(f"\t{key}:")
            file.write(f" {values},\n")
            file.write("\n")
        file.write("}\n")

def main():
    args = read_n_parse_args()
    parity_positions = define_parity_positions(int(args.parity_number))
    copy_parity_positions = parity_positions.copy()
    unique_parity_positions = find_unique_positions(copy_parity_positions)
    print("unique parity positions", unique_parity_positions)
    all_positions = create_matrix()
    print("all_positions", all_positions)
    data_index_orientation_checksum_positions = get_except_parity(all_positions, args)
    print("data_index_orientation_checksum_positions", len(data_index_orientation_checksum_positions))
    seeds_of_unique_positions = generate_seeds_4_unique(unique_parity_positions, data_index_orientation_checksum_positions, int(args.parity_coverage))
    print("seed_of_unique_positions", seeds_of_unique_positions)
    complete_parity_mapping = generate_all(seeds_of_unique_positions)
    print("complete_parity_mapping", complete_parity_mapping)
    file_name = create_file_name(int(args.parity_number))
    write_in_txt(complete_parity_mapping, file_name)
    return complete_parity_mapping


if __name__ == '__main__':
    main()
