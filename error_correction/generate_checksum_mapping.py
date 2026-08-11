"""
This file will create the mapping scheme for dNAM with 8x10 origami
"""
import random
from collections import defaultdict
import argparse
import numpy as np
import origami_design as od

# Origami Dimensions
ROW = 8
COLUMN = 10

def get_parity_positions():
    parity_positions = od.parity_mapping_24()

def get_checksum_pos_for_quardant(base_position):
    checksum_positions = []
    for i in base_position[0]:
        for j in base_position[1]:
            checksum_positions.append((i, j))
    
    return checksum_positions


def generate_checksum_mapping(checksum_positions, parity_positions):
    for pos in checksum_positions:
        get_checksum_pos_for_quardant(pos)

def main():
    checksum_positions = od.checksum_mapping()
    parity_position = get_parity_positions()
    generate_checksum_mapping(checksum_positions, parity_positions)


if __name__ == '__main__':
    main()
