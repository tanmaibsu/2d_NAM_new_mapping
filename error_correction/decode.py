import argparse
from processfile import ProcessFile
import os
from pathlib import Path
from utility_methods import flip_bits_exhaustively
import random
import cProfile
import pstats
import math
import csv

def read_args():
    """
    Read the arguments from command line
    :return:
    """
    parser = argparse.ArgumentParser(description="Decode a given origami matrices to a text file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-bulk", "--bulk_folder", help="Folder to decode", default="")
    group.add_argument("-f", "--file_in", help="File to decode")


    parser.add_argument("-o", "--file_out", help="File to write output", required=True)
    parser.add_argument("-fz", "--file_size", help="File size that will be decoded", type=int, default=20)
    parser.add_argument("-pn", "--parity_number", help="Number of Parity to decode", type=int, default=40)
    parser.add_argument('-tp', '--threshold_parity',
                        help='Minimum weight for a parity bit cell to be consider that as an error', default=2, type=int)
    parser.add_argument("-td", "--threshold_data",
                        help='Minimum weight for a data bit cell to be consider as an error', default=2, type=int)
    parser.add_argument("-v", "--verbose", help="Print details on the console. "
                                                "0 -> error, 1 -> debug, 2 -> info, 3 -> warning", default=0, type=int)
    parser.add_argument("-r", "--redundancy", help="How much redundancy was used during encoding",
                        default=50, type=float)
    parser.add_argument("-ior", "--individual_origami_info", help="Store individual origami information",
                        action='store_true', default=True)
    parser.add_argument("-e", "--error", help="Maximum number of error that the algorithm "
                                              "will try to fix", type=int, default=8)
    parser.add_argument("-fp", "--false_positive", help="0 can also be 1.", type=int, default=1)

    parser.add_argument("-d", "--degree", help="Degree old/new", default="new", type=str)

    parser.add_argument("-cf", "--correct_file", help="Original encoded file. Helps to check the status automatically."
                        , type=str, default=False)

    args = parser.parse_args()
    return args

def create_file_name(args):
     ior_file_name = f"{args.file_out}_ior.csv" if args.individual_origami_info else None
     
     if ior_file_name:
            try:
                with open(ior_file_name, "a") as ior_file:
                    ior_file.write(
                        "node, origami data, Error positions, decoded stream, success, decoding time\n")
            except Exception as e:
                self.logger.error("IOR file creation failed: %s", e)
                return


def main():
    args = read_args()
    dnam_decode = ProcessFile(verbose=args.verbose)
    
    encoded_origamis_path = Path(args.bulk_folder)
    
    create_file_name(args)

    def flip_n_bits(binary_str, error_pos):
        """
        Randomly flips 'n' bits in a binary string, but only if the bit is '1'.

        :param binary_str: String containing binary bits (e.g., "1010101").
        :param n: Number of bits to flip.
        :return: Modified binary string with 'n' flipped bits.
        """
        # [1], [1, 31], [1, 31, 34], [1, 31, 34, 19], [1, 31, 34, 19, 76], [1, 31, 34, 19, 76, 48, 30], [1, 31, 34, 19, 76, 48, 30, 61]
        if not binary_str:
            return binary_str  # Return original if empty or no flips needed

        binary_list = list(binary_str)  # Convert string to list (mutable)
        # one_indices = [i for i, bit in enumerate(binary_list) if bit == '1']  # Find indices of '1' bits
        # indices_to_flip = random.sample(one_indices, min(1, len(one_indices)))  # Pick 'n' unique '1' indices
        # error_pos = error_pos + indices_to_flip
        # error_pos = [1], [1, 31], [1, 31, 34], [1, 31, 34, 19], [1, 31, 34, 19, 76], [1, 31, 34, 19, 76, 48, 30], [1, 31, 34, 19, 76, 48, 30, 61]
        for idx in error_pos:
            binary_list[idx] = '0'  # Flip '1' to '0'
        
        print("<--------error pos--------->", error_pos)

        return "".join(binary_list), error_pos

    def convert_to_single_arr(data):
        if len(data) == 1:
            return data 
        
        single_data = ""
        for row in data:
            for elm in row:
                if elm == "0" or elm == "1":
                    single_data += str(elm)

        return [single_data]
    
    def do_exhaustive_test(folder, type):
        orig_idx = 0
        for origami in sorted(folder.iterdir()):
            data_file = open(origami, "r")
            data = data_file.readlines()
            origami_data = convert_to_single_arr(data)
            data_file.close()
            if type == "single_bit":
                n = 1
                # print(os.path.relpath(origami, start=os.getcwd()))
                idx = 0
                origami = origami_data[0]
                for i in range(len(origami)):
                    if origami[i] == "0":
                        continue
                    else:
                        # Flip the bit
                        origami_list = list(origami)
                        origami_list[i] = "0"
                        # Convert back to string
                        origami = ''.join(origami_list)
                        dnam_decode.decode([origami], origami_data[0], orig_idx,i, [i], args.file_out, args.file_size, int(args.parity_number),
                                        threshold_data=args.threshold_data,
                                        threshold_parity=args.threshold_parity,
                                        maximum_number_of_error=args.error,
                                        false_positive=args.false_positive,
                                        individual_origami_info=args.individual_origami_info,
                                        correct_file=args.correct_file)
                        # Unflip the bit
                        origami_list[i] = "1"
                        # Convert back to string
                        origami = ''.join(origami_list)
            elif type == "double_bit":
                origami = origami_data[0]
                for i in range(len(origami)):
                    if origami[i] == "0":
                        continue
                    for j in range(i + 1, len(origami)):
                        if origami[j] == "0":
                            continue
                        
                        origami_list = list(origami)
                        # flip the dual bits
                        origami_list[i] = "0"
                        origami_list[j] = "0"

                        # Convert back to string
                        origami = ''.join(origami_list)
                        idx = [i, j]
                        dnam_decode.decode([origami], origami_data[0], orig_idx, idx, [idx], args.file_out, args.file_size, int(args.parity_number),
                                        threshold_data=args.threshold_data,
                                        threshold_parity=args.threshold_parity,
                                        maximum_number_of_error=args.error,
                                        false_positive=args.false_positive,
                                        individual_origami_info=args.individual_origami_info,
                                        correct_file=args.correct_file)
                        # unflip the bits
                        origami_list[i] = "1"
                        origami_list[j] = "1"
                        # Convert back to string
                        origami = ''.join(origami_list)
            elif type == "triple_bit":
                origami = origami_data[0]
                for i in range(len(origami)):
                    if origami[i] == "0":
                        continue
                    for j in range(i + 1, len(origami)):
                        if origami[j] == "0":
                            continue
                        for k in range(j + 1, len(origami)):
                            if origami[k] == "0":
                                continue

                            origami_list = list(origami)
                            # Flip three bits
                            origami_list[i] = "0"
                            origami_list[j] = "0"
                            origami_list[k] = "0"

                            # Convert back to string
                            origami = ''.join(origami_list)
                            idx = [i, j, k]

                            dnam_decode.decode([origami], origami_data[0], orig_idx, idx, [idx], args.file_out, args.file_size, int(args.parity_number),
                                               threshold_data=args.threshold_data,
                                               threshold_parity=args.threshold_parity,
                                               maximum_number_of_error=args.error,
                                               false_positive=args.false_positive,
                                               individual_origami_info=args.individual_origami_info,
                                               correct_file=args.correct_file)

                            # Unflip three bits
                            origami_list[i] = "1"
                            origami_list[j] = "1"
                            origami_list[k] = "1"

                            # Convert back to string
                            origami = ''.join(origami_list)
            orig_idx = orig_idx + 1
        
    def import_original_origami_list():
        return [
            "01000100011100011101110000100111101010000111000010011001000011010000100001101000",
            "01011100111111110001000100000000111101000010101110111100010011000011000000011001",
            "11110111011111101001001011100011000001100000100001000010001110000011000011100110",
            "00000100101000010101000010000010010100000100100000000000000010000111100000000011"
        ]
    
    def import_original_origami_list_6_nodes():
        return [
            "01000100011010110111101010111001101001010101011011011010010110110101100001100000",
            "00100100001110110111000101110010110101110000111110111110111110011001101001100001",
            "10000001101001011001100011010000110000111110011001101010110011011101100010000010",
            "00110111101100111101110000000111101110001000011011110101101010011011000100010011",
            "00000100011111011101010100010001010101101011101100011010101110010110000000010100",
            "01000010001110110011000000010101111100100110100100000011001011110011000000000101",
        ]
    
    def decode_encoded_wetlab_data(args):
        # === Load the CSV with Python stock csv.reader ===
        file_path = "encoded_6_nodes_wetlab/2025-11-26_mixed_6_nodes_rep_3.csv"   # adjust path if needed
        
        # Import your original origami reference
        # original_origami_list = import_original_origami_list()
        original_origami_list = import_original_origami_list_6_nodes()

        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)  # reads rows into dicts keyed by column names
            rows = list(reader)

        # === Iterate over nodes (ID 0–3) ===
        for node_id in range(6):  # only 0,1,2,3
            # if node_id in [0, 1]:  # preserve your skip condition
            #     continue

            print(f"\n--- Processing Node {node_id} ---")

            # Subset rows for this node
            node_rows = [row for row in rows if row.get("ID") and int(float(row["ID"])) == node_id]

            for row in node_rows:
                # Extract the binary string (strip leading 'b' if necessary)
                binary_string = row["Binary String"]
                if binary_string.startswith("b"):
                    origami_data = binary_string[1:]
                else:
                    origami_data = binary_string
                
                false_negatives = row["False Negatives"]
                false_positives=row["False Positives"]
                
                if int(false_negatives) + int(false_positives) > 9:
                    continue 

                # === Call your decoder ===
                dnam_decode.decode(
                    [origami_data],
                    original_origami_list[node_id],
                    node_id,
                    [],
                    [],
                    args.file_out,
                    args.file_size,
                    int(args.parity_number),
                    threshold_data=args.threshold_data,
                    threshold_parity=args.threshold_parity,
                    maximum_number_of_error=args.error,
                    false_positive=args.false_positive,
                    individual_origami_info=args.individual_origami_info,
                    correct_file=args.correct_file,
                    false_negatives=row["False Negatives"],
                    false_positives=row["False Positives"]
                )


    def decode_in_bulk(encoded_origamis_path):
        max_n_errors_induced = 10
        i = 0

        # [[1], [1, 31], [1, 31, 34], [1, 31, 34, 19], [1, 31, 34, 19, 76], [1, 31, 34, 19, 76, 48], [1, 31, 34, 19, 76, 48, 22], [1, 31, 34, 19, 76, 48, 22, 61]]
        # [[[1], [1, 31], [1, 31, 34], [1, 31, 34, 19], [1, 31, 34, 19, 76], [1, 31, 34, 19, 76, 48], [1, 31, 34, 19, 76, 48, 22], [1, 31, 34, 19, 76, 48, 22, 61]], [[9], [9, 11], [9, 11, 35], [9, 11, 35, 19], [9, 11, 35, 19, 33], [9, 11, 35, 19, 33, 65], [9, 11, 35, 19, 33, 65, 53], [9, 11, 35, 19, 33, 65, 53, 28]], [[5], [5, 66], [5, 66, 44], [5, 66, 44, 10], [5, 66, 44, 10, 77], [5, 66, 44, 10, 77, 31], [5, 66, 44, 10, 77, 31, 26], [5, 66, 44, 10, 77, 31, 26, 58]], [[8], [8, 17], [8, 17, 35], [8, 17, 35, 60], [8, 17, 35, 60, 24], [8, 17, 35, 60, 24, 41], [8, 17, 35, 60, 24, 41, 33], [8, 17, 35, 60, 24, 41, 33, 65]]]
        # error_poss = [[[5], [5, 14], [5, 14, 20], [5, 14, 20, 31], [5, 14, 20, 31, 42], [5, 14, 20, 31, 42, 52], [5, 14, 20, 31, 42, 52, 63], [5, 14, 20, 31, 42, 52, 63, 34]],
        # [[11], [11, 4], [11, 4, 18], [11, 4, 18, 33], [11, 4, 18, 33, 38], [11, 4, 18, 33, 38, 57], [11, 4, 18, 33, 38, 57], [11, 4, 18, 33, 38, 57, 65], [11, 4, 18, 33, 38, 57, 65, 79]],
        # [[7], [7, 14], [7, 14, 22], [7, 14, 22, 31], [7, 14, 22, 31, 37], [7, 14, 22, 31, 37, 48], [7, 14, 22, 31, 37, 48, 58], [7, 14, 22, 31, 37, 48, 58, 78]],
        # [[8],[8, 17], [8, 17, 24], [8, 17, 24, 41], [8, 17, 24, 41, 5], [8, 17, 24, 41, 5, 65], [8, 17, 24, 41, 5, 65, 78], [8, 17, 24, 41, 5, 65, 78, 79]]]
        for origami in sorted(encoded_origamis_path.iterdir()):
            data_file = open(origami, "r")
            data = data_file.readlines()
            data = convert_to_single_arr(data)
            data_file.close()
            err_pos = []
            n = 1
            # print(os.path.relpath(origami, start=os.getcwd()))
            for errors in error_poss[i]:
                origami_data, errors_index = flip_n_bits(data[0], n)
                # err_pos = errors_index
                dnam_decode.decode([origami_data], errors, [], args.file_out, args.file_size, int(args.parity_number),
                                threshold_data=args.threshold_data,
                                threshold_parity=args.threshold_parity,
                                maximum_number_of_error=args.error,
                                false_positive=args.false_positive,
                                individual_origami_info=args.individual_origami_info,
                                correct_file=args.correct_file)
            i += 1

    def decode_single_file():
        print(args.file_in)
        data_file = open(args.file_in, "r")
        print(data_file)
        data = data_file.readlines()
        data = convert_to_single_arr(data)
        data_file.close()
        dnam_decode.decode(data, data[0], 0, 0, [-100], args.file_out, args.file_size, int(args.parity_number),
                                threshold_data=args.threshold_data,
                                threshold_parity=args.threshold_parity,
                                maximum_number_of_error=args.error,
                                false_positive=args.false_positive,
                                individual_origami_info=args.individual_origami_info,
                                correct_file=args.correct_file)

    # if args.bulk_folder != "":
    #     encoded_origamis_path = Path(args.bulk_folder)
    #     decode_in_bulk(encoded_origamis_path)
    # else:
    #     decode_single_file()
    
    # do_exhaustive_test(Path(args.bulk_folder), "single_bit")
    # do_exhaustive_test(Path(args.bulk_folder), "double_bit")
    # do_exhaustive_test(Path(args.bulk_folder), "triple_bit")
    decode_encoded_wetlab_data(args)
    
    


if __name__ == '__main__':
    with cProfile.Profile() as profile:
        main()
    
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()


