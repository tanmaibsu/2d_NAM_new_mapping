import argparse
from processfile import ProcessFile
import os
from pathlib import Path
import random


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
    parser.add_argument("-p", "--parity_number", help="Number of Parity to decode", type=int, default=40)
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


def main():
    args = read_args()
    dnam_decode = ProcessFile(verbose=args.verbose)
    
    encoded_origamis_path = Path(args.bulk_folder)


    # def flip_n_bits(binary_str, n):
    #     """
    #     Randomly flips 'n' bits in a binary string.

    #     :param binary_str: String containing binary bits (e.g., "1010101").
    #     :param n: Number of bits to flip.
    #     :return: Modified binary string with 'n' flipped bits.
    #     """
    #     if not binary_str or n <= 0:
    #         return binary_str  # Return original if empty or no flips needed

    #     binary_list = list(binary_str)  # Convert string to list (mutable)
    #     indices = random.sample(range(len(binary_list)), min(n, len(binary_list)))  # Pick 'n' unique indices

    #     for idx in indices:
    #         binary_list[idx] = '0' if binary_list[idx] == '1' else '1'  # Flip bit

    #     return "".join(binary_list), indices  # Convert back to string

    def flip_n_bits(binary_str, n):
        """
        Randomly flips 'n' bits in a binary string, but only if the bit is '1'.

        :param binary_str: String containing binary bits (e.g., "1010101").
        :param n: Number of bits to flip.
        :return: Modified binary string with 'n' flipped bits.
        """
        if not binary_str or n <= 0:
            return binary_str  # Return original if empty or no flips needed

        binary_list = list(binary_str)  # Convert string to list (mutable)
        one_indices = [i for i, bit in enumerate(binary_list) if bit == '1']  # Find indices of '1' bits
        indices_to_flip = random.sample(one_indices, min(n, len(one_indices)))  # Pick 'n' unique '1' indices

        for idx in indices_to_flip:
            binary_list[idx] = '0'  # Flip '1' to '0'

        return "".join(binary_list), indices_to_flip


    def convert_to_single_arr(data):
        if len(data) == 1:
            return data 
        
        single_data = ""
        for row in data:
            for elm in row:
                if elm == "0" or elm == "1":
                    single_data += str(elm)

        return [single_data]

    def decode_in_bulk(encoded_origamis_path):

        max_n_errors_induced = 9
        for origami in sorted(encoded_origamis_path.iterdir()):
            data_file = open(origami, "r")
            data = data_file.readlines()
            data = convert_to_single_arr(data)
            data_file.close()
            # print(os.path.relpath(origami, start=os.getcwd()))
            for errors in range(1, max_n_errors_induced+1):
                origami_data, errors_index = flip_n_bits(data[0], errors)
                dnam_decode.decode([origami_data], errors, errors_index, args.file_out, args.file_size, int(args.parity_number),
                                threshold_data=args.threshold_data,
                                threshold_parity=args.threshold_parity,
                                maximum_number_of_error=args.error,
                                false_positive=args.false_positive,
                                individual_origami_info=args.individual_origami_info,
                                correct_file=args.correct_file)

    def decode_single_file():
        print(args.file_in)
        data_file = open(args.file_in, "r")
        print(data_file)
        data = data_file.readlines()
        data = convert_to_single_arr(data)
        data_file.close()
        dnam_decode.decode(data, 0, [-100], args.file_out, args.file_size, int(args.parity_number),
                                threshold_data=args.threshold_data,
                                threshold_parity=args.threshold_parity,
                                maximum_number_of_error=args.error,
                                false_positive=args.false_positive,
                                individual_origami_info=args.individual_origami_info,
                                correct_file=args.correct_file)

    if args.bulk_folder != "":
        encoded_origamis_path = Path(args.bulk_folder)
        decode_in_bulk(encoded_origamis_path)
    else:
        decode_single_file()


if __name__ == '__main__':
    main()
