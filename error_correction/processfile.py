import time
from collections import Counter
import multiprocessing
from functools import partial
import math
from log import get_logger
from origami_greedy import Origami
# from origamiprepostprocess import OrigamiPrePostProcess
import csv
import random


class ProcessFile(Origami):
    """
    Decoding and encoding will call this class. And this class will call
    the origami method to handle individual origami. This file will also call
    """

    def __init__(self, verbose):
        """
        This will combine all the origami and reconstruct the file
        :param verbose:
        """
        super().__init__(verbose=verbose)
        self.verbose = 1
        self.logger = get_logger(verbose, __name__)
        # Will be updated later during checking number how much redundancy we will need
        self.number_of_bit_per_origami = 29

    def _find_optimum_index_bits(self, bits_needed_to_store, parity_number):
        """
        Find the optimum number of index bits
        :param bits_needed_to_store:
        :param available_capacity:
        :return:
        """
        total_capacity = self.row * self.column
        print(self.get_parity_relation(parity_number))
        checksum_allocation = len(self.get_checksum_relation(parity_number))
        parity_allocation = len(self.get_parity_relation(parity_number))
        available_capacity = total_capacity - checksum_allocation - parity_allocation - 4

        for i in range(1, available_capacity):
            capacity_after_index = available_capacity - i
            index_bit_required = math.ceil(bits_needed_to_store / capacity_after_index)
            if 2**i >= index_bit_required:
                return i, capacity_after_index, index_bit_required
        raise Exception("File size is to large to store in the given capacity")

    def encode(self, file_in, file_out, formatted_output=False, parity_number=40):
        """
        Encode the file
        :param file_in: File that need to be encoded
        :param file_out: File where output will be saved
        :param formatted_output: Output will written as a matrix
        :return:
        """
        try:
            file_in = open(file_in, 'rb')
            file_out = open(file_out, "w")
        except Exception as e:
            self.logger.exception(e)
            self.logger.error("Error opening the file")
            return -1, -1, -1, -1  # simulation file expect this format
        data = file_in.read()
        file_in.close()
        # Converting data into binary
        data_in_binary = ''.join(format(letter, '08b') for letter in data)
        # divide the origami based on number of bit per origami

        bits_needed_to_store = len(data_in_binary)
        index_bits, data_bit, segment_size = self._find_optimum_index_bits(bits_needed_to_store, parity_number)
        print("<-----------index bits----------->")
        print(index_bits)
        print("<-------------------------------->")
        print("<-----------data bit----------->")
        print(data_bit)
        print("<---------------segment_size----------------->")
        print(segment_size)
        # Divide into origami from datastream
        for index in range(segment_size):
            start = index * data_bit
            end = start + data_bit
            origami_bits = data_in_binary[start:end].ljust(data_bit, '0')  # pad if needed

            encoded_stream = self._encode(origami_bits, index, data_bit, parity_number)

            if formatted_output:
                print(f"Matrix -> {index}", file=file_out)
                self.print_matrix(self.data_stream_to_matrix(encoded_stream), in_file=file_out)
            else:
                file_out.write(encoded_stream + '\n')

        file_out.close()
        self.logger.info("Encoding done")
        return segment_size, data_bit

    def single_origami_decode(self, single_origami, ior_file_name, correct_dictionary, common_parity_index,
                          minimum_temporary_weight, maximum_number_of_error, false_positive,
                          induced_errors, errors_positions):
        start_time = time.time()
        index, origami_data = single_origami

        self.logger.info("Decoding origami (%d): %s", index, origami_data)
        if len(origami_data) != self.row * self.column:
            self.logger.warning("Origami (%d) is incomplete. Expected length: %d, Found: %d",
                                index, self.row * self.column, len(origami_data))
            return

        try:
            decoded_matrix = super().decode(origami_data, common_parity_index,
                                            minimum_temporary_weight, maximum_number_of_error,
                                            false_positive)
        except Exception as e:
            self.logger.exception("Decoding failed for origami (%d): %s", index, str(e))
            return

        if decoded_matrix == -1:
            self.logger.warning("Decoding unsuccessful for origami (%d)", index)
            return

        decoded_index = decoded_matrix['index']
        decoded_data = decoded_matrix['binary_data']
        error_count = decoded_matrix['total_probable_error']
        error_locations = decoded_matrix['probable_error_locations']
        orientation = decoded_matrix.get('orientation', 'N/A')

        self.logger.info("Recovered origami index: %s | Data: %s", decoded_index, decoded_data)
        if error_count > 0:
            self.logger.info("Detected %d errors at positions: %s", error_count, error_locations)
        else:
            self.logger.info("No errors detected")

        # Check correctness
        status = " "
        if correct_dictionary:
            try:
                status = int(correct_dictionary[int(decoded_index)] == decoded_data)
            except Exception as e:
                self.logger.warning("Comparison error for origami (%d): %s", index, str(e))
                status = -1

        # Write individual origami result
        if ior_file_name:
            decoding_time = round(time.time() - start_time, 3)
            decoded_stream = self.matrix_to_data_stream(decoded_matrix['matrix'])
            log_entry = (
                f"{index},{origami_data},{induced_errors},{errors_positions},{status},{error_count},"
                f"{str(error_locations).replace(',', ' ')},{orientation},{decoded_index},"
                f"{decoded_stream},{decoded_data},{decoding_time}\n"
            )
            try:
                with open(ior_file_name, "a") as ior_file:
                    ior_file.write(log_entry)
            except Exception as e:
                self.logger.error("Failed to write to IOR file for origami (%d): %s", index, str(e))

        return [decoded_matrix, status]


    def decode(self, data, induced_errors, errors_positions, file_out, file_size, parity_number,
            threshold_data, threshold_parity, maximum_number_of_error,
            individual_origami_info, false_positive, correct_file=False):

        print("Errors Positions", errors_positions)

        correct_origami = 0
        incorrect_origami = 0
        total_error_fixed = 0

        ior_file_name = f"{file_out}_ior.csv" if individual_origami_info else None
        if ior_file_name:
            try:
                with open(ior_file_name, "a") as ior_file:
                    ior_file.write(
                        "Line number in file, origami, Induced Errors, Errors Positions, status,error,error location,"
                        "orientation,decoded index,decoded origami, decoded data,decoding time\n")
            except Exception as e:
                self.logger.error("IOR file creation failed: %s", e)
                return

        # Calculate matrix and parity details
        index_bits, data_bit, segment_size = self._find_optimum_index_bits(file_size * 8, parity_number)
        self.matrix_details, self.parity_bit_relation, self.checksum_bit_relation = self._matrix_details(data_bit, parity_number)
        self.data_bit_to_parity_bit = self.get_data_bit_to_parity_bit(self.parity_bit_relation)

        # Load correct file if provided
        correct_dictionary = {}
        if correct_file:
            with open(correct_file) as cf:
                for so in cf:
                    ci, cd = self._extract_text_and_index(self.data_stream_to_matrix(so.strip()))
                    correct_dictionary[ci] = cd

        # Prepare data and decoding function
        origami_data = [(i, origami.strip()) for i, origami in enumerate(data)]
        p_single_decode = partial(
            self.single_origami_decode,
            ior_file_name=ior_file_name,
            correct_dictionary=correct_dictionary,
            common_parity_index=threshold_data,
            minimum_temporary_weight=threshold_parity,
            maximum_number_of_error=maximum_number_of_error,
            false_positive=false_positive,
            induced_errors=induced_errors,
            errors_positions=errors_positions
        )

        # Use multiprocessing for decoding
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            return_values = pool.map(p_single_decode, origami_data)

        # Process results
        decoded_dictionary_wno = {}
        for result in return_values:
            if result and result[0]:
                index = result[0]['index']
                binary_data = result[0]['binary_data']
                decoded_dictionary_wno.setdefault(index, []).append(binary_data)
                total_error_fixed += int(result[0]['total_probable_error'])

                if correct_file:
                    if result[1]:
                        correct_origami += 1
                    else:
                        incorrect_origami += 1

        # Majority vote to recover original data
        final_data = [None] * segment_size
        for idx, binaries in decoded_dictionary_wno.items():
            final_data[idx] = Counter(binaries).most_common(1)[0][0]

        # Check for missing parts
        missing_origami = [i for i, val in enumerate(final_data) if val is None]
        if missing_origami:
            return -1, incorrect_origami, correct_origami, total_error_fixed, missing_origami

        # Reconstruct file
        recovered_binary = "".join(final_data)
        # remove padding
        recovered_binary = recovered_binary[:8 * (len(recovered_binary) // 8)]  

        with open(file_out, "wb") as out_file:
            for i in range(0, len(recovered_binary), 8):
                byte = int(recovered_binary[i:i+8], 2)
                # skip trailing padding
                if byte == 0 and i + 8 == len(recovered_binary): 
                    continue
                out_file.write(bytes([byte]))

        self.logger.info("Number of missing origami: %s", missing_origami)
        self.logger.info("Total error fixed: %s", total_error_fixed)
        self.logger.info("File recovery was successful")

        return 1, incorrect_origami, correct_origami, total_error_fixed, missing_origami



# This is for debugging purpose
if __name__ == '__main__':
    pass
