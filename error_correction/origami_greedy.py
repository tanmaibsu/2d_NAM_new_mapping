from functools import reduce
from collections import Counter
import get_parity_n_checksum as pcm
import copy
import numpy as np
import logging
from log import get_logger


class Origami:
    """
    This class handle individual origami. Both the encoding and decoding is handled by this class. This class is
    inherited by ProcessFile class. The ProcessFile class calls/handel all the method of this class. Each origami is
    represented by a matrix. So the term matrix and origami is used interchangeably.
    """

    def __init__(self, verbose=0):
        """
        :param verbose: is it running on debug mode or info mode
        """
        self.row = 8
        self.column = 10
        self.checksum_bit_per_origami = 4
        self.encoded_matrix = None
        self.recovered_matrix_info = []
        self.list_of_error_combination = []
        self.orientation_details = {
            '0': 'Orientation of the origami is correct',
            '1': 'Origami wasX flipped in horizontal direction',
            '2': 'Origami was flipped in vertical direction.',
            '3': 'Origami was flipped in both direction. '
        }
        self.logger = get_logger(verbose, __name__)
    @staticmethod
    def get_parity_relation(parity_number=40):
        parity_relation = {}
        print(parity_number)
        if parity_number == 16:
            print(" I am here parity 16")
            parity_relation = pcm.parity_mapping_16() 
        elif parity_number == 24:
            parity_relation = pcm.parity_mapping_24() 
        else:
            print(" I am here parity 40")
            parity_relation = pcm.parity_mapping_40() 
        return parity_relation
            
    @staticmethod
    def get_checksum_relation(parity_number=40):
        checksum_relation = {}
        if parity_number == 16:
            print(" I am here checksum 16")
            checksum_relation = pcm.checksum_mapping_16()
        elif parity_number == 24:
            checksum_relation = pcm.checksum_mapping_24()
        else:
            print(" I am here checksum 40")
            checksum_relation = pcm.checksum_mapping_40()
        return checksum_relation

    def _matrix_details(self, data_bit_per_origami: int, parity_number: int) -> object:
        """
        Returns the relationship of the matrix. Currently all the the relationship is hardcoded.
        This method returns the following details:
            parity bits: 20 bits
            indexing bits: depends on the file size
            orientation bits: 4 bits
            checksum bits: 4 bits
            data bits: 48 - indexing bits
        :rtype: object
        :param data_bit_per_origami: Number of bits that will be encoded in each origami
        :returns matrix_details -> Label for each cell
                 parity_bit_relation: Parity bit mapping
                 checksum_bit_relation: Checksum bit mapping

        """
        parity_bit_relation = self.get_parity_relation(parity_number)
        checksum_bit_relation = self.get_checksum_relation(parity_number)

        data_index_orientation = set([i for v in checksum_bit_relation.values() for i in v])
        orientation_bits = set([(1, 0), (1, 9), (6, 0), (6, 9)])
        index_bits = set([(2, 0), (3, 0), (4, 0)])
        data_index = data_index_orientation - orientation_bits
        data_index = sorted(list(data_index))
        print("<----------data_bit_per_origami----------->")
        print(data_bit_per_origami)
        data_bits = data_index[:data_bit_per_origami]
        index_bits = data_index[data_bit_per_origami: ]
        # data_bits = data_index - index_bits

        matrix_details = dict(
            data_bits=list(data_bits),
            orientation_bits=sorted(list(orientation_bits)),
            indexing_bits=list(index_bits),
            checksum_bits=list(checksum_bit_relation.keys()),
            parity_bits=list(parity_bit_relation.keys()),
            orientation_data=[1, 1, 1, 0]
        )
        return matrix_details, parity_bit_relation, checksum_bit_relation

    def create_initial_matrix_from_binary_stream(self, binary_stream: str, index: int) -> object:
        """
        Inserts droplet data, orientation, and index bits into a matrix.

        :param binary_stream: Binary data to encode in the matrix
        :param index: Index value to encode
        :return: Matrix with data, orientation, and index bits inserted
        """
        binary_list = list(binary_stream)
        data_matrix = np.full((self.row, self.column), -1)  # Initialize with -1

        # Insert data bits
        for i, (row, col) in enumerate(self.matrix_details["data_bits"]):
            data_matrix[row][col] = binary_list[i]

        # Insert orientation bits
        for i, (row, col) in enumerate(self.matrix_details["orientation_bits"]):
            data_matrix[row][col] = self.matrix_details['orientation_data'][i]

        # Check if index is within supported range
        max_index = 2 ** len(self.matrix_details["indexing_bits"])
        if index >= max_index:
            self.logger.error(f"Index {index} exceeds supported maximum of {max_index - 1}")
            raise ValueError(f"Index {index} exceeds maximum supported index of {max_index - 1}")

        # Convert index to binary with padding
        index_bits_required = len(self.matrix_details["indexing_bits"])
        index_bin = format(index, f'0{index_bits_required}b')

        # Insert index bits
        for i, (row, col) in enumerate(self.matrix_details["indexing_bits"]):
            data_matrix[row][col] = index_bin[i]

        self.logger.info("Inserted droplet data, orientation, and index bits into the matrix")
        return data_matrix


    @staticmethod
    def _xor_matrix(matrix, relation):
        """
        Applies XOR on matrix data bits using a provided mapping.

        :param matrix: 2D matrix with binary data
        :param relation: Dictionary mapping parity/checksum bit positions to related data bit positions
        :return: Updated matrix with XOR-computed values
        """
        for (parity_row, parity_col), data_bit_positions in relation.items():
            # Extract values from the matrix for the given data bit positions
            data_values = [int(matrix[row][col]) for row, col in data_bit_positions]

            # Compute XOR of all related data bits
            xor_result = reduce(lambda x, y: x ^ y, data_values)

            # Store the XOR result in the corresponding parity/checksum bit position
            matrix[parity_row][parity_col] = xor_result

        return matrix

    def _encode(self, binary_stream, index, data_bit_per_origami, parity_number):
        """
        Handle the encoding. Most of the time handle xoring.
        :param binary_stream: Binary value of the data
        :param index: Index of the current matrix
        :param data_bit_per_origami: Number of bits that will be encoded in each origami
        :return: Encoded matrix
        """
        # Create the initial matrix which will contain the word,index and binary bits for fixing orientation but no
        # error encoding. So the parity bits will have the initial value of -1
        self.number_of_bit_per_origami = data_bit_per_origami
        self.matrix_details, self.parity_bit_relation, self.checksum_bit_relation = \
            self._matrix_details(data_bit_per_origami, parity_number)
        self.data_bit_to_parity_bit = Origami.get_data_bit_to_parity_bit(self.parity_bit_relation)
        encoded_matrix = self.create_initial_matrix_from_binary_stream(binary_stream, index)

        # Set the cell value in checksum bits. This has to be before the parity bit xoring. Cause the parity bit
        # contains the checksum bits. And the default value of the checksum bit is -1. So if the parity xor happens
        # before checksum xor then some of the parity bit will have value negative. as that would be xor with -1
        encoded_matrix = Origami._xor_matrix(encoded_matrix, self.checksum_bit_relation)
        self.logger.info("Finish calculating the checksum")
        # XOR for the parity code
        encoded_matrix = Origami._xor_matrix(encoded_matrix, self.parity_bit_relation)
        self.logger.info("Finish calculating the parity bits")
        return Origami.matrix_to_data_stream(encoded_matrix)

    def encode(self, binary_stream, index, data_bit_per_origami):
        """
        This method will not be called internally. This is added just for testing purpose

        :param binary_stream: Binary value of the data
        :param index: Index of the current matrix
        :param data_bit_per_origami: Number of bits that will be encoded in each origami
        :return: Encoded matrix
        :return:
        """
        return self._encode(binary_stream, index, data_bit_per_origami)

    @staticmethod
    def get_data_bit_to_parity_bit(parity_bit_relation):
        """
        Reverse the parity bit to data bit.
        :param parity_bit_relation: A dictionary that contains parity bit as key and
         the respective indices that will be XORed in the parity bit as value.
        :return: data_bit_to_parity_bit: A dictionary that contains indices as key and and respective parity
         indices that used that indices for XORing.
        """
        data_bit_to_parity_bit = {}
        for single_parity_bit in parity_bit_relation:
            # Loop through each parity bit relation and add those
            for single_data_bit in parity_bit_relation[single_parity_bit]:
                data_bit_to_parity_bit.setdefault(single_data_bit, []).append(single_parity_bit)
        return data_bit_to_parity_bit

    def show_encoded_matrix(self):
        """
        Display encoded matrix

        :returns: None
        """
        self.print_matrix(self.encoded_matrix)

    @staticmethod
    def print_matrix(matrix, in_file=False):
        """
        Display a given matrix

        :param: matrix: A 2-D matrix
        :param: in_file: if we want to save the encoding information in a file.

        :returns: None
        """
        for row in range(len(matrix)):
            for column in range(len(matrix.T)):
                if not in_file:
                    print(matrix[row][column], end="\t")
                else:
                    print(matrix[row][column], end="\t", file=in_file)
            if not in_file:
                print("")
            else:
                print("", file=in_file)

    @staticmethod
    def matrix_to_data_stream(matrix):
        """
        Convert 2-D matrix to string

        :param: matrix: A 2-D matrix
        :returns: data_stream: string of 2-D matrix
        """
        data_stream = []
        for row in range(len(matrix)):
            for column in range(len(matrix.T)):
                data_stream.append(matrix[row][column])
        return ''.join(str(i) for i in data_stream)

    def data_stream_to_matrix(self, data_stream):
        """
        Convert a sting to 2-D matrix

        The length of data stream should be 48 bit currently this algorithm is only working with 6x8 matrix

        :param: data_stream: 48 bit of string
        :returns: matrix: return 2-D matrix
        """
        matrix = np.full((self.row, self.column), -1)
        data_stream_index = 0
        for row in range(len(matrix)):
            for column in range(len(matrix.T)):
                matrix[row][column] = data_stream[data_stream_index]
                data_stream_index += 1
        return matrix

    def _fix_orientation(self, matrix, option=0):
        """
        Fix the orientation of the decoded matrix. Option parameter will decide which way matrix will be tested now.
        Initially we will check the default matrix(as it was passed). Later we will called this method recursively
        and increase the option value. If option value is 3 and the orientation doesn't match then we will mark this
        origami as not fixed.

        First option is using current matrix
        Second option is reversing the current matrix that will fix the vertically flipped issue
        Third option is mirroring the current matrix that will fix the horizontally flipped issue
        Fourth option is both reverse then mirror the current matrix that will fix
        both vertically flipped and horizontally flipped issue

        :param: matrix: Decoded matrix
                option: On which direction the matrix will be flipped now

        Returns:
            matrix: Orientation fixed matrix.
        """

        if option == 0:
            corrected_matrix = matrix
        elif option == 1:
            # We will just take the reverse/Flip in horizontal direction
            corrected_matrix = np.flipud(matrix)
        elif option == 2:
            # We will take the mirror/flip in vertical direction
            corrected_matrix = np.fliplr(matrix)
        elif option == 3:
            # Flip in both horizontal and vertical direction
            corrected_matrix = np.flipud(np.fliplr(matrix))
        else:
            # The orientation couldn't be determined
            # This is not correctly oriented. Will remove that after testing
            self.logger.info("Couldn't orient the origami")
            return -1, matrix
        orientation_check = True
        for i, bit_index in enumerate(self.matrix_details["orientation_bits"]):
            if corrected_matrix[bit_index[0]][bit_index[1]] != self.matrix_details["orientation_data"][i]:
                orientation_check = False
        if orientation_check:
            # returning option will tell us which way the origami was oriented.
            self.logger.info("Origami was oriented successfully")
            return option, corrected_matrix
        else:
            # Matrix isn't correctly oriented so we will try with other orientation
            return self._fix_orientation(matrix, option + 1)

    def _find_possible_error_location(self, matrix):
        """
        Return all the correct and incorrect parity bits.

        :param: matrix: 2-D matrix
        :returns: correct_indexes: Indices of all correct parity bits
                  incorrect_indexes: Indices of all incorrect parity bit
        """
        correct_indexes = []
        incorrect_indexes = []
        for parity_bit_index in self.parity_bit_relation:
            # Now xoring every element again and checking it's correct or not
            nearby_values = [int(matrix[a[0]][a[1]]) for a in self.parity_bit_relation[parity_bit_index]]
            xored_value = reduce(lambda i, j: int(i) ^ int(j), nearby_values)
            if matrix[parity_bit_index[0]][parity_bit_index[1]] == int(xored_value):
                correct_indexes.append(parity_bit_index)
            else:
                incorrect_indexes.append(parity_bit_index)
        return correct_indexes, incorrect_indexes

    def _is_matrix_correct(self, matrix):
        """
        Check if all the bits of the matrix are correct or not

        Parameter:
            matrix: A 2-D matrix
        Returns:
            Boolean: True if matrix is correct false otherwise
        """
        correct_indexes, incorrect_indexes = self._find_possible_error_location(matrix)
        return len(incorrect_indexes) == 0

    def _decode(self, matrix, threshold_parity, threshold_data,
            maximum_number_of_error, false_positive):
        """
        Attempts to decode a matrix by flipping bits to match parity and orientation rules.
        
        :param matrix: Matrix to decode
        :param threshold_parity: Parity mismatch tolerance threshold
        :param threshold_data: Data mismatch tolerance threshold
        :param maximum_number_of_error: Max number of bit flips to try
        :param false_positive: False positive tolerance
        :return: Decoded matrix if successful, else -1
        """
        matrix_details = {}

        # Initial matrix check without altering any bit
        _, matrix_weight, probable_errors = self._get_matrix_weight(
            matrix, [], threshold_parity, threshold_data, false_positive
        )

        self.logger.debug(f"Initial matrix weight: {matrix_weight}, Probable errors: {probable_errors}")

        if matrix_weight == 0:
            self.logger.info("No parity mismatch found initially.")
            recovered = self.return_matrix(matrix, [])
            if recovered != -1:
                return recovered

        # Try fixing by flipping one probable bit
        for error in probable_errors:
            key = tuple(error)
            changed_matrix, weight, new_probable_errors = self._get_matrix_weight(
                matrix, [error], threshold_parity, threshold_data, false_positive
            )

            matrix_details[key] = {
                "error_value": weight,
                "probable_error": new_probable_errors
            }

            if weight == 0:
                self.logger.info("All parity matched after flipping one bit.")
                recovered = self.return_matrix(changed_matrix, [error])
                if recovered != -1:
                    return recovered

        # Sort probable fixes by lowest error weight
        matrix_details = dict(sorted(matrix_details.items(), key=lambda x: x[1]["error_value"]))

        # Try combinations of multiple bit flips (up to `maximum_number_of_error`)
        for base_error, detail in matrix_details.items():
            checked_combination = [base_error]
            pending_errors = detail["probable_error"]
            retry_queue = {}

            while len(checked_combination) < maximum_number_of_error and pending_errors:
                matrix_weights = {}

                for error_candidate in pending_errors:
                    test_combination = checked_combination + [error_candidate]
                    test_matrix, test_weight, test_probable_errors = self._get_matrix_weight(
                        matrix, test_combination, threshold_parity, threshold_data, false_positive
                    )

                    if test_weight == 0:
                        recovered = self.return_matrix(test_matrix, test_combination)
                        if recovered != -1:
                            return recovered

                    # Store weight and possible next moves
                    if test_weight not in matrix_weights:
                        matrix_weights[test_weight] = {
                            "cell_checked_so_far": [],
                            "probable_error": []
                        }

                    matrix_weights[test_weight]["cell_checked_so_far"].append(tuple(test_combination))
                    matrix_weights[test_weight]["probable_error"].append(test_probable_errors)

                # Keep top-2 lowest matrix weights
                sorted_weights = sorted(matrix_weights.keys())
                min_weights = sorted_weights[:2] if len(sorted_weights) >= 2 else [sorted_weights[0]]

                for mw in min_weights:
                    self.logger.info(f"Current matrix weight: {mw} with path: {matrix_weights[mw]['cell_checked_so_far']}")
                    for i, combo in enumerate(matrix_weights[mw]["cell_checked_so_far"]):
                        retry_queue[combo] = matrix_weights[mw]["probable_error"][i]

                # Prepare for next loop with most promising path
                for combo in sorted(retry_queue.keys(), key=len, reverse=True):
                    checked_combination = list(combo)
                    pending_errors = set(retry_queue[combo]) - set(checked_combination)
                    del retry_queue[combo]
                    if len(checked_combination) < maximum_number_of_error:
                        break

        # No solution found
        return -1

    def return_matrix(self, correct_matrix, error_locations):
        """
        This will check the orientation and checksum of the matrix.
        If the orientation is correct then droplet data and index will be extracted.
        If orientation cannot be correct then -1 will be return

        :param correct_matrix: Matrix with no error
        :param error_locations: Error location that has been fixed
        :return: single_recovered_matrix: Dictionary which contains details of an individual origami
        """
        # will return this dictionary which will have all the origami details
        single_recovered_matrix = {}
        orientation_info, correct_matrix = self._fix_orientation(correct_matrix)

        if not orientation_info == -1 and self.check_checksum(correct_matrix):
            # fix up the error locations based on the orientation
            error_locations = self._mirror_locations(error_locations, orientation_info)
            single_recovered_matrix['orientation_details'] = self.orientation_details[
                str(orientation_info)]
            single_recovered_matrix['orientation'] = orientation_info
            single_recovered_matrix['matrix'] = correct_matrix
            single_recovered_matrix['orientation_fixed'] = True
            single_recovered_matrix['total_probable_error'] = len(error_locations)
            single_recovered_matrix['probable_error_locations'] = error_locations
            single_recovered_matrix['is_recovered'] = True
            single_recovered_matrix['checksum_checked'] = True
            single_recovered_matrix['index'], single_recovered_matrix[
                'binary_data'] = \
                self._extract_text_and_index(correct_matrix)
            self.logger.info("Origami error fixed. Error corrected: " + str(error_locations))
            print("<----------decoded_matrix--------->")
            print(correct_matrix)
            print("<--------------------------------->")
            return single_recovered_matrix
        else:  # If orientation or checksum doesn't match we will return -1
            self.logger.info("Orientation/checksum didn't match")
            return -1

    def _mirror_locations(self, error_locations, orientation_info):
        updated_locations = []
        for error_location in error_locations:
            if orientation_info == 0:
                updated_locations.append(error_location)
            elif orientation_info == 1:
                updated_locations.append((self.row - 1 - error_location[0], error_location[1]))
            elif orientation_info == 2:
                updated_locations.append((error_location[0], self.column - 1 - error_location[1]))
            elif orientation_info == 3:
                updated_locations.append((self.row - 1 - error_location[0], self.column - 1 - error_location[1]))
        return updated_locations

    def _extract_text_and_index(self, matrix):
        """
        Get droplet data and index of the droplet from the origami
        :param matrix: Matrix from where information will be extracted.
        :return: (index, droplet_data)
        """
        if matrix is None:
            return
        # Extracting index first
        index_bin = []
        for bit_index in self.matrix_details['indexing_bits']:
            index_bin.append(matrix[bit_index[0]][bit_index[1]])
        index_decimal = int(''.join(str(i) for i in index_bin), 2)
        # Extracting the text now
        # Extracting text index
        text_bin_data = ""
        for bit_index in self.matrix_details['data_bits']:
            text_bin_data += str(matrix[bit_index[0]][bit_index[1]])

        return index_decimal, text_bin_data

    def _get_matrix_weight(self, matrix, changing_location, threshold_parity, threshold_data, false_positive):
        """
        Calculates the "matrix weight", indicating error severity in the matrix.
        
        :param matrix: The matrix to analyze
        :param changing_location: List of (row, col) positions to flip before analysis
        :param threshold_parity: Threshold for parity error significance
        :param threshold_data: Threshold for data error significance
        :param false_positive: Allowance for ignoring false positives
        :return: (modified_matrix, matrix_weight, probable_error_data_parity)
        """
        matrix_copy = copy.deepcopy(matrix)

        false_positive_data = 0
        false_positive_parity = 0

        # Flip the bits at the specified positions
        for (i, j) in changing_location:
            if matrix_copy[i][j] == 0:
                matrix_copy[i][j] = 1
            else:
                matrix_copy[i][j] = 0
                if (i, j) in self.parity_bit_relation:
                    false_positive_parity += 1
                else:
                    false_positive_data += 1

        # Identify parity bits that are incorrect
        parity_correct, parity_incorrect = self._find_possible_error_location(matrix_copy)
        probable_error_indexes = [pos for p in parity_incorrect for pos in self.parity_bit_relation[p]]

        # Analyze checksum mismatches
        checksum_errors = []
        checksum_related_errors = []
        for checksum_index, related_cells in self.checksum_bit_relation.items():
            xor_value = reduce(lambda x, y: x ^ y, [int(matrix_copy[i][j]) for (i, j) in related_cells])
            expected_value = matrix_copy[checksum_index[0]][checksum_index[1]]

            if xor_value != expected_value:
                checksum_errors.append(checksum_index)
                probable_error_indexes.append(checksum_index)
                checksum_related_errors.extend(related_cells)

        # Weigh data bit errors by frequency and checksum involvement
        probable_data_error = {}
        for (pos, count) in Counter(probable_error_indexes).most_common():
            weight = count + (
                2 if pos in checksum_related_errors and pos in checksum_errors else
                1 if pos in checksum_related_errors or pos in checksum_errors else 0
            )
            probable_data_error.setdefault(weight, []).append(pos)

        # Collect parity bit errors linked from data positions
        all_probable_parity = []
        for pos in probable_data_error.values():
            for data_pos in pos:
                all_probable_parity.extend(self.data_bit_to_parity_bit[data_pos])

        all_probable_parity.extend(parity_incorrect)
        counted_parity_errors = Counter(all_probable_parity).most_common()

        # Determine false positive limits
        fp_data_limit = (false_positive + 1) // 2
        fp_parity_limit = false_positive // 2 if false_positive else 0

        matrix_weight = 0
        probable_parity_error = []

        # Process parity errors
        for (pos, weight) in counted_parity_errors:
            matrix_weight += weight
            if weight >= threshold_parity:
                if matrix_copy[pos[0]][pos[1]] == 0:
                    probable_parity_error.append(pos)
                elif false_positive_parity < fp_parity_limit:
                    probable_parity_error.append(pos)
                    false_positive_parity += 1

        # Process data bit errors
        probable_data_errors = []
        for weight in sorted(probable_data_error.keys(), reverse=True):
            if weight >= threshold_data:
                for pos in probable_data_error[weight]:
                    if matrix_copy[pos[0]][pos[1]] == 0:
                        probable_data_errors.append(pos)
                    elif false_positive_data < fp_data_limit:
                        probable_data_errors.append(pos)
                        false_positive_data += 1
            matrix_weight += weight * len(probable_data_error[weight])

        # Combine final probable error locations
        probable_error_data_parity = probable_data_errors + probable_parity_error
        normalized_weight = matrix_weight / len(parity_correct) if parity_correct else matrix_weight

        return matrix_copy, normalized_weight, probable_error_data_parity


    def decode(self, data_stream, threshold_data, threshold_parity,
               maximum_number_of_error, false_positive):
        """
        Decode the given data stream into word and their respective index

        Parameters:
            data_stream: A string of 48 bit
            Otherwise only recovered word and position of error

        Return:
            decoded_data: A dictionary of index and world which is the most possible solution
            :param threshold_parity:
            :param data_stream:
            :param threshold_data:
            :param false_positive:
            :param maximum_number_of_error:
        """
        # If length of decoded data is not 48 then show error
        if len(data_stream) != self.row * self.column:
            raise ValueError("The data stream length should be", self.row * self.column)
        # Initial check which parity bit index gave error and which gave correct results
        # Converting the data stream to data array first
        data_matrix_for_decoding = self.data_stream_to_matrix(data_stream)
        return self._decode(data_matrix_for_decoding, threshold_data,
                            threshold_parity, maximum_number_of_error, false_positive)

        #   After fixing orientation we need to check the checksum bit.
        #   If we check before orientation fixed then it will not work

        # sorting the matrix

    def check_checksum(self, matrix):
        for check_sum_bit in self.checksum_bit_relation:
            nearby_values = [int(matrix[a[0]][a[1]]) for a in self.checksum_bit_relation[check_sum_bit]]
            xor_value = reduce(lambda i, j: int(i) ^ int(j), nearby_values)
            if xor_value != matrix[check_sum_bit[0]][check_sum_bit[1]]:
                self.logger.info("Checksum did not matched")
                return False
        return True


# This is only for debugging purpose
if __name__ == "__main__":
    bin_stream = "00110110010101010110101011010"
    origami_object = Origami(2)
    encoded_file = origami_object.data_stream_to_matrix(origami_object.encode(bin_stream, 0, 29))

    # encoded_file[1][0] = 0
    # encoded_file[2][2] = 0
    # encoded_file[0][6] = 0
    # encoded_file[7][5] = 0

    # encoded_file = (np.fliplr(encoded_file))
    encoded_file = origami_object.data_stream_to_matrix('11011110001100101110101000110000011000110100001111011001101010001101100001001000')

    decoded_file = origami_object.decode(origami_object.matrix_to_data_stream(encoded_file), 2, 3, 5, 0)

    print(decoded_file)
    if not decoded_file == -1 and decoded_file['binary_data'] == bin_stream:
        print("Decoded successfully")
    else:
        print("wasn't decoded successfully")
