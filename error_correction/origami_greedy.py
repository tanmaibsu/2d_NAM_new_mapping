import copy
import numpy as np
import heapq
from functools import reduce
from collections import Counter
import get_parity_n_checksum as pcm
from log import get_logger

class Origami:
    """
    This class handle individual origami. Both the encoding and decoding is handled by this class.
    """

    def __init__(self, verbose=0):
        self.row = 8
        self.column = 10
        self.checksum_bit_per_origami = 4
        self.encoded_matrix = None
        self.recovered_matrix_info = []
        self.list_of_error_combination = []
        self.orientation_details = {
            '0': 'Orientation of the origami is correct',
            '1': 'Origami was flipped in horizontal direction',
            '2': 'Origami was flipped in vertical direction.',
            '3': 'Origami was flipped in both direction. '
        }
        self.logger = get_logger(verbose, __name__)

    @staticmethod
    def get_parity_relation(parity_number=40):
        parity_relation = {}
        if parity_number == 16:
            parity_relation = pcm.parity_mapping_16() 
        elif parity_number == 24:
            parity_relation = pcm.parity_mapping_24() 
        else:
            parity_relation = pcm.parity_mapping_40() 
        return parity_relation
            
    @staticmethod
    def get_checksum_relation(parity_number=40):
        checksum_relation = {}
        if parity_number == 16:
            checksum_relation = pcm.checksum_mapping_16()
        elif parity_number == 24:
            checksum_relation = pcm.checksum_mapping_24()
        else:
            checksum_relation = pcm.checksum_mapping_40()
        return checksum_relation

    def _matrix_details(self, data_bit_per_origami: int, parity_number: int) -> object:
        parity_bit_relation = self.get_parity_relation(parity_number)
        checksum_bit_relation = self.get_checksum_relation(parity_number)

        data_index_orientation = set([i for v in checksum_bit_relation.values() for i in v])
        orientation_bits = set([(1, 0), (1, 9), (6, 0), (6, 9)])
        index_bits = set([(2, 0), (3, 0), (4, 0)])
        data_index = data_index_orientation - orientation_bits
        data_index = sorted(list(data_index))
        
        data_bits = data_index[:data_bit_per_origami]
        index_bits = data_index[data_bit_per_origami:]

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
        binary_list = list(binary_stream)
        data_matrix = np.full((self.row, self.column), -1) 

        for i, (row, col) in enumerate(self.matrix_details["data_bits"]):
            data_matrix[row][col] = binary_list[i]

        for i, (row, col) in enumerate(self.matrix_details["orientation_bits"]):
            data_matrix[row][col] = self.matrix_details['orientation_data'][i]

        max_index = 2 ** len(self.matrix_details["indexing_bits"])
        if index >= max_index:
            raise ValueError(f"Index {index} exceeds maximum supported index of {max_index - 1}")

        index_bits_required = len(self.matrix_details["indexing_bits"])
        index_bin = format(index, f'0{index_bits_required}b')

        for i, (row, col) in enumerate(self.matrix_details["indexing_bits"]):
            data_matrix[row][col] = index_bin[i]

        return data_matrix

    @staticmethod
    def _xor_matrix(matrix, relation):
        for (parity_row, parity_col), data_bit_positions in relation.items():
            data_values = [int(matrix[row][col]) for row, col in data_bit_positions]
            matrix[parity_row][parity_col] = reduce(lambda x, y: x ^ y, data_values)
        return matrix

    def _encode(self, binary_stream, index, data_bit_per_origami, parity_number):
        self.number_of_bit_per_origami = data_bit_per_origami
        self.matrix_details, self.parity_bit_relation, self.checksum_bit_relation = \
            self._matrix_details(data_bit_per_origami, parity_number)
        self.data_bit_to_parity_bit = Origami.get_data_bit_to_parity_bit(self.parity_bit_relation)
        
        encoded_matrix = self.create_initial_matrix_from_binary_stream(binary_stream, index)
        encoded_matrix = Origami._xor_matrix(encoded_matrix, self.checksum_bit_relation)
        encoded_matrix = Origami._xor_matrix(encoded_matrix, self.parity_bit_relation)
        return Origami.matrix_to_data_stream(encoded_matrix)

    def encode(self, binary_stream, index, data_bit_per_origami):
        return self._encode(binary_stream, index, data_bit_per_origami, parity_number=24)

    @staticmethod
    def get_data_bit_to_parity_bit(parity_bit_relation):
        data_bit_to_parity_bit = {}
        for single_parity_bit in parity_bit_relation:
            for single_data_bit in parity_bit_relation[single_parity_bit]:
                data_bit_to_parity_bit.setdefault(single_data_bit, []).append(single_parity_bit)
        return data_bit_to_parity_bit

    @staticmethod
    def matrix_to_data_stream(matrix):
        data_stream = []
        for row in range(len(matrix)):
            for column in range(len(matrix.T)):
                data_stream.append(matrix[row][column])
        return ''.join(str(i) for i in data_stream)

    def data_stream_to_matrix(self, data_stream):
        matrix = np.full((self.row, self.column), -1)
        data_stream_index = 0
        for row in range(len(matrix)):
            for column in range(len(matrix.T)):
                matrix[row][column] = int(data_stream[data_stream_index])
                data_stream_index += 1
        return matrix

    def _fix_orientation(self, matrix, option=0):
        if option == 0:
            corrected_matrix = matrix
        elif option == 1:
            corrected_matrix = np.flipud(matrix)
        elif option == 2:
            corrected_matrix = np.fliplr(matrix)
        elif option == 3:
            corrected_matrix = np.flipud(np.fliplr(matrix))
        else:
            return -1, matrix
            
        orientation_check = True
        for i, bit_index in enumerate(self.matrix_details["orientation_bits"]):
            if corrected_matrix[bit_index[0]][bit_index[1]] != self.matrix_details["orientation_data"][i]:
                orientation_check = False
        if orientation_check:
            return option, corrected_matrix
        else:
            return self._fix_orientation(matrix, option + 1)

    def _find_possible_error_location(self, matrix):
        correct_indexes = []
        incorrect_indexes = []
        for parity_bit_index in self.parity_bit_relation:
            nearby_values = [int(matrix[a[0]][a[1]]) for a in self.parity_bit_relation[parity_bit_index]]
            xored_value = reduce(lambda i, j: int(i) ^ int(j), nearby_values)
            if matrix[parity_bit_index[0]][parity_bit_index[1]] == int(xored_value):
                correct_indexes.append(parity_bit_index)
            else:
                incorrect_indexes.append(parity_bit_index)
        return correct_indexes, incorrect_indexes

    def _is_matrix_correct(self, matrix):
        correct_indexes, incorrect_indexes = self._find_possible_error_location(matrix)
        return len(incorrect_indexes) == 0

    # =========================================================================
    # YOUR EXACT DOMAIN-SPECIFIC MATRIX SCORING FUNCTION
    # =========================================================================

    def _get_matrix_weight(self, matrix, changing_location, threshold_parity, threshold_data, false_positive):
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

        parity_correct, parity_incorrect = self._find_possible_error_location(matrix_copy)
        probable_error_indexes = [pos for p in parity_incorrect for pos in self.parity_bit_relation[p]]
        
        checksum_errors = []
        checksum_related_errors = []
        for checksum_index, related_cells in self.checksum_bit_relation.items():
            xor_value = reduce(lambda x, y: x ^ y, [int(matrix_copy[i][j]) for (i, j) in related_cells])
            expected_value = matrix_copy[checksum_index[0]][checksum_index[1]]

            if xor_value != expected_value:
                checksum_errors.append(checksum_index)
                probable_error_indexes.append(checksum_index)
                checksum_related_errors.extend(related_cells)

        probable_data_error = {}
        for (pos, count) in Counter(probable_error_indexes).most_common():
            weight = count + (
                2 if pos in checksum_related_errors and pos in checksum_errors else
                1 if pos in checksum_related_errors or pos in checksum_errors else 0
            )
            probable_data_error.setdefault(weight, []).append(pos)
        
        all_probable_parity = []
        for pos in probable_data_error.values():
            for data_pos in pos:
                if data_pos in self.data_bit_to_parity_bit:
                    all_probable_parity.extend(self.data_bit_to_parity_bit[data_pos])

        all_probable_parity.extend(parity_incorrect)
        counted_parity_errors = Counter(all_probable_parity).most_common()

        fp_data_limit = (false_positive + 1) // 2
        fp_parity_limit = false_positive // 2 if false_positive else 0

        matrix_weight = 0
        probable_parity_error = []

        for (pos, weight) in counted_parity_errors:
            matrix_weight += weight
            if weight >= threshold_parity:
                if matrix_copy[pos[0]][pos[1]] == 0:
                    probable_parity_error.append(pos)
                elif false_positive_parity < fp_parity_limit:
                    probable_parity_error.append(pos)
                    false_positive_parity += 1

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

        probable_error_data_parity = probable_data_errors + probable_parity_error
        normalized_weight = matrix_weight / len(parity_correct) if parity_correct else matrix_weight

        return matrix_copy, normalized_weight, probable_error_data_parity

    # =========================================================================
    # NEW BEST-FIRST SEARCH WRAPPER USING YOUR HEURISTIC
    # =========================================================================

    def _decode(self, matrix, threshold_parity, threshold_data, maximum_number_of_error, false_positive):
        """
        Replaces the old greedy retry_queue with a Priority Queue (A* style) search.
        This uses your exact `_get_matrix_weight` function but prevents it from getting
        stuck in local minima when approaching 8 errors.
        """
        # 1. Initial matrix check
        _, initial_weight, probable_errors = self._get_matrix_weight(
            matrix, [], threshold_parity, threshold_data, false_positive
        )

        if initial_weight == 0:
            recovered = self.return_matrix(matrix, [])
            if recovered != -1:
                return recovered

        # Priority Queue stores: (weight, depth, tie_breaker, tuple_of_flips, pending_errors)
        pq = []
        counter = 0
        heapq.heappush(pq, (initial_weight, 0, counter, (), probable_errors))

        # Track visited combinations to prevent infinite loops (e.g. flipping A then B vs B then A)
        visited = {()}

        while pq:
            current_weight, num_flips, _, current_combo, pending_errors = heapq.heappop(pq)

            if num_flips >= maximum_number_of_error:
                continue

            # Branch out using the probable errors identified by your scoring function
            for error_candidate in pending_errors:
                err_tuple = tuple(error_candidate)
                
                # Skip if we already flipped this bit in the current path
                if err_tuple in current_combo:
                    continue

                # Create the new combination and sort it so (A, B) is treated the same as (B, A)
                test_combination = tuple(sorted(list(current_combo) + [err_tuple]))
                
                if test_combination in visited:
                    continue
                visited.add(test_combination)

                # Test the new combination using your exact matrix weight function
                test_matrix, test_weight, test_probable_errors = self._get_matrix_weight(
                    matrix, test_combination, threshold_parity, threshold_data, false_positive
                )

                # If the weight is 0, we found the completely valid solution!
                if test_weight == 0:
                    recovered = self.return_matrix(test_matrix, list(test_combination))
                    if recovered != -1:
                        return recovered

                # Otherwise, push it back to the priority queue to explore further
                if len(test_combination) < maximum_number_of_error:
                    counter += 1
                    heapq.heappush(pq, (test_weight, len(test_combination), counter, test_combination, test_probable_errors))

        # No solution found within error bounds
        return -1

    # =========================================================================

    def return_matrix(self, correct_matrix, error_locations):
        single_recovered_matrix = {}
        orientation_info, correct_matrix = self._fix_orientation(correct_matrix)

        if not orientation_info == -1 and self.check_checksum(correct_matrix):
            error_locations = self._mirror_locations(error_locations, orientation_info)
            single_recovered_matrix['orientation_details'] = self.orientation_details[str(orientation_info)]
            single_recovered_matrix['orientation'] = orientation_info
            single_recovered_matrix['matrix'] = correct_matrix
            single_recovered_matrix['orientation_fixed'] = True
            single_recovered_matrix['total_probable_error'] = len(error_locations)
            single_recovered_matrix['probable_error_locations'] = error_locations
            single_recovered_matrix['is_recovered'] = True
            single_recovered_matrix['checksum_checked'] = True
            single_recovered_matrix['index'], single_recovered_matrix['binary_data'] = self._extract_text_and_index(correct_matrix)
            return single_recovered_matrix
        else:  
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
        if matrix is None:
            return
        index_bin = []
        for bit_index in self.matrix_details['indexing_bits']:
            index_bin.append(matrix[bit_index[0]][bit_index[1]])
        index_decimal = int(''.join(str(i) for i in index_bin), 2)
        
        text_bin_data = ""
        for bit_index in self.matrix_details['data_bits']:
            text_bin_data += str(matrix[bit_index[0]][bit_index[1]])

        return index_decimal, text_bin_data

    def decode(self, data_stream, threshold_data, threshold_parity, maximum_number_of_error, false_positive):
        if len(data_stream) != self.row * self.column:
            raise ValueError("The data stream length should be", self.row * self.column)
        data_matrix_for_decoding = self.data_stream_to_matrix(data_stream)
        return self._decode(data_matrix_for_decoding, threshold_parity, threshold_data, maximum_number_of_error, false_positive)

    def check_checksum(self, matrix):
        for check_sum_bit in self.checksum_bit_relation:
            nearby_values = [int(matrix[a[0]][a[1]]) for a in self.checksum_bit_relation[check_sum_bit]]
            xor_value = reduce(lambda i, j: int(i) ^ int(j), nearby_values)
            if xor_value != int(matrix[check_sum_bit[0]][check_sum_bit[1]]):
                return False
        return True


if __name__ == "__main__":
    origami_object = Origami(verbose=1)
    
    # Init matrix definitions mapping to tests
    origami_object.matrix_details, origami_object.parity_bit_relation, origami_object.checksum_bit_relation = \
        origami_object._matrix_details(46, 24)
    origami_object.data_bit_to_parity_bit = origami_object.get_data_bit_to_parity_bit(origami_object.parity_bit_relation)

    nodes = [
        "01000100011111110001100000100111101010000111000010001001000011110001000001101000",
        "01011100111100001111000100001000111101100110101100101100011011110101100000011001",
        "11110111011111101001001011100011000001100000100001000010001110000011000011100110",
        "00000100101000010101000010000010010100000100100000000000000010000111100000000011"
    ]

    for i, node_str in enumerate(nodes):
        result = origami_object.decode(node_str, 2, 2, 8, 0)
        if result != -1:
            print(f"Node index {result['index']} recovered! Flips used: {result['total_probable_error']}")
        else:
            print(f"Failed to decode Node {i}")