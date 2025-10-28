from functools import reduce
from collections import Counter
import get_parity_n_checksum as pcm
import copy
import numpy as np
import logging
import heapq
import itertools
from log import get_logger


class Origami:
    """
    This class handles individual origami. Both encoding and decoding are handled by this class.
    Each origami is represented by a matrix, so the term matrix and origami are used interchangeably.
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
            '2': 'Origami was flipped in vertical direction',
            '3': 'Origami was flipped in both directions'
        }
        self.logger = get_logger(verbose, __name__)

    # --------------------------- Parity & Checksum Relations -----------------------------

    @staticmethod
    def get_parity_relation(parity_number=40):
        if parity_number == 16:
            return pcm.parity_mapping_16()
        elif parity_number == 24:
            return pcm.parity_mapping_24()
        else:
            return pcm.parity_mapping_40()

    @staticmethod
    def get_checksum_relation(parity_number=40):
        if parity_number == 16:
            return pcm.checksum_mapping_16()
        elif parity_number == 24:
            return pcm.checksum_mapping_24()
        else:
            return pcm.checksum_mapping_40()

    # --------------------------- Matrix Layout Details -----------------------------

    def _matrix_details(self, data_bit_per_origami: int, parity_number: int):
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

    # --------------------------- Encoding Section -----------------------------

    def create_initial_matrix_from_binary_stream(self, binary_stream: str, index: int):
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
            xor_result = reduce(lambda x, y: x ^ y, data_values)
            matrix[parity_row][parity_col] = xor_result
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
        return self._encode(binary_stream, index, data_bit_per_origami, 40)

    @staticmethod
    def get_data_bit_to_parity_bit(parity_bit_relation):
        data_bit_to_parity_bit = {}
        for single_parity_bit in parity_bit_relation:
            for single_data_bit in parity_bit_relation[single_parity_bit]:
                data_bit_to_parity_bit.setdefault(single_data_bit, []).append(single_parity_bit)
        return data_bit_to_parity_bit

    # --------------------------- Matrix Utilities -----------------------------

    @staticmethod
    def print_matrix(matrix, in_file=False):
        for row in range(len(matrix)):
            for column in range(len(matrix.T)):
                if not in_file:
                    print(matrix[row][column], end="\t")
                else:
                    print(matrix[row][column], end="\t", file=in_file)
            print("" if not in_file else "", file=in_file if in_file else None)

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
                matrix[row][column] = data_stream[data_stream_index]
                data_stream_index += 1
        return matrix

    # --------------------------- Orientation Fixing -----------------------------

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
            print("<--------------------->")
            print(corrected_matrix)
            print("<--------------------->")
            return option, corrected_matrix
        else:
            return self._fix_orientation(matrix, option + 1)

    # --------------------------- Error Weight Calculation -----------------------------

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

    def _get_matrix_weight(self, matrix, changing_location, threshold_parity, threshold_data, false_positive):
        matrix_copy = copy.deepcopy(matrix)

        false_positive_data = 0
        false_positive_parity = 0

        for (i, j) in changing_location:
            matrix_copy[i][j] = 1 if matrix_copy[i][j] == 0 else 0
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
                probable_parity_error.append(pos)

        probable_data_errors = []
        for weight in sorted(probable_data_error.keys(), reverse=True):
            if weight >= threshold_data:
                for pos in probable_data_error[weight]:
                    probable_data_errors.append(pos)
            matrix_weight += weight * len(probable_data_error[weight])

        probable_error_data_parity = probable_data_errors + probable_parity_error
        normalized_weight = matrix_weight / len(parity_correct) if parity_correct else matrix_weight

        return matrix_copy, normalized_weight, probable_error_data_parity

    # --------------------------- Orientation & Checksum Check -----------------------------

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
            single_recovered_matrix['index'], single_recovered_matrix['binary_data'] = \
                self._extract_text_and_index(correct_matrix)
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
        index_bin = []
        for bit_index in self.matrix_details['indexing_bits']:
            index_bin.append(matrix[bit_index[0]][bit_index[1]])
        index_decimal = int(''.join(str(i) for i in index_bin), 2)
        text_bin_data = ""
        for bit_index in self.matrix_details['data_bits']:
            text_bin_data += str(matrix[bit_index[0]][bit_index[1]])
        return index_decimal, text_bin_data

    def check_checksum(self, matrix):
        for check_sum_bit in self.checksum_bit_relation:
            nearby_values = [int(matrix[a[0]][a[1]]) for a in self.checksum_bit_relation[check_sum_bit]]
            xor_value = reduce(lambda i, j: int(i) ^ int(j), nearby_values)
            if xor_value != matrix[check_sum_bit[0]][check_sum_bit[1]]:
                return False
        return True

    # --------------------------- A* Search Decoding -----------------------------

    def _decode_a_star(
        self,
        matrix,
        threshold_parity,
        threshold_data,
        maximum_number_of_error,
        false_positive,
        *,
        alpha: float = 1.0,
        beam_width: int = 16
    ):
        tie = itertools.count()
        open_heap = []
        visited_best_g = {}

        start_mat, h0, start_probables = self._get_matrix_weight(
            matrix, [], threshold_parity, threshold_data, false_positive
        )
        if h0 == 0:
            recovered = self.return_matrix(start_mat, [])
            if recovered != -1:
                return recovered

        start_key = tuple(sorted(()))
        g0 = 0
        f0 = g0 + alpha * h0
        heapq.heappush(open_heap, (f0, g0, h0, next(tie), start_key, start_mat, start_probables))
        visited_best_g[start_key] = g0

        while open_heap:
            f, g, h, _, state_key, cur_mat, cur_probables = heapq.heappop(open_heap)
            if g >= maximum_number_of_error:
                continue

            next_candidates = []
            seen = set(state_key)
            for p in cur_probables:
                if p not in seen:
                    next_candidates.append(p)
                    if len(next_candidates) >= beam_width:
                        break

            for flip in next_candidates:
                new_state = tuple(sorted(state_key + (flip,)))
                new_g = g + 1
                if new_state in visited_best_g and visited_best_g[new_state] <= new_g:
                    continue

                test_mat, new_h, new_probables = self._get_matrix_weight(
                    matrix, list(new_state), threshold_parity, threshold_data, false_positive
                )

                if new_h == 0:
                    recovered = self.return_matrix(test_mat, list(new_state))
                    if recovered != -1:
                        return recovered

                new_f = new_g + alpha * new_h
                heapq.heappush(open_heap, (new_f, new_g, new_h, next(tie), new_state, test_mat, new_probables))
                visited_best_g[new_state] = new_g

        return -1

    def decode_a_star(self, data_stream, threshold_data, threshold_parity,
                      maximum_number_of_error, false_positive,
                      *, alpha: float = 1.0, beam_width: int = 16):
        if len(data_stream) != self.row * self.column:
            raise ValueError("The data stream length should be", self.row * self.column)
        matrix = self.data_stream_to_matrix(data_stream)
        return self._decode_a_star(
            matrix,
            threshold_parity,
            threshold_data,
            maximum_number_of_error,
            false_positive,
            alpha=alpha,
            beam_width=beam_width
        )

    # If you want to override decode to use A*
    def decode(self, data_stream, threshold_data, threshold_parity,
               maximum_number_of_error, false_positive):
        return self.decode_a_star(data_stream, threshold_data, threshold_parity,
                                  maximum_number_of_error, false_positive)


# --------------------------- Debug Run -----------------------------

if __name__ == "__main__":
    bin_stream = "00110110010101010110101011010"
    origami_object = Origami(verbose=2)
    encoded_matrix = origami_object.data_stream_to_matrix(
        origami_object.encode(bin_stream, 0, 29)
    )
    # Example error injection
    # encoded_matrix[1][0] = 0
    # encoded_matrix[2][2] = 0

    decoded_file = origami_object.decode(
        origami_object.matrix_to_data_stream(encoded_matrix),
        threshold_data=2,
        threshold_parity=3,
        maximum_number_of_error=5,
        false_positive=0
    )

    print(decoded_file)
    if decoded_file != -1 and decoded_file['binary_data'] == bin_stream:
        print("Decoded successfully")
    else:
        print("Decoding failed")
