from functools import reduce
from collections import Counter
import get_parity_n_checksum as pcm
import copy
import numpy as np
import logging
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
        self.data_bit_to_parity_bit = self.get_data_bit_to_parity_bit(self.parity_bit_relation)

        encoded_matrix = self.create_initial_matrix_from_binary_stream(binary_stream, index)
        encoded_matrix = self._xor_matrix(encoded_matrix, self.checksum_bit_relation)
        encoded_matrix = self._xor_matrix(encoded_matrix, self.parity_bit_relation)
        return self.matrix_to_data_stream(encoded_matrix)

    def encode(self, binary_stream, index, data_bit_per_origami):
        return self._encode(binary_stream, index, data_bit_per_origami, 40)

    @staticmethod
    def get_data_bit_to_parity_bit(parity_bit_relation):
        data_bit_to_parity_bit = {}
        for single_parity_bit in parity_bit_relation:
            for single_data_bit in parity_bit_relation[single_parity_bit]:
                data_bit_to_parity_bit.setdefault(single_data_bit, []).append(single_parity_bit)
        return data_bit_to_parity_bit

    # --------------------------- Utilities -----------------------------

    @staticmethod
    def matrix_to_data_stream(matrix):
        data_stream = []
        for row in range(len(matrix)):
            for column in range(len(matrix.T)):
                data_stream.append(matrix[row][column])
        return ''.join(str(i) for i in data_stream)

    def data_stream_to_matrix(self, data_stream):
        matrix = np.full((self.row, self.column), -1)
        k = 0
        for r in range(self.row):
            for c in range(self.column):
                matrix[r][c] = data_stream[k]
                k += 1
        return matrix

    # --------------------------- Orientation -----------------------------

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

    # --------------------------- Error Detection -----------------------------

    def _find_possible_error_location(self, matrix):
        correct_indexes, incorrect_indexes = [], []
        for parity_bit_index in self.parity_bit_relation:
            nearby_values = [int(matrix[a][b]) for a, b in self.parity_bit_relation[parity_bit_index]]
            xored_value = reduce(lambda i, j: int(i) ^ int(j), nearby_values)
            if matrix[parity_bit_index[0]][parity_bit_index[1]] == int(xored_value):
                correct_indexes.append(parity_bit_index)
            else:
                incorrect_indexes.append(parity_bit_index)
        return correct_indexes, incorrect_indexes

    def check_checksum(self, matrix):
        for check_sum_bit in self.checksum_bit_relation:
            nearby_values = [int(matrix[a[0]][a[1]]) for a in self.checksum_bit_relation[check_sum_bit]]
            xor_value = reduce(lambda i, j: int(i) ^ int(j), nearby_values)
            if xor_value != matrix[check_sum_bit[0]][check_sum_bit[1]]:
                return False
        return True

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

    # --------------------------- Full Matrix Recovery -----------------------------

    def return_matrix(self, correct_matrix, error_locations):
        single_recovered_matrix = {
            'orientation_details': None,
            'orientation': None,
            'matrix': correct_matrix,
            'orientation_fixed': False,
            'total_probable_error': len(error_locations),
            'probable_error_locations': error_locations,
            'is_recovered': False,
            'checksum_checked': False,
            'index': -1,
            'binary_data': ""
        }

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
            return single_recovered_matrix  # Return even if failed, to maintain consistent structure

    def _extract_text_and_index(self, matrix):
        index_bin = []
        for bit_index in self.matrix_details['indexing_bits']:
            index_bin.append(matrix[bit_index[0]][bit_index[1]])
        index_decimal = int(''.join(str(i) for i in index_bin), 2)
        text_bin_data = ""
        for bit_index in self.matrix_details['data_bits']:
            text_bin_data += str(matrix[bit_index[0]][bit_index[1]])
        return index_decimal, text_bin_data

    # --------------------------- Beam Search Helpers -----------------------------

    def _hash_matrix(self, matrix): return tuple(int(x) for x in matrix.flatten())

    def _flip_in_place(self, mat, pos):
        i, j = pos
        mat[i][j] = 1 if int(mat[i][j]) == 0 else 0

    def _violations_and_candidates(self, matrix):
        parity_correct, parity_incorrect = self._find_possible_error_location(matrix)
        checksum_incorrect = []
        checksum_related_union = set()
        for c_idx, related in self.checksum_bit_relation.items():
            xor_val = 0
            for (r, c) in related:
                xor_val ^= int(matrix[r][c])
            expected = int(matrix[c_idx[0]][c_idx[1]])
            if xor_val != expected:
                checksum_incorrect.append(c_idx)
                checksum_related_union.update(related)
        cand = set()
        for p in parity_incorrect:
            cand.update(self.parity_bit_relation[p])
        cand.update(checksum_related_union)
        touched = {}
        violated_set = set(parity_incorrect)
        for bit in cand:
            refs = self.data_bit_to_parity_bit.get(bit, [])
            touched[bit] = sum(1 for r in refs if r in violated_set)
        checksum_bonus = {bit: 1 if bit in checksum_related_union else 0 for bit in cand}
        scores = {bit: touched[bit] * 2 + checksum_bonus[bit] for bit in cand}
        ranked = sorted(cand, key=lambda b: scores[b], reverse=True)
        return len(parity_incorrect) + len(checksum_incorrect), parity_incorrect, checksum_incorrect, ranked, scores

    # --------------------------- Stochastic Beam Search -----------------------------

    def _decode_beam(
    self,
    matrix,
    *,
    beam_width=12,
    max_flips=8,
    restarts=2,
    per_parent_expansions=4,
    stagnation_patience=2,
    rng_seed=0
    ):
        if rng_seed:
            np.random.seed(rng_seed)

        if not hasattr(self, "data_bit_to_parity_bit"):
            self.data_bit_to_parity_bit = self.get_data_bit_to_parity_bit(self.parity_bit_relation)

        eval_cache, seen = {}, set()

        def evaluate(mat):
            h = self._hash_matrix(mat)
            if h not in eval_cache:
                eval_cache[h] = self._violations_and_candidates(mat)
            return eval_cache[h]

        best_global = (float("inf"), None, None)

        for _ in range(restarts + 1):
            # population stores (v, matrix, flips)
            v0, *_ = evaluate(matrix)
            population = [(v0, matrix.copy(), tuple())]
            stagnation, best_run = 0, float("inf")

            for _ in range(max_flips):
                scored = []
                for v, mat, flips in population:
                    v, p_bad, c_bad, ranked, scores = evaluate(mat)
                    if v == 0:
                        rec = self.return_matrix(mat, list(flips))
                        if rec["is_recovered"]:
                            return rec
                    scored.append((v, mat, flips, ranked))

                scored.sort(key=lambda x: x[0])
                if scored[0][0] < best_run:
                    best_run = scored[0][0]
                    stagnation = 0
                    if best_run < best_global[0]:
                        best_global = (best_run, scored[0][1].copy(), scored[0][2])
                else:
                    stagnation += 1

                children = []
                for v, mat, flips, ranked in scored[:beam_width]:
                    if not ranked:
                        continue
                    k = min(per_parent_expansions, len(ranked))
                    picks = ranked[:k]
                    if stagnation >= stagnation_patience and len(ranked) > k:
                        picks[-1] = ranked[np.random.randint(k, len(ranked))]

                    for bit in picks:
                        new_mat = mat.copy()
                        self._flip_in_place(new_mat, bit)
                        new_flips = tuple(sorted(flips + (bit,)))
                        h = self._hash_matrix(new_mat)
                        if h in seen:
                            continue
                        seen.add(h)
                        v2, *_ = evaluate(new_mat)
                        children.append((v2, new_mat, new_flips))

                if not children:
                    break

                # Keep best children
                children.sort(key=lambda x: x[0])
                population = children[:beam_width]

                # Early success check
                if population and population[0][0] == 0:
                    rec = self.return_matrix(population[0][1], list(population[0][2]))
                    if rec["is_recovered"]:
                        return rec

        # No exact recovery; return best found
        if best_global[1] is not None:
            fallback = self.return_matrix(best_global[1], list(best_global[2] or ()))
            if fallback["is_recovered"]:
                return fallback
        return {"is_recovered": False, "binary_data": "", "index": -1}

    # --------------------------- Public Decode Interface -----------------------------

    def decode(self, data_stream, threshold_data, threshold_parity,
               maximum_number_of_error, false_positive):
        """
        Uses stochastic beam search for decoding (replaces A*).
        """
        if len(data_stream) != self.row * self.column:
            raise ValueError("Invalid data stream length.")

        if not hasattr(self, "matrix_details"):
            self.matrix_details, self.parity_bit_relation, self.checksum_bit_relation = \
                self._matrix_details(getattr(self, "number_of_bit_per_origami", 29), 40)
            self.data_bit_to_parity_bit = self.get_data_bit_to_parity_bit(self.parity_bit_relation)

        mat = self.data_stream_to_matrix(data_stream)
        return self._decode_beam(
            mat,
            beam_width=12,
            max_flips=maximum_number_of_error or 6,
            restarts=2,
            per_parent_expansions=4,
            stagnation_patience=2,
            rng_seed=0
        )


# --------------------------- Debug Run -----------------------------

if __name__ == "__main__":
    bin_stream = "00110110010101010110101011010"
    origami_object = Origami(verbose=2)
    encoded_matrix = origami_object.data_stream_to_matrix(
        origami_object.encode(bin_stream, 0, 29)
    )
    decoded_file = origami_object.decode(
        origami_object.matrix_to_data_stream(encoded_matrix),
        threshold_data=2,
        threshold_parity=3,
        maximum_number_of_error=6,
        false_positive=0
    )
    print(decoded_file)
    if decoded_file['is_recovered'] and decoded_file['binary_data'] == bin_stream:
        print("Decoded successfully")
    else:
        print("Decoding failed")
