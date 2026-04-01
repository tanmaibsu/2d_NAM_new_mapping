# origami_greedy.py
from functools import reduce
from collections import Counter, defaultdict
import copy
import numpy as np
from log import get_logger
import get_parity_n_checksum as pcm


class Origami:
    """
    Handles encoding/decoding for a single 8x10 origami matrix.

    IMPORTANT (alignment with ProcessFile):
      - ProcessFile calls: super().decode(origami_str, threshold_data, threshold_parity, max_errors, false_positive)
      - Therefore Origami.decode signature MUST be:
            decode(data_stream, threshold_data, threshold_parity, maximum_number_of_error, false_positive)
      - And internal _decode must follow the same order.

    This implementation:
      1) Keeps your original heuristic decoder (as _decode_legacy)
      2) Adds a syndrome-driven iterative decoder (good for up to ~8 flips)
      3) Enforces STRICT acceptance before returning any recovered matrix (no more heuristic "weight==0" acceptance)
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
            '1': 'Origami wasX flipped in horizontal direction',
            '2': 'Origami was flipped in vertical direction.',
            '3': 'Origami was flipped in both direction. '
        }

        self.logger = get_logger(verbose, __name__)

        # Will be initialized by _matrix_details
        self.matrix_details = None
        self.parity_bit_relation = None
        self.checksum_bit_relation = None
        self.data_bit_to_parity_bit = None
        self.number_of_bit_per_origami = None

    # ------------------------------------------------------------------
    # Relations (your mapping providers)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Layout: data/index/orientation/ checksum / parity positions
    # ------------------------------------------------------------------
    def _matrix_details(self, data_bit_per_origami: int, parity_number: int):
        parity_bit_relation = self.get_parity_relation(parity_number)
        checksum_bit_relation = self.get_checksum_relation(parity_number)

        # cells used by checksum mapping (keys + all values)
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
            orientation_data=[1, 1, 1, 0],
        )
        return matrix_details, parity_bit_relation, checksum_bit_relation

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def create_initial_matrix_from_binary_stream(self, binary_stream: str, index: int):
        binary_list = list(binary_stream)
        data_matrix = np.full((self.row, self.column), -1)

        # Insert data bits
        for i, (r, c) in enumerate(self.matrix_details["data_bits"]):
            data_matrix[r][c] = int(binary_list[i])

        # Insert orientation bits
        for i, (r, c) in enumerate(self.matrix_details["orientation_bits"]):
            data_matrix[r][c] = int(self.matrix_details["orientation_data"][i])

        # Insert index bits
        max_index = 2 ** len(self.matrix_details["indexing_bits"])
        if index >= max_index:
            raise ValueError(f"Index {index} exceeds maximum supported index of {max_index - 1}")

        bits_required = len(self.matrix_details["indexing_bits"])
        index_bin = format(index, f'0{bits_required}b')

        for i, (r, c) in enumerate(self.matrix_details["indexing_bits"]):
            data_matrix[r][c] = int(index_bin[i])

        return data_matrix

    @staticmethod
    def _xor_matrix(matrix, relation):
        for (pr, pc), deps in relation.items():
            vals = [int(matrix[r][c]) for (r, c) in deps]
            xor_result = reduce(lambda x, y: x ^ y, vals)
            matrix[pr][pc] = int(xor_result)
        return matrix

    def _encode(self, binary_stream, index, data_bit_per_origami, parity_number):
        self.number_of_bit_per_origami = data_bit_per_origami
        self.matrix_details, self.parity_bit_relation, self.checksum_bit_relation = \
            self._matrix_details(data_bit_per_origami, parity_number)

        self.data_bit_to_parity_bit = Origami.get_data_bit_to_parity_bit(self.parity_bit_relation)

        m = self.create_initial_matrix_from_binary_stream(binary_stream, index)

        # checksum first
        m = Origami._xor_matrix(m, self.checksum_bit_relation)
        # then parity
        m = Origami._xor_matrix(m, self.parity_bit_relation)

        return Origami.matrix_to_data_stream(m)

    def encode(self, binary_stream, index, data_bit_per_origami):
        # kept for compatibility/testing
        return self._encode(binary_stream, index, data_bit_per_origami, parity_number=40)

    @staticmethod
    def get_data_bit_to_parity_bit(parity_bit_relation):
        data_bit_to_parity_bit = {}
        for pbit, deps in parity_bit_relation.items():
            for d in deps:
                data_bit_to_parity_bit.setdefault(d, []).append(pbit)
        return data_bit_to_parity_bit

    # ------------------------------------------------------------------
    # Matrix conversion + printing
    # ------------------------------------------------------------------
    @staticmethod
    def print_matrix(matrix, in_file=False):
        for r in range(len(matrix)):
            for c in range(len(matrix.T)):
                if not in_file:
                    print(matrix[r][c], end="\t")
                else:
                    print(matrix[r][c], end="\t", file=in_file)
            if not in_file:
                print("")
            else:
                print("", file=in_file)

    @staticmethod
    def matrix_to_data_stream(matrix):
        data_stream = []
        for r in range(len(matrix)):
            for c in range(len(matrix.T)):
                data_stream.append(int(matrix[r][c]))
        return ''.join(str(i) for i in data_stream)

    def data_stream_to_matrix(self, data_stream):
        matrix = np.full((self.row, self.column), -1, dtype=int)
        k = 0
        for r in range(self.row):
            for c in range(self.column):
                matrix[r][c] = int(data_stream[k])
                k += 1
        return matrix

    # ------------------------------------------------------------------
    # Orientation
    # ------------------------------------------------------------------
    def _fix_orientation(self, matrix, option=0):
        if option == 0:
            corrected = matrix
        elif option == 1:
            corrected = np.flipud(matrix)
        elif option == 2:
            corrected = np.fliplr(matrix)
        elif option == 3:
            corrected = np.flipud(np.fliplr(matrix))
        else:
            self.logger.info("Couldn't orient the origami")
            return -1, matrix

        ok = True
        for i, (r, c) in enumerate(self.matrix_details["orientation_bits"]):
            if int(corrected[r][c]) != int(self.matrix_details["orientation_data"][i]):
                ok = False
                break

        if ok:
            return option, corrected
        return self._fix_orientation(matrix, option + 1)

    def _mirror_locations(self, error_locations, orientation_info):
        updated = []
        for (r, c) in error_locations:
            if orientation_info == 0:
                updated.append((r, c))
            elif orientation_info == 1:
                updated.append((self.row - 1 - r, c))
            elif orientation_info == 2:
                updated.append((r, self.column - 1 - c))
            elif orientation_info == 3:
                updated.append((self.row - 1 - r, self.column - 1 - c))
        return updated

    # ------------------------------------------------------------------
    # Parity + checksum checking (STRICT)
    # ------------------------------------------------------------------
    def _find_possible_error_location(self, matrix):
        correct = []
        incorrect = []
        for pbit, deps in self.parity_bit_relation.items():
            nearby = [int(matrix[r][c]) for (r, c) in deps]
            x = reduce(lambda i, j: int(i) ^ int(j), nearby)
            if int(matrix[pbit[0]][pbit[1]]) == int(x):
                correct.append(pbit)
            else: 
                incorrect.append(pbit)
        return correct, incorrect

    def check_checksum(self, matrix):
        for cbit, deps in self.checksum_bit_relation.items():
            nearby = [int(matrix[r][c]) for (r, c) in deps]
            x = reduce(lambda i, j: int(i) ^ int(j), nearby)
            if int(x) != int(matrix[cbit[0]][cbit[1]]):
                return False
        return True

    def _strict_matrix_ok(self, matrix):
        """
        Accept only if:
          - orientation is fixable
          - ALL parity checks pass
          - ALL checksum checks pass
        """
        ori, oriented = self._fix_orientation(matrix)
        if ori == -1:
            return False, -1, matrix

        _, bad_parity = self._find_possible_error_location(oriented)
        if bad_parity:
            return False, ori, oriented

        if not self.check_checksum(oriented):
            return False, ori, oriented

        return True, ori, oriented

    # ------------------------------------------------------------------
    # Extract index + data
    # ------------------------------------------------------------------
    def _extract_text_and_index(self, matrix):
        if matrix is None:
            return None, None

        index_bin = [str(int(matrix[r][c])) for (r, c) in self.matrix_details['indexing_bits']]
        index_decimal = int(''.join(index_bin), 2) if index_bin else 0

        text_bin = ""
        for (r, c) in self.matrix_details['data_bits']:
            text_bin += str(int(matrix[r][c]))

        return index_decimal, text_bin

    # ------------------------------------------------------------------
    # Return matrix
    # ------------------------------------------------------------------
    def return_matrix(self, correct_matrix, error_locations):
        single = {}

        orientation_info, corrected = self._fix_orientation(correct_matrix)
        if orientation_info == -1:
            self.logger.info("Orientation didn't match")
            return -1

        if not self.check_checksum(corrected):
            self.logger.info("Checksum didn't match")
            return -1

        # Fix error locations based on final orientation
        error_locations = self._mirror_locations(error_locations, orientation_info)

        single['orientation_details'] = self.orientation_details[str(orientation_info)]
        single['orientation'] = orientation_info
        single['matrix'] = corrected
        single['orientation_fixed'] = True
        single['total_probable_error'] = len(error_locations)
        single['probable_error_locations'] = error_locations
        single['is_recovered'] = True
        single['checksum_checked'] = True
        single['index'], single['binary_data'] = self._extract_text_and_index(corrected)

        return single

    # ------------------------------------------------------------------
    # scoring function
    # ------------------------------------------------------------------
    def _get_matrix_weight(self, matrix, changing_location, threshold_parity, threshold_data, false_positive):
        matrix_copy = copy.deepcopy(matrix)

        false_positive_data = 0
        false_positive_parity = 0

        # Flip bits at specified positions
        for (i, j) in changing_location:
            if int(matrix_copy[i][j]) == 0:
                matrix_copy[i][j] = 1
            else:
                matrix_copy[i][j] = 0
                if (i, j) in self.parity_bit_relation:
                    false_positive_parity += 1
                else:
                    false_positive_data += 1

        parity_correct, parity_incorrect = self._find_possible_error_location(matrix_copy)
        # retrun all the possible error cells for tne possible parity cells
        probable_error_indexes = [pos for p in parity_incorrect for pos in self.parity_bit_relation[p]]

        # checksum mismatches
        checksum_errors = []
        checksum_related_errors = []
        for checksum_index, related_cells in self.checksum_bit_relation.items():
            xor_value = reduce(lambda x, y: x ^ y, [int(matrix_copy[r][c]) for (r, c) in related_cells])
            expected_value = int(matrix_copy[checksum_index[0]][checksum_index[1]])
            if xor_value != expected_value:
                checksum_errors.append(checksum_index)
                probable_error_indexes.append(checksum_index)
                checksum_related_errors.extend(related_cells)

        # probable_error_indexes has parity with thei coverage and checksum errors
        probable_data_error = {}
        # weight 2: if the selected position is both in parity and checksum error positions
        # 1: if the selected position is in any one of them
        for (pos, count) in Counter(probable_error_indexes).most_common():
            weight = count + (
                2 if (pos in checksum_related_errors and pos in checksum_errors) else
                1 if (pos in checksum_related_errors or pos in checksum_errors) else 0
            )
            # grouping possible errors position by their weight
            probable_data_error.setdefault(weight, []).append(pos)

        # parity bits linked from data positions
        all_probable_parity = []
        for lst in probable_data_error.values():
            for data_pos in lst:
                all_probable_parity.extend(self.data_bit_to_parity_bit.get(data_pos, []))

        all_probable_parity.extend(parity_incorrect)
        counted_parity_errors = Counter(all_probable_parity).most_common()

        fp_data_limit = (false_positive + 1) // 2
        fp_parity_limit = false_positive // 2 if false_positive else 0

        matrix_weight = 0
        probable_parity_error = []

        for (pos, w) in counted_parity_errors:
            matrix_weight += w
            if w >= threshold_parity:
                if int(matrix_copy[pos[0]][pos[1]]) == 0:
                    probable_parity_error.append(pos)
                elif false_positive_parity < fp_parity_limit:
                    probable_parity_error.append(pos)
                    false_positive_parity += 1

        probable_data_errors = []
        for w in sorted(probable_data_error.keys(), reverse=True):
            if w >= threshold_data:
                for pos in probable_data_error[w]:
                    if int(matrix_copy[pos[0]][pos[1]]) == 0:
                        probable_data_errors.append(pos)
                    elif false_positive_data < fp_data_limit:
                        probable_data_errors.append(pos)
                        false_positive_data += 1
            matrix_weight += w * len(probable_data_error[w])

        probable_error_data_parity = probable_data_errors + probable_parity_error
        normalized_weight = matrix_weight / len(parity_correct) if parity_correct else matrix_weight

        return matrix_copy, normalized_weight, probable_error_data_parity

    # ------------------------------------------------------------------
    # NEW: Iterative syndrome-driven decoder (good for ~8 errors)
    # ------------------------------------------------------------------
    def _build_check_graph(self):
        check_to_vars = {}
        var_to_checks = defaultdict(list)

        all_checks = {}
        all_checks.update(self.parity_bit_relation)
        all_checks.update(self.checksum_bit_relation)

        for check_cell, deps in all_checks.items():
            vars_in_check = [check_cell] + list(deps)
            check_to_vars[check_cell] = vars_in_check
            for v in vars_in_check:
                var_to_checks[v].append(check_cell)

        checks = list(all_checks.keys())
        return checks, check_to_vars, var_to_checks

    def _check_satisfied(self, matrix, check_cell, check_to_vars):
        x = 0
        for (r, c) in check_to_vars[check_cell]:
            x ^= int(matrix[r][c])
        return x == 0

    def _failed_checks(self, matrix, checks, check_to_vars):
        failed = set()
        for ch in checks:
            if not self._check_satisfied(matrix, ch, check_to_vars):
                failed.add(ch)
        return failed

    def _iterative_decode(self, matrix, max_flips=8, max_iters=40, beam_width=12):
        checks, check_to_vars, _ = self._build_check_graph()

        beam = [(copy.deepcopy(matrix), [])]  # (mat, flips)

        for _ in range(max_iters):
            next_candidates = []

            for mat, flips in beam:
                ok, _, _ = self._strict_matrix_ok(mat)
                if ok:
                    return mat, flips

                failed = self._failed_checks(mat, checks, check_to_vars)

                scores = Counter()
                for ch in failed:
                    for v in check_to_vars[ch]:
                        scores[v] += 1

                if not scores:
                    continue

                for (v, _) in scores.most_common(beam_width * 2):
                    if len(flips) >= max_flips:
                        continue
                    r, c = v
                    nxt = copy.deepcopy(mat)
                    nxt[r][c] = 1 if int(nxt[r][c]) == 0 else 0
                    next_candidates.append((nxt, flips + [v]))

            if not next_candidates:
                break

            scored = []
            for m, f in next_candidates:
                failed = self._failed_checks(m, checks, check_to_vars)
                scored.append((len(failed), len(f), m, f))
            scored.sort(key=lambda x: (x[0], x[1]))

            beam = [(m, f) for _, __, m, f in scored[:beam_width]]

        return -1, []

    # ------------------------------------------------------------------
    # LEGACY: your original heuristic decoder (moved here unchanged, but
    #         signature aligned with ProcessFile: threshold_data first)
    # ------------------------------------------------------------------
    def _decode_legacy(self, matrix, threshold_data, threshold_parity,
                       maximum_number_of_error, false_positive):
        """
        Your original heuristic decoder.
        NOTE: threshold_data is the DATA threshold, threshold_parity is the PARITY threshold.
        """
        matrix_details = {}

        # Initial matrix check without altering any bit
        _, matrix_weight, probable_errors = self._get_matrix_weight(
            matrix, [], threshold_parity, threshold_data, false_positive
        )

        if matrix_weight == 0:
            # IMPORTANT: strict accept, not heuristic accept
            ok, _, oriented = self._strict_matrix_ok(matrix)
            if ok:
                return self.return_matrix(oriented, [])

        # Try flipping one probable bit
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
                ok, _, oriented = self._strict_matrix_ok(changed_matrix)
                if ok:
                    return self.return_matrix(oriented, [error])

        # Sort probable fixes by lowest error weight
        matrix_details = dict(sorted(matrix_details.items(), key=lambda x: x[1]["error_value"]))

        # Try combinations of multiple bit flips (up to maximum_number_of_error)
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
                        ok, _, oriented = self._strict_matrix_ok(test_matrix)
                        if ok:
                            return self.return_matrix(oriented, list(test_combination))

                    matrix_weights.setdefault(test_weight, {"cell_checked_so_far": [], "probable_error": []})
                    matrix_weights[test_weight]["cell_checked_so_far"].append(tuple(test_combination))
                    matrix_weights[test_weight]["probable_error"].append(test_probable_errors)

                if not matrix_weights:
                    break

                sorted_weights = sorted(matrix_weights.keys())
                min_weights = sorted_weights[:2] if len(sorted_weights) >= 2 else [sorted_weights[0]]

                for mw in min_weights:
                    for i, combo in enumerate(matrix_weights[mw]["cell_checked_so_far"]):
                        retry_queue[combo] = matrix_weights[mw]["probable_error"][i]

                progressed = False
                for combo in sorted(retry_queue.keys(), key=len, reverse=True):
                    checked_combination = list(combo)
                    pending_errors = list(set(retry_queue[combo]) - set(checked_combination))
                    del retry_queue[combo]
                    progressed = True
                    if len(checked_combination) < maximum_number_of_error:
                        break

                if not progressed:
                    break

        return -1

    # ------------------------------------------------------------------
    # HYBRID decode core (iterative + strict accept + fallback to legacy)
    # ------------------------------------------------------------------
    def _decode(self, matrix, threshold_data, threshold_parity,
            maximum_number_of_error, false_positive):
        """
        Hybrid:
        0) strict accept (0 flips)
        1) iterative syndrome-driven decode (good up to ~8)
        2) fallback to legacy heuristic search
        """

        # 0) strict accept
        ok, _, oriented = self._strict_matrix_ok(matrix)
        if ok:
            return self.return_matrix(oriented, [])

        # 1) iterative decode
        fixed, flips = self._iterative_decode(
            matrix,
            max_flips=maximum_number_of_error,
            max_iters=40,
            beam_width=6
        )

        
        if not isinstance(fixed, int):
            ok, i, ori = self._strict_matrix_ok(fixed)
            if ok:
                flips = self._mirror_locations(flips, ori)
                return self.return_matrix(oriented, flips)

        # 2) fallback
        return self._decode_legacy(
            matrix,
            threshold_data,
            threshold_parity,
            maximum_number_of_error,
            false_positive
        )

    # ------------------------------------------------------------------
    # Public decode API (MUST match ProcessFile.super().decode call)
    # ------------------------------------------------------------------
    def decode(self, data_stream, threshold_data, threshold_parity,
               maximum_number_of_error, false_positive):
        """
        Decode a single origami stream.

        Signature MUST match ProcessFile usage:
          super().decode(origami_str, threshold_data, threshold_parity, max_err, false_positive)
        """
        if len(data_stream) != self.row * self.column:
            raise ValueError("The data stream length should be", self.row * self.column)

        data_matrix = self.data_stream_to_matrix(data_stream)

        # IMPORTANT: order is (threshold_data, threshold_parity)
        return self._decode(data_matrix, threshold_data, threshold_parity,
                            maximum_number_of_error, false_positive)


# Debug/test example (optional)
if __name__ == "__main__":
    # NOTE: For decode to work, you must have matrix_details/parity/checksum relations set.
    # In your pipeline, ProcessFile sets these before calling Origami.decode.
    # Here, we show how to do it manually for testing.

    orig = Origami(verbose=2)

    parity_number = 24
    data_bit_per_origami = 29  # example; must match your pipeline
    orig.matrix_details, orig.parity_bit_relation, orig.checksum_bit_relation = \
        orig._matrix_details(data_bit_per_origami, parity_number)
    orig.data_bit_to_parity_bit = orig.get_data_bit_to_parity_bit(orig.parity_bit_relation)

    # Put a real encoded stream here to test
    # encoded_stream = orig._encode("00110110010101010110101011010", 0, data_bit_per_origami, parity_number)
    # decoded = orig.decode(encoded_stream, threshold_data=2, threshold_parity=3, maximum_number_of_error=8, false_positive=0)
    # print(decoded)
    pass