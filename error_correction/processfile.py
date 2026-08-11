import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from log import get_logger
from origami_greedy import Origami
import csv
import json
import os

class ProcessFile(Origami):
    """
    Orchestrates encoding/decoding of origami segments, buffers per-origami
    results in memory, and can flush them to CSV on demand.
    """

    def __init__(self, verbose):
        super().__init__(verbose=verbose)
        self.verbose = verbose
        self.logger = get_logger(verbose, __name__)

        # Per-origami buffers (persist across decode calls if accumulate=True)
        self.node = []
        self.origami_data = []
        self.decoded_stream = []
        self.false_negatives = []
        self.false_positives = []
        self.success = []
        self.decoded_time = []
        self.error_positions = []

        # Default (can be overridden by your layout)
        self.number_of_bit_per_origami = 29

    # ---------- Buffer helpers ----------
    def reset_ior_buffers(self):
        """Clear buffered per-origami rows (use before a new batch, if desired)."""
        self.node.clear()
        self.origami_data.clear()
        self.decoded_stream.clear()
        self.false_negatives.clear()
        self.false_positives.clear()
        self.success.clear()
        self.decoded_time.clear()

    def write_ior_csv(self, csv_path):
        """Flush current buffers to a CSV file."""
        header = ["node", "origami_data", "decoded_stream",
                  "false_negatives", "false_positives", "success", "decoding_time"]
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(len(self.node)):
                    # err_pos = self.errors_positions[i]
                    # if not isinstance(err_pos, str):
                    #     err_pos = json.dumps(err_pos)
                    writer.writerow([
                        self.node[i],
                        self.origami_data[i],
                        self.decoded_stream[i],
                        self.false_negatives[i],
                        self.false_positives[i],
                        int(bool(self.success[i])),
                        self.decoded_time[i],
                    ])
            self.logger.info("Wrote per-origami CSV: %s", csv_path)
        except Exception as e:
            self.logger.exception("Failed writing CSV %s: %s", csv_path, e)

    def _append_data_4_io(self, orig_idx, origami_data, decoded_stream,
                          false_negatives, false_positives, success, decoding_time):
        # Appends are atomic in CPython; safe enough with threads for this use.
        self.node.append(orig_idx)
        self.origami_data.append(origami_data)
        self.decoded_stream.append(decoded_stream)
        self.false_negatives.append(false_negatives),
        self.false_positives.append(false_positives),
        self.success.append(bool(success))
        self.decoded_time.append(decoding_time)

    # ---------- Capacity helpers ----------
    def _find_optimum_index_bits(self, bits_needed_to_store, parity_number):
        """
        Returns: (index_bits, data_bits_per_origami, segment_count)
        """
        total_capacity = self.row * self.column
        checksum_allocation = len(self.get_checksum_relation(parity_number))
        parity_allocation = len(self.get_parity_relation(parity_number))
        # Reserve 4 orientation bits by convention
        available_capacity = total_capacity - checksum_allocation - parity_allocation - 4

        for i in range(1, available_capacity):
            data_bits_per_origami = available_capacity - i
            segment_count = math.ceil(bits_needed_to_store / data_bits_per_origami)
            if 2 ** i >= segment_count:
                return i, data_bits_per_origami, segment_count
        raise Exception("File size is too large to store in the given capacity")

    # ---------- Encode ----------
    def encode(self, file_in, file_out, formatted_output=False, parity_number=40):
        """
        Returns: (segment_count, data_bits_per_origami) or (-1, -1) on failure
        """
        try:
            with open(file_in, 'rb') as fin:
                data = fin.read()
            fout = open(file_out, "w")
        except Exception as e:
            self.logger.exception(e)
            self.logger.error("Error opening the file(s)")
            return -1, -1

        data_in_binary = ''.join(format(b, '08b') for b in data)
        bits_needed_to_store = len(data_in_binary)
        _, data_bit, segment_size = self._find_optimum_index_bits(bits_needed_to_store, parity_number)

        for index in range(segment_size):
            start = index * data_bit
            end = start + data_bit
            origami_bits = data_in_binary[start:end].ljust(data_bit, '0')  # pad if needed
            encoded_stream = self._encode(origami_bits, index, data_bit, parity_number)

            if formatted_output:
                print(f"Matrix -> {index}", file=fout)
                self.print_matrix(self.data_stream_to_matrix(encoded_stream), in_file=fout)
            else:
                print(encoded_stream, file=fout)

        fout.close()
        self.logger.info("Encoding done")
        return segment_size, data_bit

    # ---------- Worker (thread) ----------
    def single_origami_decode(self, single_origami, ior_file_name, correct_dictionary,
                              common_parity_index, minimum_temporary_weight,
                              maximum_number_of_error, false_positive, induced_errors,
                              errors_positions, original_origami, orig_idx, false_negatives, false_positives):
        start_time = time.time()
        index, origami_str = single_origami

        self.logger.info("Decoding origami (%d)", index)
        if len(origami_str) != self.row * self.column:
            self.logger.warning(
                "Origami (%d) is incomplete. Expected: %d, Found: %d",
                index, self.row * self.column, len(origami_str)
            )
            return {
                "io_row": dict(
                    orig_idx=orig_idx, origami_data=origami_str, decoded_stream="",
                    false_negatives=false_negatives, false_positives=false_positives, success=False,
                    decoding_time=round(time.time() - start_time, 3),
                ),
                "summary": None
            }

        try:
            decoded_matrix = super().decode(
                origami_str,
                common_parity_index,
                minimum_temporary_weight,
                maximum_number_of_error,
                false_positive
            )
        except Exception as e:
            self.logger.exception("Decoding failed for origami (%d): %s", index, str(e))
            return {
                "io_row": dict(
                    orig_idx=orig_idx, origami_data=origami_str, decoded_stream="",
                    false_negatives=false_negatives, false_positives=false_positives, success=False,
                    decoding_time=round(time.time() - start_time, 3),
                ),
                "summary": None
            }

        if decoded_matrix == -1:
            self.logger.warning("Decoding unsuccessful for origami (%d)", index)
            return {
                "io_row": dict(
                    orig_idx=orig_idx, origami_data=origami_str, decoded_stream="",
                    false_negatives=false_negatives, false_positives=false_positives, success=False,
                    decoding_time=round(time.time() - start_time, 3),
                ),
                "summary": None
            }

        decoded_index = decoded_matrix['index']
        decoded_data = decoded_matrix['binary_data']
        error_count = int(decoded_matrix['total_probable_error'])

        # Optional correctness vs correct_dictionary
        status = None
        if correct_dictionary:
            try:
                status = int(correct_dictionary[int(decoded_index)] == decoded_data)
            except Exception as e:
                self.logger.warning("Comparison error for origami (%d): %s", index, str(e))
                status = -1

        decoded_stream = self.matrix_to_data_stream(decoded_matrix['matrix'])
        io_row = dict(
            orig_idx=orig_idx,
            origami_data=origami_str,
            decoded_stream=decoded_stream,
            false_negatives=false_negatives, 
            false_positives=false_positives,
            success=(original_origami == decoded_stream),
            decoding_time=round(time.time() - start_time, 3),
        )
        summary = {
            "index": decoded_index,
            "binary_data": decoded_data,
            "total_probable_error": error_count,
            "status": status,
        }
        return {"io_row": io_row, "summary": summary}

    # ---------- Decode (accumulate + optional flush) ----------
    def decode(self, data, original_origami, orig_idx, induced_errors, errors_positions,
               file_out, file_size, parity_number, threshold_data, threshold_parity,
               maximum_number_of_error, individual_origami_info, false_positive, false_negatives,
               false_positives, correct_file=False, *, scheme="last_two",
               accumulate=True, write_csv=False, csv_path=None):
        """
        Decodes a batch of origami strings.
        - If accumulate=True (default), buffers are NOT cleared; results append.
          If accumulate=False, buffers are cleared before decoding.
        - If write_csv=True, writes CSV at the end of this call (path controlled by csv_path).
          Otherwise, call self.write_ior_csv(path) later to flush once for many calls.

        Returns: (status, incorrect_count, correct_count, total_error_fixed, missing_list)
        """
        self.logger.info("Errors Positions: %s", errors_positions)

        if not accumulate:
            self.reset_ior_buffers()

        correct_origami = 0
        incorrect_origami = 0
        total_error_fixed = 0

        # Layout details
        _, data_bit, segment_size = self._find_optimum_index_bits(file_size * 8, parity_number)
        self.matrix_details, self.parity_bit_relation, self.checksum_bit_relation = self._matrix_details(data_bit, parity_number, scheme)
        self.data_bit_to_parity_bit = self.get_data_bit_to_parity_bit(self.parity_bit_relation)

        # Load correct file if provided
        correct_dictionary = {}
        if correct_file:
            try:
                with open(correct_file) as cf:
                    for so in cf:
                        ci, cd = self._extract_text_and_index(self.data_stream_to_matrix(so.strip()))
                        correct_dictionary[ci] = cd
            except Exception as e:
                self.logger.exception("Failed to load correct_file '%s': %s", correct_file, e)

        # Prepare input list
        origami_data_list = [(i, origami.strip()) for i, origami in enumerate(data)]

        # Threaded decode to avoid multiprocessing pickling issues
        max_workers = min(32, (os.cpu_count() or 1))
        decoded_dictionary_wno = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.single_origami_decode,
                    single_origami=pair,
                    ior_file_name=None,
                    correct_dictionary=correct_dictionary,
                    common_parity_index=threshold_data,
                    minimum_temporary_weight=threshold_parity,
                    maximum_number_of_error=maximum_number_of_error,
                    false_positive=false_positive,
                    induced_errors=induced_errors,
                    errors_positions=errors_positions,
                    original_origami=original_origami,
                    orig_idx=orig_idx,
                    false_negatives=false_negatives, 
                    false_positives=false_positives
                )
                for pair in origami_data_list
            ]

            for fut in as_completed(futures):
                out = fut.result()
                io_row = out["io_row"]
                summary = out["summary"]

                # Buffer row in memory
                self._append_data_4_io(
                    io_row["orig_idx"],
                    io_row["origami_data"],
                    io_row["decoded_stream"],
                    io_row["false_negatives"],
                    io_row["false_positives"],
                    io_row["success"],
                    io_row["decoding_time"],
                )

                # Aggregate for majority voting + stats
                if summary:
                    idx = summary["index"]
                    bin_data = summary["binary_data"]
                    decoded_dictionary_wno.setdefault(idx, []).append(bin_data)
                    total_error_fixed += summary["total_probable_error"]

                    if correct_file:
                        if summary["status"] == 1:
                            correct_origami += 1
                        elif summary["status"] == 0:
                            incorrect_origami += 1

        # Majority vote to reconstruct
        final_data = [None] * segment_size
        for idx, binaries in decoded_dictionary_wno.items():
            try:
                int_idx = int(idx)
            except Exception:
                continue
            if 0 <= int_idx < segment_size and binaries:
                final_data[int_idx] = Counter(binaries).most_common(1)[0][0]

        missing_origami = [i for i, val in enumerate(final_data) if val is None]
        if missing_origami:
            # Optional per-call flush if asked
            if write_csv or individual_origami_info:
                path = csv_path or f"{file_out}_ior.csv"
                self.write_ior_csv(path)
            return -1, incorrect_origami, correct_origami, total_error_fixed, missing_origami

        # Convert recovered bits to exactly file_size bytes
        recovered_binary = "".join(final_data)
        out_bytes = bytearray()
        for i in range(0, len(recovered_binary), 8):
            chunk = recovered_binary[i:i + 8]
            if len(chunk) < 8:
                break
            out_bytes.append(int(chunk, 2))

        if len(out_bytes) < file_size:
            out_bytes.extend(b"\x00" * (file_size - len(out_bytes)))
        elif len(out_bytes) > file_size:
            out_bytes = out_bytes[:file_size]

        # Write recovered file
        try:
            with open(file_out, "wb") as out_file:
                out_file.write(bytes(out_bytes))
        except Exception as e:
            self.logger.exception("Failed writing recovered file '%s': %s", file_out, e)
            # Optional per-call flush if asked even on error
            if write_csv or individual_origami_info:
                path = csv_path or f"{file_out}_ior.csv"
                self.write_ior_csv(path)
            return -1, incorrect_origami, correct_origami, total_error_fixed, []

        # Optional per-call CSV flush
        if write_csv or individual_origami_info:
            path = csv_path or f"{file_out}_ior.csv"
            self.write_ior_csv(path)

        self.logger.info("Total error fixed: %s", total_error_fixed)
        self.logger.info("File recovery successful")
        return 1, incorrect_origami, correct_origami, total_error_fixed, []
