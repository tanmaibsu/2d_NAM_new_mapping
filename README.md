# Nucleic Acid Memory for 8X10 matrix
This artifact implements the new parity bit mapping. The previous implementation works with 40 parity bits which reduces the code rate. It also experiments the number of errors the decoding algorithm can correct. After changing the parity bits, still how much errors the decoding algorithm can correct.

# Implemented Functionalities
 - The previous code could not handle the input for decoding if we provided it as a matrix. It could handle only the linear array. 
 - While randomly inserting errors to check the number of errors, I write the code to flip only the bits with a "1". 
 - For the command line argument, I added the number of parity bits that can be provided while encoding or decoding, which is needed to run with a different number of parity bits.
 - I needed to adapt all the portions of the code accordingly for this. Now we can decode in bulk too, rather than only a single file. I added two command-line arguments, which are mutually exclusive for this. 
 - In the previous implementation, the parity mapping was hand-picked. It is very time-consuming to create the mapping manually. This time I have generated the mapping dynically. Rules for the parity mapping:
    - There should be no repeating positions in each parity relation.
    - No mirrored point for any point in a parity relation.
    - For each of the parity positions, the corresponding axis (X, Y, XY) mirror point should be found.
    - There should be some point which are repeated in different parity positions.
 - The two parity-24 mappings (for the first vs. last group of nodes) used to be swapped by hand in `error_correction/get_parity_n_checksum.py`. They are now selected at runtime with the `--scheme` command-line flag, so no code editing is required between decode runs.
 - Wet-lab CSV decoding now discovers node ids automatically from the input file and accepts the reference origami list from a file (`--reference_file`), so it is no longer hardcoded to a fixed number of nodes.

## Requirements:
The codes are tested with **python 3.7**  
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages
```bash
pip install numpy scipy numba matplotlib lmfit tqdm yaml h5py

```
Or use the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Usages

### Generate Parity Mapping
**To generate mapping for the defined number of parity bits. Please run the following command for 24 parity bits with 12 parity coverage.**

```bash
python3 error_correction/generate_parity_mapping.py -pn 24 -pc 12 
```

**For 40 parity bits with 4 parity coverage. The command is: **
```bash
python3 error_correction/generate_parity_mapping.py -pn 40 -pc 4
```

## Error correction encoding/decoding algorithm

### Usage of error correction code
#### Encoding
User the following command to encode a given file to a list of origami matrices
```
python3 error_correction/encode.py
                    -h , --help, show this help message and exit
                    -f , --file_in, file to encode
                    -pn , --parity_number, number of parity bits for error correction
                    -o , --file_out, File to write the output
                    -fo, --formatted_output, Print the origami as matrix instead of single line
                    -v , --verbose, Print details on the console. 0 -> error. 1->debug, 2->info, 3->warning
```
### Example
#### the encoding contains 40 parity bits
```
python error_correction/encode.py -f test_input.txt -o test_output.txt -pn 40

```


#### Decoding options
```
python3 error_correction/decode.py
                    -h , --help, show this help message and exit
                    -f , --file_in, single file to decode (mutually exclusive with -bulk)
                    -bulk , --bulk_folder, folder/CSV to decode (mutually exclusive with -f)
                    -o , --file_out, file to write the output (required)
                    -pn , --parity_number, number of parity bits used during encoding (default 40)
                    -fz , --file_size, file size that was encoded (default 20)
                    -s , --scheme, parity-24 scheme: first_two | last_two (default last_two)
                    -rf , --reference_file, file of reference origami strings, one per line (node i = line i)
                    -n , --nodes, comma-separated node ids to decode, e.g. 0,1 (default: all nodes in the file)
                    -m , --mode, which routine to run: wetlab | single | exhaustive (default wetlab)
                    -eb , --error_bits, bits to flip per combination in --mode exhaustive (default 1)
                    -cf , --correct_file, original encoded file, used to auto-check decoding status
                    -v , --verbose, 0 -> error, 1 -> debug, 2 -> info, 3 -> warning
```

The run mode is selected with `--mode` (no need to edit the code):
 - `--mode wetlab` *(default)* — decode a wet-lab CSV passed with `-bulk` (see below).
 - `--mode single` — decode a single encoded file passed with `-f`.
 - `--mode exhaustive` — for every file in the `-bulk` folder, flip every combination of
   `--error_bits` set bits and decode, to measure error-correction capability.

```bash
# single file
python3 error_correction/decode.py --mode single -f origamis_24/origami1.txt -o decoded_output -pn 24

# exhaustive 2-bit error test over a folder
python3 error_correction/decode.py --mode exhaustive -bulk origamis_24 -o decoded_output -pn 24 --error_bits 2
```

### Example
#### if you want to decode single file with 40 parity bits
```
python3 error_correction/decode.py -f origamis/origami1.txt -o decoded_output -pn 40

```

#### if you want to decode single file with 24 parity bits
```
python3 error_correction/decode.py -f origamis_24/origami1.txt -o decoded_output -pn 24

```

#### You can also decode in bulk. Please provide a folder path with encoded origamis.
```
python3 error_correction/decode.py -bulk origamis_24 -o decoded_output -pn 24

```

### Parity-24 scheme selection (`--scheme`)
When decoding with **24 parity bits**, two hand-tuned parity mappings exist depending on
which group of nodes you are decoding. Previously the mapping was swapped by hand inside
`error_correction/get_parity_n_checksum.py`; this is now selected with the `--scheme` flag:

 - `--scheme first_two` — use when decoding the **first half** of a node group.
 - `--scheme last_two` — use when decoding the **last half** of a node group *(default)*.

The flag only affects `-pn 24`; it is ignored for 16 and 40 parity bits.

```bash
# decode the first two nodes with their scheme
python3 error_correction/decode.py -bulk encoded_data_wetlab.csv -o decoded_first_two -pn 24 --scheme first_two --nodes 0,1

# decode the last two nodes with their scheme
python3 error_correction/decode.py -bulk encoded_data_wetlab.csv -o decoded_last_two  -pn 24 --scheme last_two  --nodes 2,3
```

### Decoding wet-lab data (CSV input)
Wet-lab decoding reads a CSV (passed via `-bulk`) that contains the columns
`ID`, `Binary String`, `False Negatives`, and `False Positives`. The node ids are
discovered automatically from the `ID` column, so there is no need to edit the code to
add/skip nodes — restrict them with `--nodes` if desired.

Reference (originally encoded) origami strings are supplied with `--reference_file`
(one binary string per line, where line *i* is the reference for node *i*). This supports
an **arbitrary number of nodes**. If `--reference_file` is omitted, a built-in 4-node
reference list is used. Ready-made reference files are provided in the repository root:

 - `reference_origamis_4_nodes.txt` — 4-node reference sequences.
 - `reference_origamis_6_nodes.txt` — 6-node reference sequences.

```bash
# 4-node wet-lab decode
python3 error_correction/decode.py \
    -bulk encoded_data_wetlab.csv \
    -o decoded_wetlab \
    -pn 24 \
    --scheme last_two \
    --reference_file reference_origamis_4_nodes.txt \
    --nodes 0,1,2,3

# 6-node wet-lab decode
python3 error_correction/decode.py \
    -bulk 2026-01-12_P11_mixed_6_rep_1_decoded.csv \
    -o decoded_wetlab_6 \
    -pn 24 \
    --scheme last_two \
    --reference_file reference_origamis_6_nodes.txt \
    --nodes 0,1,2,3,4,5
```

## Simultation with randomly inserted errors.
### We added errors in the encoded origami to check how much errors the decoding algorithm can correct.

### It is done by randomly flipping bits 1 to 0.


### Please run the following command for the simulation.
```
python error_correction/decode.py -bulk origamis_24 -o decoded_output -pn 24
```