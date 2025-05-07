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

## Simultation with randomly inserted errors.
### We added errors in the encoded origami to check how much errors the decoding algorithm can correct.

### It is done by randomly flipping bits 1 to 0.


### Please run the following command for the simulation.
```
python error_correction/decode.py -bulk origamis_24 -o decoded_output -pn 24
```