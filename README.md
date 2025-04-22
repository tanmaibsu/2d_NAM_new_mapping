# Nucleic Acid Memory for 8X10 matrix
This artifact implements the new parity bit mapping. The previous implementation works with 40 parity bits which reduces the code rate. It also experiments the number of errors the decoding algorithm can correct. After changing the parity bits, still how much errors the decoding algorithm can correct.

### Requirements:
The codes are tested with **python 3.7**  
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages
```bash
pip install numpy scipy numba matplotlib lmfit tqdm yaml h5py

```
Or use the requirements.txt file:
```bash
pip install -r requirements.txt
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

#### the encoding contains 24 parity bits
```
python3 error_correction/encode.py -f test_input.txt -o test_output.txt -pn 24

```

#### Decoding
Use the following command to decode any encoded file:
```
python3 error_correction/decode.py
                  -h, --help, show this help message and exit
                  -f, --file_in, File to decode
                  -o, --file_out, File to write output
                  -pn, --parity_number, number of parity bits for error correction, default=40
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

### Simultation with randomly inserted errors.
#### We added errors in the encoded origami to check how much errors the decoding algorithm can correct.
#### It is done by randomly flipping bits 1 to 0.


#### Please run the following command for the simulation
```
python error_correction/decode.py -bulk origamis_24 -o decoded_output -pn 24
```