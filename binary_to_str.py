def binary_to_string(binary_lines):
    """
    Converts multi-line binary data into decoded ASCII string.
    Each line can contain any length of bits (must be multiple of 8 for valid decoding).
    """
    joined = "".join(binary_lines).replace("\n", "").strip()
    
    # Split into bytes of 8 bits
    bytes_list = [joined[i:i+8] for i in range(0, len(joined), 8)]
    
    # Convert each byte to ASCII character (ignore incomplete ones)
    chars = []
    for b in bytes_list:
        if len(b) == 8:
            chars.append(chr(int(b, 2)))
    return "".join(chars)

# Example input
binary_lines = [
    "01000100011100011101110000100111101010000111000010011001000011010000100001101000",
    "01011100111111110001000100000000111101000010101110111100010011000011000000011001",
    "11110111011111101001001011100011000001100000100001000010001110000011000011100110",
    "00000100101000010101000010000010010100000100100000000000000010000111100000000011"
]

decoded_text = binary_to_string(binary_lines)
print("Decoded string:")
print(decoded_text)