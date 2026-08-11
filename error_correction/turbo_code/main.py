from turbo_encoder import rsc_encode, interleave, deinterleave
from turbo_decoder import turbo_decode

# Step 1: Convert message to binary
message = "Data is in our DNA!\n"
bitstr = ''.join(format(ord(c), '08b') for c in message)
data = [int(b) for b in bitstr]

# Step 2: Split data into origami-sized chunks BEFORE encoding
ORIGAMI_SIZE = 80
chunks = [data[i:i+ORIGAMI_SIZE//3] for i in range(0, len(data), ORIGAMI_SIZE//3)]  # approx. 1/3 of sheet
print("chunks", chunks)

encoded_chunks = []
interleaver_indices_list = []
for chunk in chunks:
    sys1, parity1 = rsc_encode(chunk)
    interleaved_data, inter_idx = interleave(chunk)
    _, parity2 = rsc_encode(interleaved_data)
    punctured_parity2 = [p for i, p in enumerate(parity2) if i % 2 == 0]
    encoded = sys1 + parity1 + punctured_parity2
    encoded_chunks.append(encoded)
    interleaver_indices_list.append((inter_idx, len(chunk)))

print("encoded_chunks", len(encoded_chunks))
# Flatten encoded chunks
encoded_bits = [b for chunk in encoded_chunks for b in chunk]
print("encode_bits", encoded_bits)
origami_pages = [encoded_bits[i:i+ORIGAMI_SIZE] for i in range(0, len(encoded_bits), ORIGAMI_SIZE)]
print("Total origami sheets (8x10):", len(origami_pages))

# Step 3: Decode each chunk
recovered_bits = []
start = 0
for (inter_idx, chunk_len), chunk in zip(interleaver_indices_list, chunks):
    sys1, parity1 = rsc_encode(chunk)
    interleaved_data, _ = interleave(chunk)
    _, parity2 = rsc_encode(interleaved_data)
    punctured_parity2 = [p for i, p in enumerate(parity2) if i % 2 == 0]
    recovered_parity2 = [0] * len(chunk)
    recovered_parity2[::2] = punctured_parity2
    decoded = turbo_decode(sys1, parity1, recovered_parity2, inter_idx)
    recovered_bits.extend(decoded)

print("recovered_bits", recovered_bits)

# Convert bits to text
recovered_text = ''.join(chr(int(''.join(map(str, recovered_bits[i:i+8])), 2)) for i in range(0, len(recovered_bits), 8))

print("Recovered text:")
print(recovered_text)