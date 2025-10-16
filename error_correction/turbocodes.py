# import random

# def convolutional_encode(bits, g1=0b111, g2=0b101):
#     # Constraint length = 3, g1 and g2 are generator polynomials
#     state = [0, 0]
#     output = []
#     for bit in bits:
#         state = [bit] + state[:-1]
#         o1 = bit ^ state[0] ^ state[1]  # g1: 111
#         o2 = bit ^ state[1]             # g2: 101
#         output.append((bit, o1, o2))    # systematic + parity
#     return output

# def interleave(bits, seed=42):
#     random.seed(seed)
#     indices = list(range(len(bits)))
#     random.shuffle(indices)
#     return [bits[i] for i in indices], indices

# def deinterleave(bits, indices):
#     output = [0]*len(bits)
#     for i, idx in enumerate(indices):
#         output[idx] = bits[i]
#     return output

# def add_noise(encoded, num_errors=5):
#     flat = [bit for triplet in encoded for bit in triplet]
#     error_indices = random.sample(range(len(flat)), num_errors)
#     for idx in error_indices:
#         flat[idx] ^= 1  # Flip bit
#     # Reconstruct into triplets
#     return [tuple(flat[i:i+3]) for i in range(0, len(flat), 3)]

# def majority_vote(a, b):
#     return [int((x + y) >= 1) for x, y in zip(a, b)]

# def turbo_decode(received, interleaver_indices, iterations=3):
#     # Simplified soft decision: use majority vote over iterations
#     L = len(received)
#     decoded = [triplet[0] for triplet in received]  # start with systematic bits
#     for _ in range(iterations):
#         # Decoder 1
#         est1 = [triplet[0] for triplet in received]
#         # Decoder 2
#         interleaved = [triplet[0] for triplet in received]
#         interleaved = deinterleave(interleaved, interleaver_indices)
#         est2 = interleaved
#         # Combine
#         decoded = majority_vote(est1, est2)
#     return decoded

# # === Main ===
# data = '01000100011111000011110000101111101010100011000000001001000010100011000001101000'
# bits = [int(b) for b in data]

# # Encode
# encoded1 = convolutional_encode(bits)
# interleaved_bits, indices = interleave(bits)
# encoded2 = convolutional_encode(interleaved_bits)
# print("encoded2-->", encoded2)

# # Combine outputs
# combined = [(sys, p1, p2) for (sys, p1, _), (_, _, p2) in zip(encoded1, encoded2)]

# print("combined", combined)

# # Introduce errors
# noisy = add_noise(combined, num_errors=1)
# print("noisy", noisy)

# # Decode
# decoded_bits = turbo_decode(combined, indices)


# # Compare
# decoded_str = ''.join(map(str, decoded_bits))
# original_str = ''.join(map(str, bits))

# print("Original: ", original_str)
# print("Decoded : ", decoded_str)
# print("Success : ", original_str == decoded_str)

from itpp import turbo, vec, bvec

# Convert your bit string to itpp bit vector
data = '01000100011111000011110000101111101010100011000000001001000010100011000001101000'
input_bits = bvec([int(b) for b in data])

# Create turbo coder and decoder
tc = turbo.Turbo_Code()
tc.set_parameters([13, 15], 4, 20)  # Constraint length=4, generators in octal, interleaver size=20
tc.set_interleaver_random(len(input_bits))

# Encode
encoded = tc.encode(input_bits)

# Simulate channel (no errors first)
received = vec([1.0 if bit == 0 else -1.0 for bit in encoded])  # BPSK mapping

# Decode
decoded = tc.decode(received, 5)  # 5 iterations

# Check result
decoded_bits = ''.join([str(int(b)) for b in decoded])
print("Original :", data)
print("Decoded  :", decoded_bits)
print("Success  :", data == decoded_bits)

