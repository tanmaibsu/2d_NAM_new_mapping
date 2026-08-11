import random


def rsc_encode(bits):
    state = [0, 0]
    sys = []
    parity = []
    for bit in bits:
        input_bit = bit ^ state[1]
        p = input_bit ^ state[0] ^ state[1]
        sys.append(bit)
        parity.append(p)
        state = [input_bit] + state[:1]
    return sys, parity


def interleave(bits, seed=42):
    rng = random.Random(seed)
    indices = list(range(len(bits)))
    rng.shuffle(indices)
    return [bits[i] for i in indices], indices


def deinterleave(bits, indices):
    result = [0] * len(bits)
    for i, idx in enumerate(indices):
        result[idx] = bits[i]
    return result
