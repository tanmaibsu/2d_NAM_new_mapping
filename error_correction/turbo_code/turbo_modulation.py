import numpy as np


def bpsk(bits):
    return np.array([1.0 if b == 0 else -1.0 for b in bits])


def add_awgn(signal, snr_db):
    snr = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * snr))
    noise = np.random.normal(0, sigma, len(signal))
    return signal + noise
