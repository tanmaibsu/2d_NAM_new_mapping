import numpy as np
from turbo_encoder import deinterleave


def siso_decode(sys_llr, parity_llr, apriori_llr, n_iter=1):
    N = len(sys_llr)
    Lext = np.zeros(N)
    state_trans = [
        [(0, 0), (2, 1)],
        [(0, 1), (2, 0)],
        [(1, 1), (3, 0)],
        [(1, 0), (3, 1)],
    ]
    for _ in range(n_iter):
        alpha = np.full((N + 1, 4), -np.inf)
        beta = np.full((N + 1, 4), -np.inf)
        gamma = np.zeros((N, 4, 2))
        alpha[0][0] = 0
        beta[N][0] = 0

        for i in range(N):
            for s in range(4):
                for b in [0, 1]:
                    next_s, out = state_trans[s][b]
                    branch_metric = 0.5 * (sys_llr[i] * b + parity_llr[i] * out + apriori_llr[i] * b)
                    gamma[i][s][b] = branch_metric

        for i in range(1, N + 1):
            for s in range(4):
                for b in [0, 1]:
                    prev_s, _ = state_trans[s][b]
                    alpha[i][s] = max(alpha[i][s], alpha[i - 1][prev_s] + gamma[i - 1][prev_s][b])

        for i in range(N - 1, -1, -1):
            for s in range(4):
                for b in [0, 1]:
                    next_s, _ = state_trans[s][b]
                    beta[i][s] = max(beta[i][s], beta[i + 1][next_s] + gamma[i][s][b])

        for i in range(N):
            num = max([alpha[i][s] + gamma[i][s][1] + beta[i + 1][state_trans[s][1][0]] for s in range(4)])
            den = max([alpha[i][s] + gamma[i][s][0] + beta[i + 1][state_trans[s][0][0]] for s in range(4)])
            Lext[i] = num - den

        apriori_llr = Lext.copy()
    return Lext


def turbo_decode(sys, parity1, parity2, interleaver_indices, n_iter=5):
    N = len(sys)
    apriori1 = np.zeros(N)
    apriori2 = np.zeros(N)

    LLR = np.zeros(N)
    for _ in range(n_iter):
        ext1 = siso_decode(sys, parity1, apriori1)
        interleaved_ext1 = [ext1[i] for i in interleaver_indices]
        apriori2 = interleaved_ext1

        interleaved_sys = [sys[i] for i in interleaver_indices]
        ext2 = siso_decode(interleaved_sys, parity2, apriori2)
        deinterleaved_ext2 = deinterleave(ext2, interleaver_indices)

        LLR = np.array(sys) + np.array(deinterleaved_ext2)
        apriori1 = deinterleaved_ext2

    decoded = [0 if l >= 0 else 1 for l in LLR]
    return decoded