from typing import List, Tuple, Iterator,Optional

def find_idx(arr: List[int], start: int) -> Optional[int]:
    """
    Find the index of the first 1 in arr at or after `start`.

    :param arr: List[int] of 0s and 1s
    :param start: Index to begin the search from
    :return: Index of the next 1, or None if no 1 is found
    """
    n = len(arr)
    # clamp start to [0, n)
    i = max(0, start)
    while i < n:
        if arr[i] == 1:
            return i
        i += 1
    return None



def flip_bits_exhaustively(arr: List[int], n, idx):
    if n == 1:
        idx_to_flip = find_idx(arr, idx)
        new_arr = arr.copy()
        new_arr[idx] = 0

        return idx, new_arr
                
