from typing import Any, List, Tuple
import torch


def product_sum(l: List[float], r: List[float]) -> List[float]:
    return sum([a * b for a, b in zip(l, r)])

def conv1d_list(
        lst: List[float],
        kernel: List[float],
        padding: int = 0,
        stride: int = 1
) -> List[float]:
    
    # lst = [e_1, e_2, e_3, .... e_n]
    # kernel: [k_1, k_2, k_3, ... k_m]

    lst = [0] * padding + lst + [0] * padding
    # n = len(lst) + 2 * padding
    n = len(lst)
    k = len(kernel)
    assert k < n

    features = []
    for i in range(0, n - k + 1, stride):
        features.append(product_sum(lst[i:i + k], kernel))

    return features

def product_loop(*loops):
    res = []
    if len(loops) == 1:
        return [[e] for e in loops[0]]
    
    for i in loops[0]:
        for rest in product_loop(loops[1:]):
            res.append([i] + rest)
    
    return res

def get_dimension(lst):
    if not isinstance(lst[0], list):
        return 1
    else:
        return 1 + get_dimension(lst[0])
    
def change_element(lst, indices, elem):
    n = get_dimension(lst)
    assert n == len(indices)
    
    if n == 1:
        lst[indices[0]] = elem
    else:
        for idx, sub_list in enumerate(lst):
            if idx == indices[0]:
                change_element(sub_list, indices[1:], elem)

def segment_lst(lst, indices: List[Tuple]):
    res = lst
    for (start, end) in indices:
        res = res[start:end]
    
    return res

def get_lst(lst, indices):
    res = lst

    for i in indices:
        res = lst[i]
    
    return res

def product_sum_ndim(a, b):
    n = get_dimension(a)
    m = get_dimension(b)

    assert n == m
    l = len(a)
    s = 0
    for indices in product_loop(*[range(l) for _ in range(n)]):
        s += get_lst(a, indices) * get_lst(b, indices)
    
    return s

def convnd_list(
        lst: Any,
        kernel: Any,
        padding: int = 0,
        stride: int = 1,
        conv_dim: int = 2,
) -> Any:
    '''
    for i_1 in range(0, n - k + 1, stride):
        for i_2 in range(0, n - k + 1, stride):
    [i_1, i_2, ..... i_n] ~ [i_1 + k, i_2 + k, ...... i_n + k]
    features[i_1, i_2, ...., i_n] = product_sum_ndim(kerner, lst[i_1:i_1 + k][i_2:i_2 + k].....[i_n:i_n + k])
    '''
    N = get_dimension(lst)
    n = len(lst)
    k = len(kernel)

    features = torch.zeros(*[n for _ in range(N)]).tolist()

    for indices in product_loop([range(0, n - k + 1, stride) for _ in range(N)]):
        segment_range = [(i, i + k) for i in indices]
        segment = segment_lst(lst, segment_range)
        change_element(features, indices, product_sum_ndim(kernel, segment))
    
    return features
    

if __name__ == '__main__':
    lst = conv1d_list(
        list(range(1, 6)),
        [1, 0, 1]
    )
    # [1, 2, 3, 4, 5]
    # [1, 0, 1]
    # [4, 6, 8]
    print(lst)
