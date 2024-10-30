from collections import Counter
import math

def n_grams(sequence, n):
    return [tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)]

def count_grams(sequence, n):
    return Counter(n_grams(sequence, n))
def modified_precision(candidate, reference, n):
    candidate_grams = count_grams(candidate, n)
    reference_grams = count_grams(reference, n)

    overlap = candidate_grams & reference_grams
    overlap_count = sum(overlap.values())

    total_count = sum(candidate_grams.values())

    if total_count == 0:
        return 0
    
    return overlap_count / total_count

def brevity_penalty(candidate, reference):
    candidate_len = len(candidate)
    reference_len = len(reference)

    if candidate_len > reference_len:
        return 1
    elif candidate_len == 0:
        return 0
    else:
        return math.exp(1 - reference_len / candidate_len)

def bleu(candidate, reference, max_n = 4):
    precisions = []
    for n in range(1, max_n+1):
        precisions.append(modified_precision(candidate, reference, n))

    if all(p == 0 for p in precisions):
        bleu_score = 0
    else:
        bleu_score = math.exp(sum(math.log(p) for p in precisions if p > 0) / max_n)
    
    bp = brevity_penalty(candidate, reference)

    return bleu_score * bp
