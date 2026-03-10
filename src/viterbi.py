"""
Viterbi Algorithm for HMM-based sequence decoding (POS tagging).

Implements the classic dynamic-programming Viterbi decoder that finds
the most-probable tag sequence for a given observation sequence using
log-probabilities to avoid underflow.
"""

import math
from src.hmm_model import HMMModel


def viterbi_decode(model: HMMModel, words: list) -> list:
    """
    Run the Viterbi algorithm on a single sentence.

    Parameters
    ----------
    model : HMMModel
        A trained HMM model.
    words : list of str
        Observed word sequence (one sentence, without START/END tokens).

    Returns
    -------
    list of str
        The most-probable POS tag sequence of the same length as *words*.
    """
    tags = list(model.tags)
    n = len(words)

    if n == 0:
        return []

    # viterbi[t][tag] = best log-probability up to position t with tag
    viterbi = [dict() for _ in range(n)]
    # backpointer[t][tag] = previous tag that led to the best path
    backpointer = [dict() for _ in range(n)]

    # --- Initialisation (t = 0) ---
    word0 = words[0]
    for tag in tags:
        log_init = model.get_log_initial(tag)
        log_emit = model.get_log_emission(tag, word0)
        viterbi[0][tag] = log_init + log_emit
        backpointer[0][tag] = None

    # --- Recursion (t = 1 … n-1) ---
    for t in range(1, n):
        word_t = words[t]
        for tag_j in tags:
            log_emit = model.get_log_emission(tag_j, word_t)

            best_score = -math.inf
            best_prev = None
            for tag_i in tags:
                prev_score = viterbi[t - 1][tag_i]
                if prev_score == -math.inf:
                    continue
                log_trans = model.get_log_transition(tag_i, tag_j)
                score = prev_score + log_trans + log_emit
                if score > best_score:
                    best_score = score
                    best_prev = tag_i

            viterbi[t][tag_j] = best_score
            backpointer[t][tag_j] = best_prev

    # --- Termination ---
    best_last_tag = max(viterbi[n - 1], key=lambda t: viterbi[n - 1][t])

    # --- Backtrace ---
    path = [None] * n
    path[n - 1] = best_last_tag
    for t in range(n - 2, -1, -1):
        bp = backpointer[t + 1][path[t + 1]]
        # Fall back to the highest-scoring tag at step t if backpointer is None
        path[t] = bp if bp is not None else max(viterbi[t], key=lambda tag: viterbi[t][tag])

    return path


def viterbi_decode_corpus(model: HMMModel, sentences: list) -> list:
    """
    Decode every sentence in a corpus.

    Parameters
    ----------
    model : HMMModel
        A trained HMM model.
    sentences : list of list of str
        Each inner list is a tokenised sentence.

    Returns
    -------
    list of list of str
        Predicted tag sequences.
    """
    return [viterbi_decode(model, sent) for sent in sentences]
