"""
HMM Model for Part-of-Speech Tagging.

This module implements a Hidden Markov Model trained with
maximum-likelihood estimation and Laplace (add-one) smoothing.
"""

from collections import defaultdict
import math


class HMMModel:
    """
    First-order Hidden Markov Model for sequence labelling (POS tagging).

    Parameters
    ----------
    smoothing : float
        Additive smoothing constant (default 1.0 for Laplace smoothing).
    """

    # Tag used for the start-of-sentence boundary
    START_TAG = "<START>"
    # Tag used for the end-of-sentence boundary
    END_TAG = "<END>"

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing

        # Vocabulary: set of observed words
        self.vocab: set = set()
        # Set of POS tags (excluding START / END)
        self.tags: set = set()

        # Raw counts collected during training
        self._tag_counts: dict = defaultdict(int)
        self._transition_counts: dict = defaultdict(lambda: defaultdict(int))
        self._emission_counts: dict = defaultdict(lambda: defaultdict(int))

        # Log-probability tables (populated after training)
        self.log_initial: dict = {}       # log P(tag | START)
        self.log_transition: dict = {}    # log P(tag_j | tag_i)
        self.log_emission: dict = {}      # log P(word | tag)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, sentences: list):
        """
        Estimate HMM parameters from a list of tagged sentences.

        Parameters
        ----------
        sentences : list of list of (word, tag) tuples
        """
        for sentence in sentences:
            if not sentence:
                continue

            # Initial / transition counts
            prev_tag = self.START_TAG
            for word, tag in sentence:
                self.vocab.add(word.lower())
                self.tags.add(tag)
                self._tag_counts[tag] += 1
                self._transition_counts[prev_tag][tag] += 1
                self._emission_counts[tag][word.lower()] += 1
                prev_tag = tag

            # (END-state transitions are not used by the Viterbi decoder)

        self._compute_log_probabilities()

    def _compute_log_probabilities(self):
        """Convert raw counts to smoothed log-probabilities."""
        tags = list(self.tags)
        n_tags = len(tags)
        vocab_size = len(self.vocab)
        s = self.smoothing

        # --- Initial probabilities: P(tag | START) ---
        start_total = sum(self._transition_counts[self.START_TAG].values())
        self.log_initial = {
            tag: math.log(
                (self._transition_counts[self.START_TAG][tag] + s)
                / (start_total + s * n_tags)
            )
            for tag in tags
        }

        # --- Transition probabilities: P(next_tag | cur_tag) ---
        self.log_transition = {}
        for tag_i in tags:
            total = sum(self._transition_counts[tag_i].values())
            self.log_transition[tag_i] = {
                tag_j: math.log(
                    (self._transition_counts[tag_i][tag_j] + s)
                    / (total + s * n_tags)
                )
                for tag_j in tags
            }

        # --- Emission probabilities: P(word | tag) ---
        # Denominator includes one extra slot for <UNK> so that all
        # vocab words + UNK together sum to exactly 1.
        self.log_emission = {}
        for tag in tags:
            total = sum(self._emission_counts[tag].values())
            # vocab_size + 1 → the +1 accounts for the <UNK> pseudo-word
            denom = total + s * (vocab_size + 1)
            self.log_emission[tag] = {
                word: math.log((self._emission_counts[tag][word] + s) / denom)
                for word in self.vocab
            }
            # Log-probability for unknown words (UNK)
            self.log_emission[tag]["<UNK>"] = math.log(s / denom)

    # ------------------------------------------------------------------
    # Probability accessors (used by the Viterbi decoder)
    # ------------------------------------------------------------------

    def get_log_initial(self, tag: str) -> float:
        """Return log P(tag | START)."""
        return self.log_initial.get(tag, -math.inf)

    def get_log_transition(self, tag_i: str, tag_j: str) -> float:
        """Return log P(tag_j | tag_i)."""
        return self.log_transition.get(tag_i, {}).get(tag_j, -math.inf)

    def get_log_emission(self, tag: str, word: str) -> float:
        """Return log P(word | tag), back-off to UNK for OOV words."""
        word_lower = word.lower()
        emission = self.log_emission.get(tag, {})
        if word_lower in emission:
            return emission[word_lower]
        return emission.get("<UNK>", -math.inf)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def get_raw_transition_probs(self) -> dict:
        """Return a nested dict of smoothed (linear) transition probabilities."""
        import math as _math
        return {
            t_i: {t_j: _math.exp(v) for t_j, v in inner.items()}
            for t_i, inner in self.log_transition.items()
        }

    def get_raw_emission_probs(self) -> dict:
        """Return a nested dict of smoothed (linear) emission probabilities."""
        import math as _math
        return {
            tag: {w: _math.exp(v) for w, v in inner.items()}
            for tag, inner in self.log_emission.items()
        }
