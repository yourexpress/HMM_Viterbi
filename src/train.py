"""
Data loading and training utilities.

Reads POS-tagged corpora in the standard word/TAG format and trains the
HMM model.
"""

import os
from src.hmm_model import HMMModel


def load_tagged_corpus(filepath: str) -> list:
    """
    Parse a POS-tagged text file.

    Expected format: one sentence per line, tokens separated by spaces,
    each token in the form ``word/TAG``.

    Parameters
    ----------
    filepath : str
        Path to the corpus file.

    Returns
    -------
    list of list of (word, tag) tuples
        Each inner list represents one sentence.
    """
    sentences = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            sentence = []
            for token in tokens:
                # Handle tokens like "word/TAG" – take the last "/" as separator
                sep = token.rfind("/")
                if sep == -1:
                    continue
                word = token[:sep]
                tag = token[sep + 1:]
                if word and tag:
                    sentence.append((word, tag))
            if sentence:
                sentences.append(sentence)
    return sentences


def train_model(train_path: str, smoothing: float = 1.0) -> HMMModel:
    """
    Load training data and fit an HMM model.

    Parameters
    ----------
    train_path : str
        Path to the training corpus.
    smoothing : float
        Laplace smoothing constant.

    Returns
    -------
    HMMModel
        Trained model.
    """
    print(f"Loading training data from: {train_path}")
    sentences = load_tagged_corpus(train_path)
    print(f"  Loaded {len(sentences)} training sentences.")

    model = HMMModel(smoothing=smoothing)
    model.train(sentences)

    print(f"  Vocabulary size : {len(model.vocab)}")
    print(f"  Number of tags  : {len(model.tags)}")
    print(f"  Tags            : {sorted(model.tags)}")
    return model
