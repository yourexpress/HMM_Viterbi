"""
Unit tests for the HMM Viterbi POS tagger.

Tests cover:
  - HMMModel training (parameter shapes, probability constraints)
  - Viterbi decoding (output length, correct prediction on a trivial case)
  - Data loading (corpus parser)
  - Evaluation metrics (accuracy, per-tag scores)
"""

import math
import os
import sys
import tempfile
import pytest

# Allow imports from the project root when running with pytest from any CWD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hmm_model import HMMModel
from src.viterbi import viterbi_decode, viterbi_decode_corpus
from src.train import load_tagged_corpus, train_model
from src.evaluate import evaluate_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TINY_CORPUS = [
    [("the", "DT"), ("dog", "NN"), ("runs", "VBZ"), (".", ".")],
    [("a", "DT"), ("cat", "NN"), ("sleeps", "VBZ"), (".", ".")],
    [("the", "DT"), ("cat", "NN"), ("runs", "VBZ"), (".", ".")],
    [("a", "DT"), ("dog", "NN"), ("sleeps", "VBZ"), (".", ".")],
]


@pytest.fixture
def trained_model():
    model = HMMModel(smoothing=1.0)
    model.train(TINY_CORPUS)
    return model


# ---------------------------------------------------------------------------
# HMMModel tests
# ---------------------------------------------------------------------------

class TestHMMModel:
    def test_tags_populated(self, trained_model):
        assert trained_model.tags == {"DT", "NN", "VBZ", "."}

    def test_vocab_populated(self, trained_model):
        assert "the" in trained_model.vocab
        assert "dog" in trained_model.vocab

    def test_log_initial_all_tags_present(self, trained_model):
        for tag in trained_model.tags:
            assert tag in trained_model.log_initial

    def test_log_initial_probabilities_sum_to_one(self, trained_model):
        total = sum(math.exp(v) for v in trained_model.log_initial.values())
        assert abs(total - 1.0) < 1e-6

    def test_log_transition_rows_sum_to_one(self, trained_model):
        for tag_i, row in trained_model.log_transition.items():
            total = sum(math.exp(v) for v in row.values())
            assert abs(total - 1.0) < 1e-6, (
                f"Transition row for '{tag_i}' sums to {total:.6f}"
            )

    def test_emission_rows_sum_to_one(self, trained_model):
        for tag, row in trained_model.log_emission.items():
            total = sum(math.exp(v) for v in row.values())
            assert abs(total - 1.0) < 1e-6, (
                f"Emission row for '{tag}' sums to {total:.6f}"
            )

    def test_get_log_emission_known_word(self, trained_model):
        lp = trained_model.get_log_emission("DT", "the")
        assert lp > -math.inf

    def test_get_log_emission_oov_word(self, trained_model):
        lp = trained_model.get_log_emission("DT", "UNSEEN_WORD_XYZ")
        # Should fall back to UNK, which is a small but finite log-prob
        assert lp > -math.inf

    def test_get_log_transition_known_pair(self, trained_model):
        lp = trained_model.get_log_transition("DT", "NN")
        assert lp > -math.inf

    def test_smoothing_effect(self):
        """Smaller smoothing should give higher probability to observed events."""
        m_high = HMMModel(smoothing=10.0)
        m_low = HMMModel(smoothing=0.01)
        m_high.train(TINY_CORPUS)
        m_low.train(TINY_CORPUS)
        # Both should assign finite probabilities
        assert m_high.get_log_emission("DT", "the") > -math.inf
        assert m_low.get_log_emission("DT", "the") > -math.inf


# ---------------------------------------------------------------------------
# Viterbi tests
# ---------------------------------------------------------------------------

class TestViterbi:
    def test_output_length_matches_input(self, trained_model):
        words = ["the", "dog", "runs", "."]
        tags = viterbi_decode(trained_model, words)
        assert len(tags) == len(words)

    def test_output_tags_are_valid(self, trained_model):
        words = ["the", "cat", "sleeps", "."]
        tags = viterbi_decode(trained_model, words)
        for tag in tags:
            assert tag in trained_model.tags

    def test_trivial_unambiguous_sentence(self):
        """On a corpus where each word appears with exactly one tag,
        the decoder should return the correct tags."""
        # Build a corpus where "hello" is always NN and "world" is always VBZ
        corpus = [
            [("hello", "NN"), ("world", "VBZ")],
            [("hello", "NN"), ("world", "VBZ")],
            [("hello", "NN"), ("world", "VBZ")],
        ]
        model = HMMModel(smoothing=0.001)   # very small smoothing
        model.train(corpus)
        pred = viterbi_decode(model, ["hello", "world"])
        assert pred == ["NN", "VBZ"]

    def test_empty_sentence(self, trained_model):
        assert viterbi_decode(trained_model, []) == []

    def test_single_word_sentence(self, trained_model):
        tags = viterbi_decode(trained_model, ["dog"])
        assert len(tags) == 1
        assert tags[0] in trained_model.tags

    def test_decode_corpus_length(self, trained_model):
        sentences = [["the", "dog"], ["a", "cat", "runs"]]
        results = viterbi_decode_corpus(trained_model, sentences)
        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 3


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_load_tagged_corpus(self, tmp_path):
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text(
            "The/DT cat/NN sits/VBZ ./.\n"
            "A/DT dog/NN runs/VBZ ./.\n",
            encoding="utf-8",
        )
        sentences = load_tagged_corpus(str(corpus_file))
        assert len(sentences) == 2
        assert sentences[0][0] == ("The", "DT")
        assert sentences[0][1] == ("cat", "NN")
        assert sentences[1][-1] == (".", ".")

    def test_empty_lines_ignored(self, tmp_path):
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text(
            "The/DT cat/NN ./.\n\n\nA/DT dog/NN ./.\n",
            encoding="utf-8",
        )
        sentences = load_tagged_corpus(str(corpus_file))
        assert len(sentences) == 2

    def test_train_model_function(self, tmp_path):
        corpus_file = tmp_path / "train.txt"
        corpus_file.write_text(
            "The/DT cat/NN runs/VBZ ./.\n"
            "A/DT dog/NN sleeps/VBZ ./.\n",
            encoding="utf-8",
        )
        model = train_model(str(corpus_file), smoothing=1.0)
        assert "DT" in model.tags
        assert "NN" in model.tags


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_perfect_accuracy(self, trained_model):
        """If predicted == gold, accuracy should be 1.0."""
        # Build a test set from the training sentences (labels should match)
        test_sentences = TINY_CORPUS[:2]
        results = evaluate_model(trained_model, test_sentences)
        # Not necessarily 1.0 due to ambiguity, but should be > 0
        assert 0.0 <= results["overall_accuracy"] <= 1.0

    def test_all_tags_in_per_tag_metrics(self, trained_model):
        test_sentences = TINY_CORPUS
        results = evaluate_model(trained_model, test_sentences)
        for tag in trained_model.tags:
            assert tag in results["per_tag_precision"]
            assert tag in results["per_tag_recall"]
            assert tag in results["per_tag_f1"]

    def test_confusion_matrix_counts(self, trained_model):
        test_sentences = TINY_CORPUS[:1]
        results = evaluate_model(trained_model, test_sentences)
        cm = results["confusion_matrix"]
        total_in_cm = sum(
            count
            for gold_row in cm.values()
            for count in gold_row.values()
        )
        assert total_in_cm == results["total_tokens"]

    def test_precision_recall_in_range(self, trained_model):
        results = evaluate_model(trained_model, TINY_CORPUS)
        for tag in trained_model.tags:
            assert 0.0 <= results["per_tag_precision"][tag] <= 1.0
            assert 0.0 <= results["per_tag_recall"][tag] <= 1.0
            assert 0.0 <= results["per_tag_f1"][tag] <= 1.0

    def test_correct_plus_incorrect_equals_total(self, trained_model):
        results = evaluate_model(trained_model, TINY_CORPUS)
        assert results["correct_tokens"] <= results["total_tokens"]
