"""
Evaluation utilities for the HMM POS tagger.

Computes accuracy, per-tag precision/recall/F1, and builds a
confusion matrix.
"""

from collections import defaultdict
from src.viterbi import viterbi_decode
from src.train import load_tagged_corpus


# ------------------------------------------------------------------
# Core metrics
# ------------------------------------------------------------------

def evaluate_model(model, test_sentences: list) -> dict:
    """
    Evaluate a trained HMM model on a list of tagged test sentences.

    Parameters
    ----------
    model : HMMModel
        Trained model.
    test_sentences : list of list of (word, tag) tuples
        Gold-standard tagged sentences.

    Returns
    -------
    dict with keys:
        overall_accuracy  – token-level accuracy (float)
        per_tag_accuracy  – dict {tag: accuracy}
        per_tag_precision – dict {tag: precision}
        per_tag_recall    – dict {tag: recall}
        per_tag_f1        – dict {tag: F1}
        confusion_matrix  – dict {gold_tag: {pred_tag: count}}
        total_tokens      – int
        correct_tokens    – int
        predicted_tags    – list of lists (aligned with test_sentences)
        gold_tags         – list of lists (aligned with test_sentences)
    """
    all_gold = []
    all_pred = []
    confusion: dict = defaultdict(lambda: defaultdict(int))

    for sentence in test_sentences:
        if not sentence:
            continue
        words = [w for w, _ in sentence]
        gold_tags = [t for _, t in sentence]
        pred_tags = viterbi_decode(model, words)

        all_gold.append(gold_tags)
        all_pred.append(pred_tags)

        for gold, pred in zip(gold_tags, pred_tags):
            confusion[gold][pred] += 1

    # Flatten for token-level statistics
    flat_gold = [t for seq in all_gold for t in seq]
    flat_pred = [t for seq in all_pred for t in seq]

    total = len(flat_gold)
    correct = sum(g == p for g, p in zip(flat_gold, flat_pred))
    overall_accuracy = correct / total if total > 0 else 0.0

    # Per-tag metrics
    tags = sorted(set(flat_gold) | set(flat_pred))
    per_tag_accuracy = {}
    per_tag_precision = {}
    per_tag_recall = {}
    per_tag_f1 = {}

    for tag in tags:
        tp = sum(g == tag and p == tag for g, p in zip(flat_gold, flat_pred))
        fp = sum(g != tag and p == tag for g, p in zip(flat_gold, flat_pred))
        fn = sum(g == tag and p != tag for g, p in zip(flat_gold, flat_pred))
        tn = sum(g != tag and p != tag for g, p in zip(flat_gold, flat_pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / total if total > 0 else 0.0

        per_tag_accuracy[tag] = accuracy
        per_tag_precision[tag] = precision
        per_tag_recall[tag] = recall
        per_tag_f1[tag] = f1

    return {
        "overall_accuracy": overall_accuracy,
        "per_tag_accuracy": per_tag_accuracy,
        "per_tag_precision": per_tag_precision,
        "per_tag_recall": per_tag_recall,
        "per_tag_f1": per_tag_f1,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "total_tokens": total,
        "correct_tokens": correct,
        "predicted_tags": all_pred,
        "gold_tags": all_gold,
    }


def print_evaluation_report(results: dict):
    """Pretty-print evaluation results to stdout."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total tokens : {results['total_tokens']}")
    print(f"  Correct      : {results['correct_tokens']}")
    print(f"  Accuracy     : {results['overall_accuracy']:.4f} "
          f"({results['overall_accuracy'] * 100:.2f}%)")
    print()
    print(f"{'Tag':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    for tag in sorted(results["per_tag_precision"]):
        p = results["per_tag_precision"][tag]
        r = results["per_tag_recall"][tag]
        f = results["per_tag_f1"][tag]
        print(f"  {tag:<10} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
    print("=" * 60)


def evaluate_from_files(model, test_path: str) -> dict:
    """
    Convenience wrapper: load test data and evaluate.

    Parameters
    ----------
    model : HMMModel
        Trained model.
    test_path : str
        Path to the test corpus.

    Returns
    -------
    dict
        Same structure as returned by :func:`evaluate_model`.
    """
    print(f"\nLoading test data from: {test_path}")
    test_sentences = load_tagged_corpus(test_path)
    print(f"  Loaded {len(test_sentences)} test sentences.")
    results = evaluate_model(model, test_sentences)
    print_evaluation_report(results)
    return results
