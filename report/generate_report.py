"""
generate_report.py – Produce a PDF report for the HMM Viterbi POS Tagger.

Uses ReportLab to build a self-contained PDF that includes:
  - Executive summary
  - Project workflow description
  - Dataset statistics
  - Model architecture overview
  - Evaluation results (accuracy, precision, recall, F1)
  - Embedded analysis figures
  - Conclusions and discussion
"""

import os
import math
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_img(path: str, width: float, height: float):
    """Return an Image flowable if the file exists, otherwise a placeholder."""
    if os.path.isfile(path):
        return Image(path, width=width, height=height)
    return Paragraph(f"[Figure not found: {path}]", getSampleStyleSheet()["Normal"])


def _fmt(v: float) -> str:
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_pdf_report(model, results: dict, figures_dir: str, output_path: str):
    """
    Build and save the PDF report.

    Parameters
    ----------
    model : HMMModel
        Trained HMM model.
    results : dict
        Evaluation results from evaluate_model().
    figures_dir : str
        Directory containing the saved PNG figures.
    output_path : str
        Destination path for the PDF (e.g. report/report.pdf).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"\nGenerating PDF report → {output_path}")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()
    W = A4[0] - 5 * cm   # usable page width

    # Custom paragraph styles
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1A237E"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=13,
        textColor=colors.HexColor("#3949AB"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=15,
        textColor=colors.HexColor("#1A237E"),
        spaceBefore=16,
        spaceAfter=6,
        borderPad=2,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#283593"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    code_style = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontSize=8,
        leading=11,
        leftIndent=20,
        backColor=colors.HexColor("#F5F5F5"),
        borderColor=colors.HexColor("#CCCCCC"),
        borderWidth=0.5,
        borderPad=4,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=10,
    )

    story = []

    # -----------------------------------------------------------------------
    # Title page
    # -----------------------------------------------------------------------
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("HMM Viterbi POS Tagger", title_style))
    story.append(Paragraph("Implementation, Analysis and Evaluation Report", subtitle_style))
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3949AB")))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            caption_style,
        )
    )
    story.append(Spacer(1, 1.5 * cm))

    # -----------------------------------------------------------------------
    # 1. Executive Summary
    # -----------------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(
        Paragraph(
            "This report documents the design, implementation, and evaluation of a "
            "Hidden Markov Model (HMM) based Part-of-Speech (POS) tagger that uses the "
            "Viterbi algorithm for decoding.  The system is trained on a labelled English "
            "corpus using maximum-likelihood estimation with Laplace smoothing and is "
            "evaluated on a held-out test set.  Token-level accuracy and per-tag "
            "precision, recall, and F1 scores are reported together with visual analyses "
            "of the learned model parameters.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            f"<b>Overall test-set accuracy: "
            f"{results['overall_accuracy'] * 100:.2f}% "
            f"({results['correct_tokens']} / {results['total_tokens']} tokens "
            f"correctly tagged).</b>",
            body_style,
        )
    )

    # -----------------------------------------------------------------------
    # 2. Project Workflow
    # -----------------------------------------------------------------------
    story.append(Paragraph("2. Project Workflow", h1_style))
    story.append(
        Paragraph(
            "The pipeline consists of four stages, each implemented as a separate Python "
            "module inside the <code>src/</code> directory:",
            body_style,
        )
    )

    workflow_data = [
        ["Stage", "Module", "Description"],
        ["1 – Data Loading & Training",
         "src/train.py",
         "Parse the word/TAG corpus, count transitions and emissions, "
         "estimate smoothed log-probabilities."],
        ["2 – Decoding",
         "src/viterbi.py",
         "Run the Viterbi algorithm on each test sentence to find the "
         "most-probable tag sequence."],
        ["3 – Evaluation",
         "src/evaluate.py",
         "Compute token accuracy, per-tag precision/recall/F1, "
         "and build a confusion matrix."],
        ["4 – Visualisation & Report",
         "src/visualize.py\nreport/generate_report.py",
         "Generate analysis figures (heat-maps, bar charts) and "
         "compile this PDF report."],
    ]
    wt = Table(
        workflow_data,
        colWidths=[4.2 * cm, 4 * cm, W - 8.4 * cm],
    )
    wt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A237E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#EEF2FF"), colors.white]),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C5CAE9")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(wt)
    story.append(Spacer(1, 0.4 * cm))

    # -----------------------------------------------------------------------
    # 3. Dataset
    # -----------------------------------------------------------------------
    story.append(Paragraph("3. Dataset", h1_style))
    story.append(
        Paragraph(
            "The corpus uses Penn Treebank-style POS tags stored in plain-text files "
            "(one sentence per line, <code>word/TAG</code> format).  A dedicated training "
            "set and a separate held-out test set are used to prevent data leakage.",
            body_style,
        )
    )

    story.append(Paragraph("3.1 Tag Set", h2_style))
    story.append(
        Paragraph(
            "The following Penn Treebank tags appear in the corpus:",
            body_style,
        )
    )
    tag_desc = {
        "CC": "Coordinating conjunction (and, but, or)",
        "CD": "Cardinal number (one, two, three)",
        "DT": "Determiner (the, a, an, every)",
        "IN": "Preposition / subordinating conjunction (in, of, for)",
        "JJ": "Adjective (big, new, good)",
        "JJR": "Adjective, comparative (bigger, better)",
        "NN": "Noun, singular (dog, car, city)",
        "NNP": "Proper noun, singular (John, Paris, Sunday)",
        "NNS": "Noun, plural (dogs, cars, cities)",
        "PRP": "Personal pronoun (he, she, they, I)",
        "PRP$": "Possessive pronoun (his, her, their, its)",
        "RB": "Adverb (quickly, very, slowly)",
        "RP": "Particle (up, out, off)",
        "TO": "Infinitival to",
        "VB": "Verb, base form (run, eat, go)",
        "VBD": "Verb, past tense (ran, ate, went)",
        "VBG": "Verb, gerund (running, eating)",
        "VBN": "Verb, past participle (run, eaten)",
        "VBP": "Verb, present non-3rd-person (run, eat)",
        "VBZ": "Verb, present 3rd-person singular (runs, eats)",
        ".": "Sentence-final punctuation",
    }
    tag_rows = [["Tag", "Description"]] + [
        [tag, desc]
        for tag, desc in sorted(tag_desc.items())
        if tag in model.tags
    ]
    tt = Table(tag_rows, colWidths=[2 * cm, W - 2.2 * cm])
    tt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#283593")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#EEF2FF"), colors.white]),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C5CAE9")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(tt)

    # -----------------------------------------------------------------------
    # 4. Model Description
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("4. Model Description", h1_style))
    story.append(Paragraph("4.1 Hidden Markov Model", h2_style))
    story.append(
        Paragraph(
            "A first-order HMM defines a joint probability over observation sequences "
            "<i>w</i><sub>1:n</sub> and hidden state sequences <i>t</i><sub>1:n</sub>:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>P(w<sub>1:n</sub>, t<sub>1:n</sub>) = "
            "P(t<sub>1</sub>|START) × ∏ P(t<sub>i</sub>|t<sub>i-1</sub>) × "
            "∏ P(w<sub>i</sub>|t<sub>i</sub>)</b>",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "The three parameter tables are:",
            body_style,
        )
    )
    param_data = [
        ["Parameter", "Symbol", "Estimation"],
        ["Initial probability",
         "π(t) = P(t | START)",
         "Count(START→t) / Count(START→*)  [+ Laplace smoothing]"],
        ["Transition probability",
         "A(t_i, t_j) = P(t_j | t_i)",
         "Count(t_i→t_j) / Count(t_i→*)   [+ Laplace smoothing]"],
        ["Emission probability",
         "B(t, w) = P(w | t)",
         "Count(t, w) / Count(t)            [+ Laplace smoothing]"],
    ]
    pt = Table(param_data, colWidths=[4 * cm, 4.5 * cm, W - 8.7 * cm])
    pt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#283593")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#EEF2FF"), colors.white]),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C5CAE9")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(pt)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("4.2 Laplace Smoothing", h2_style))
    story.append(
        Paragraph(
            "All probability estimates use additive (Laplace) smoothing with constant "
            f"α = {model.smoothing}.  Smoothing prevents zero-probability assignments for "
            "unseen word–tag or tag–tag pairs, and handles out-of-vocabulary (OOV) words "
            "via a special &lt;UNK&gt; token.",
            body_style,
        )
    )

    story.append(Paragraph("4.3 Viterbi Decoding", h2_style))
    story.append(
        Paragraph(
            "Given a trained HMM, the Viterbi algorithm finds the most-probable tag "
            "sequence for an input sentence in <i>O(n · |T|²)</i> time using dynamic "
            "programming.  All computations are performed in log-space to prevent "
            "floating-point underflow for long sentences.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Recurrence:</b>",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "δ<sub>t</sub>(j) = max<sub>i</sub> [ δ<sub>t-1</sub>(i) + log A(i,j) ] "
            "+ log B(j, w<sub>t</sub>)",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "The algorithm runs three passes: (1) initialisation, (2) recursion, and "
            "(3) back-trace via stored backpointers to recover the optimal path.",
            body_style,
        )
    )

    # -----------------------------------------------------------------------
    # 5. Evaluation Results
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("5. Evaluation Results", h1_style))

    story.append(Paragraph("5.1 Overall Accuracy", h2_style))
    summary_data = [
        ["Metric", "Value"],
        ["Total tokens", str(results["total_tokens"])],
        ["Correctly tagged", str(results["correct_tokens"])],
        ["Token accuracy",
         f"{results['overall_accuracy'] * 100:.2f}%"],
    ]
    st = Table(summary_data, colWidths=[5 * cm, 4 * cm])
    st.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A237E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#EEF2FF"), colors.white]),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C5CAE9")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("FONTNAME", (0, 3), (-1, 3), "Helvetica-Bold"),
                ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#C5CAE9")),
            ]
        )
    )
    story.append(st)

    story.append(Paragraph("5.2 Per-Tag Metrics", h2_style))
    per_tag_rows = [["Tag", "Precision", "Recall", "F1"]]
    for tag in sorted(results["per_tag_precision"]):
        per_tag_rows.append(
            [
                tag,
                _fmt(results["per_tag_precision"][tag]),
                _fmt(results["per_tag_recall"][tag]),
                _fmt(results["per_tag_f1"][tag]),
            ]
        )
    # Macro averages
    n_tags = len(results["per_tag_precision"])
    macro_p = sum(results["per_tag_precision"].values()) / n_tags
    macro_r = sum(results["per_tag_recall"].values()) / n_tags
    macro_f = sum(results["per_tag_f1"].values()) / n_tags
    per_tag_rows.append(
        ["Macro avg.", _fmt(macro_p), _fmt(macro_r), _fmt(macro_f)]
    )

    ptt = Table(
        per_tag_rows,
        colWidths=[2.5 * cm, 3 * cm, 3 * cm, 3 * cm],
    )
    ptt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#283593")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -2),
                 [colors.HexColor("#EEF2FF"), colors.white]),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C5CAE9")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#C5CAE9")),
            ]
        )
    )
    story.append(ptt)

    # -----------------------------------------------------------------------
    # 6. Analysis Figures
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("6. Analysis and Visualisation", h1_style))

    fig_w = W
    fig_h = 10 * cm

    figures_info = [
        (
            "confusion_matrix.png",
            "Figure 1: Confusion Matrix (row-normalised). "
            "Each cell shows the fraction of gold-tag tokens (row) predicted as "
            "each tag (column).  Diagonal values represent per-tag recall.",
        ),
        (
            "per_tag_f1.png",
            "Figure 2: Per-Tag F1 Score. "
            "Bars are coloured from red (low) to green (high). "
            "The dashed red line marks overall token accuracy.",
        ),
        (
            "precision_recall_f1.png",
            "Figure 3: Precision, Recall and F1 per POS Tag. "
            "Grouped bars allow a direct comparison of all three metrics.",
        ),
        (
            "transition_matrix.png",
            "Figure 4: HMM Transition Probability Matrix. "
            "Cell (i, j) shows the smoothed probability of transitioning from "
            "tag i to tag j.",
        ),
        (
            "top_emissions.png",
            "Figure 5: Top Emission Probabilities per Tag. "
            "Each sub-plot shows the most likely words for a given POS tag, "
            "reflecting the model's learned word-class associations.",
        ),
    ]

    for fname, caption in figures_info:
        path = os.path.join(figures_dir, fname)
        story.append(
            KeepTogether(
                [
                    _safe_img(path, fig_w, fig_h),
                    Paragraph(caption, caption_style),
                ]
            )
        )

    # -----------------------------------------------------------------------
    # 7. Discussion
    # -----------------------------------------------------------------------
    story.append(Paragraph("7. Discussion", h1_style))
    story.append(
        Paragraph(
            "The confusion matrix (Figure 1) reveals where the tagger makes most errors. "
            "Common confusions occur between tags with overlapping lexical evidence, "
            "for example between <i>NN</i> (singular noun) and <i>NNP</i> (proper noun), "
            "or between <i>VBD</i> (past-tense verb) and <i>VBN</i> (past-participle). "
            "These ambiguities are inherent to POS tagging and can only be resolved with "
            "longer-range context that a first-order HMM cannot fully exploit.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "The transition matrix (Figure 4) shows linguistically sensible patterns: "
            "determiners (DT) are most often followed by adjectives (JJ) or nouns (NN), "
            "prepositions (IN) are followed by determiners (DT) or nouns, and verbs (VBD, "
            "VBZ) are preceded by pronouns (PRP) or nouns.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "The emission probabilities (Figure 5) show that each tag has "
            "characteristic vocabulary: <i>DT</i> is dominated by 'the', 'a', 'an'; "
            "<i>PRP</i> by 'he', 'she', 'they'; and <i>VBD</i> by common past-tense "
            "verbs such as 'ran', 'said', 'went'.",
            body_style,
        )
    )

    # -----------------------------------------------------------------------
    # 8. Conclusions
    # -----------------------------------------------------------------------
    story.append(Paragraph("8. Conclusions", h1_style))
    story.append(
        Paragraph(
            "This project demonstrated a complete end-to-end pipeline for HMM-based "
            "POS tagging using the Viterbi algorithm.  Key observations:",
            body_style,
        )
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=20,
        spaceAfter=3,
    )
    bullets = [
        "Laplace smoothing is essential for handling unseen word–tag combinations "
        "and out-of-vocabulary words.",
        "The Viterbi decoder reliably finds the globally optimal tag sequence in "
        "polynomial time.",
        "A first-order HMM captures local syntactic regularities effectively but "
        "is limited by its Markov assumption to a single preceding tag.",
        "Higher accuracy can be achieved with higher-order HMMs, discriminative "
        "models (e.g., CRF, BiLSTM-CRF), or pre-trained language models "
        "(e.g., BERT), at the cost of additional complexity.",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", bullet_style))

    story.append(Spacer(1, 0.5 * cm))
    story.append(
        Paragraph(
            "<b>Complementary suggestions (items not explicitly requested but "
            "recommended for a production system):</b>",
            body_style,
        )
    )
    complements = [
        "Cross-validation to obtain robust accuracy estimates.",
        "Unknown-word morphological features (suffixes, capitalisation) to improve "
        "OOV handling.",
        "Beam search as a faster approximate alternative to exact Viterbi for "
        "large tag sets.",
        "A web-based demo interface using Flask or Gradio.",
        "Integration with a standard benchmark (Penn Treebank WSJ sections 02–21 "
        "for training, section 23 for testing).",
    ]
    for c in complements:
        story.append(Paragraph(f"• {c}", bullet_style))

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------
    doc.build(story)
    print(f"PDF report saved to: {output_path}")
