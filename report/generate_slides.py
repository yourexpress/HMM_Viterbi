"""
generate_slides.py – Build a landscape PDF presentation for the HMM Viterbi POS Tagger.

Slide outline
-------------
 1.  Title
 2.  What is POS Tagging?
 3.  Our Approach – Hidden Markov Model
 4.  HMM Parameters & Smoothing
 5.  Viterbi Decoding Algorithm
 6.  Project Pipeline (4 stages)
 7.  Dataset Overview
 8.  Results – Overall Accuracy (headline number)
 9.  Results – Per-Tag Metrics Table
10.  Figure – Per-Tag F1 Bar Chart
11.  Figure – Precision / Recall / F1 Grouped Bar Chart
12.  Figure – Confusion Matrix Heat-map
13.  Figure – Transition Probability Matrix
14.  Figure – Top Emission Probabilities per Tag
15.  Key Findings & Discussion
16.  Conclusions & Future Work
"""

import os
import math
from datetime import datetime

from PIL import Image as PILImage
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    FrameBreak,
    NextPageTemplate,
)


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = landscape(A4)          # 841.9 × 595.3 pt
MARGIN = 1.4 * cm

# Slide chrome heights
HEADER_H = 1.7 * cm    # coloured title band
FOOTER_H = 0.7 * cm    # slide-number band

# Usable content area (inside margins, below header, above footer)
CONTENT_W = PAGE_W - 2 * MARGIN
CONTENT_H = PAGE_H - MARGIN - HEADER_H - FOOTER_H - 0.3 * cm  # small gap below header

# Brand colours
C_DARK  = colors.HexColor("#1A237E")   # deep indigo
C_MID   = colors.HexColor("#3949AB")   # medium indigo
C_LIGHT = colors.HexColor("#E8EAF6")   # pale lavender fill
C_ACCENT= colors.HexColor("#FFA000")   # amber accent
C_GREY  = colors.HexColor("#546E7A")   # blue-grey text
C_WHITE = colors.white
C_GREEN = colors.HexColor("#388E3C")
C_RED   = colors.HexColor("#C62828")


# ---------------------------------------------------------------------------
# Paragraph styles (used throughout every slide)
# ---------------------------------------------------------------------------

def _make_styles():
    slide_title = ParagraphStyle(
        "SlideTitle",
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=C_WHITE,
        leading=24,
        alignment=TA_LEFT,
    )
    body = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=11,
        textColor=colors.HexColor("#212121"),
        leading=17,
        spaceAfter=6,
        alignment=TA_LEFT,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=18,
        spaceAfter=5,
    )
    sub_bullet = ParagraphStyle(
        "SubBullet",
        parent=body,
        fontSize=10,
        leftIndent=38,
        spaceAfter=3,
    )
    code = ParagraphStyle(
        "Code",
        fontName="Courier",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#37474F"),
        backColor=colors.HexColor("#F5F5F5"),
        leftIndent=16,
        spaceAfter=6,
    )
    caption = ParagraphStyle(
        "Caption",
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=C_GREY,
        alignment=TA_CENTER,
    )
    big_number = ParagraphStyle(
        "BigNumber",
        fontName="Helvetica-Bold",
        fontSize=54,
        textColor=C_DARK,
        alignment=TA_CENTER,
    )
    big_label = ParagraphStyle(
        "BigLabel",
        fontName="Helvetica",
        fontSize=14,
        textColor=C_GREY,
        alignment=TA_CENTER,
    )
    formula = ParagraphStyle(
        "Formula",
        fontName="Courier-Bold",
        fontSize=10,
        textColor=C_DARK,
        leading=15,
        leftIndent=20,
        spaceAfter=8,
        backColor=colors.HexColor("#EEF2FF"),
    )
    section_head = ParagraphStyle(
        "SectionHead",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=C_MID,
        spaceBefore=8,
        spaceAfter=4,
    )
    return dict(
        slide_title=slide_title,
        body=body,
        bullet=bullet,
        sub_bullet=sub_bullet,
        code=code,
        caption=caption,
        big_number=big_number,
        big_label=big_label,
        formula=formula,
        section_head=section_head,
    )


# ---------------------------------------------------------------------------
# Canvas-level chrome (header band, footer, divider line)
# ---------------------------------------------------------------------------

class _SlideCanvas:
    """Mixin that draws the slide header and footer on every page."""

    def __init__(self, title: str, total_slides: int, slide_number: int):
        self.slide_title_text = title
        self.total_slides = total_slides
        self.slide_number = slide_number

    def __call__(self, canv, doc):
        canv.saveState()

        # ---- Header band ----
        canv.setFillColor(C_DARK)
        canv.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)

        canv.setFont("Helvetica-Bold", 14)
        canv.setFillColor(C_WHITE)
        canv.drawString(MARGIN, PAGE_H - HEADER_H + 0.45 * cm, self.slide_title_text)

        # Project name tag on the right
        canv.setFont("Helvetica", 10)
        tag = "HMM Viterbi POS Tagger"
        canv.drawRightString(PAGE_W - MARGIN, PAGE_H - HEADER_H + 0.45 * cm, tag)

        # Thin accent stripe below header
        canv.setFillColor(C_ACCENT)
        canv.rect(0, PAGE_H - HEADER_H - 0.15 * cm, PAGE_W, 0.15 * cm, fill=1, stroke=0)

        # ---- Footer band ----
        canv.setFillColor(C_LIGHT)
        canv.rect(0, 0, PAGE_W, FOOTER_H, fill=1, stroke=0)

        canv.setFont("Helvetica", 8)
        canv.setFillColor(C_GREY)
        canv.drawRightString(
            PAGE_W - MARGIN, 0.18 * cm,
            f"{self.slide_number} / {self.total_slides}",
        )

        canv.restoreState()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_img(path: str, max_w: float, max_h: float) -> Paragraph:
    """Return a size-constrained Image flowable (preserves aspect ratio)."""
    if os.path.isfile(path):
        with PILImage.open(path) as pil_img:
            iw, ih = pil_img.size
        if iw > 0 and ih > 0:
            scale = min(max_w / iw, max_h / ih)
            return Image(path, width=iw * scale, height=ih * scale)
    st = ParagraphStyle("ph", fontName="Helvetica", fontSize=9,
                        textColor=C_GREY, alignment=TA_CENTER)
    return Paragraph(f"[Figure not available: {os.path.basename(path)}]", st)


def _bullet(text: str, s) -> Paragraph:
    return Paragraph(f"\u2022\u00a0 {text}", s["bullet"])


def _sub_bullet(text: str, s) -> Paragraph:
    return Paragraph(f"\u25e6\u00a0 {text}", s["sub_bullet"])


def _body(text: str, s) -> Paragraph:
    return Paragraph(text, s["body"])


def _head(text: str, s) -> Paragraph:
    return Paragraph(text, s["section_head"])


def _spacer(h_cm: float = 0.3):
    return Spacer(1, h_cm * cm)


def _table_style(header_bg=C_MID, alt_bg=C_LIGHT,
                 font_size=9, header_font_size=10):
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  header_font_size),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [alt_bg, C_WHITE]),
        ("FONTSIZE",      (0, 1), (-1, -1), font_size),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#C5CAE9")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ])


# ---------------------------------------------------------------------------
# Individual slide content builders
# ---------------------------------------------------------------------------

def _slide_title_page(s, results) -> list:
    """Slide 1 – Title."""
    acc = results["overall_accuracy"] * 100
    correct = results["correct_tokens"]
    total = results["total_tokens"]
    story = [
        _spacer(0.6),
        Paragraph(
            "<font color='#1A237E' size='32'><b>HMM Viterbi POS Tagger</b></font>",
            ParagraphStyle("tc", fontName="Helvetica-Bold", fontSize=32,
                           textColor=C_DARK, alignment=TA_CENTER),
        ),
        _spacer(0.4),
        Paragraph(
            "Hidden Markov Model + Viterbi Algorithm for Part-of-Speech Tagging",
            ParagraphStyle("tc2", fontName="Helvetica", fontSize=15,
                           textColor=C_MID, alignment=TA_CENTER),
        ),
        _spacer(1.0),
        # Key-stat row
        Table(
            [[
                Paragraph(f"{acc:.1f}%",
                          ParagraphStyle("kv", fontName="Helvetica-Bold",
                                         fontSize=36, textColor=C_DARK,
                                         alignment=TA_CENTER)),
                Paragraph(f"{total}",
                          ParagraphStyle("kv", fontName="Helvetica-Bold",
                                         fontSize=36, textColor=C_DARK,
                                         alignment=TA_CENTER)),
                Paragraph("21",
                          ParagraphStyle("kv", fontName="Helvetica-Bold",
                                         fontSize=36, textColor=C_DARK,
                                         alignment=TA_CENTER)),
            ],
             [
                Paragraph("Test Accuracy",
                          ParagraphStyle("kl", fontName="Helvetica",
                                         fontSize=11, textColor=C_GREY,
                                         alignment=TA_CENTER)),
                Paragraph("Test Tokens",
                          ParagraphStyle("kl", fontName="Helvetica",
                                         fontSize=11, textColor=C_GREY,
                                         alignment=TA_CENTER)),
                Paragraph("POS Tags",
                          ParagraphStyle("kl", fontName="Helvetica",
                                         fontSize=11, textColor=C_GREY,
                                         alignment=TA_CENTER)),
             ]],
            colWidths=[CONTENT_W / 3] * 3,
        ),
        _spacer(1.2),
        Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y')}",
            ParagraphStyle("date", fontName="Helvetica", fontSize=9,
                           textColor=C_GREY, alignment=TA_CENTER),
        ),
    ]
    return story


def _slide_pos_tagging(s) -> list:
    """Slide 2 – What is POS Tagging?"""
    story = [
        _spacer(0.2),
        _body(
            "<b>Part-of-Speech (POS) tagging</b> assigns a grammatical category "
            "(noun, verb, adjective …) to every word in a sentence.", s),
        _spacer(0.3),
        _head("Example", s),
        Table(
            [["Token", "The", "cat", "sat", "on", "the", "mat", "."],
             ["Tag",   "DT",  "NN",  "VBD","IN", "DT",  "NN",  "."]],
            colWidths=[2.4 * cm] + [2.4 * cm] * 7,
            style=TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), C_MID),
                ("TEXTCOLOR",   (0, 0), (-1, 0), C_WHITE),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND",  (0, 1), (-1, 1), C_LIGHT),
                ("FONTNAME",    (0, 1), (-1, 1), "Helvetica-Bold"),
                ("TEXTCOLOR",   (1, 1), (-1, 1), C_DARK),
                ("FONTSIZE",    (0, 0), (-1, -1), 10),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#C5CAE9")),
                ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING",  (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",(0,0), (-1, -1), 5),
            ]),
        ),
        _spacer(0.4),
        _head("Why does it matter?", s),
        _bullet("Foundation for higher-level NLP: parsing, NER, machine translation, QA", s),
        _bullet("Penn Treebank tags used here – 21-tag subset", s),
        _spacer(0.3),
        _head("Challenge", s),
        _bullet("Words are ambiguous: 'run' can be a verb (VB) or noun (NN)", s),
        _bullet("Context determines the correct tag", s),
    ]
    return story


def _slide_hmm_approach(s) -> list:
    """Slide 3 – Our Approach: Hidden Markov Model."""
    story = [
        _spacer(0.2),
        _body(
            "A <b>first-order Hidden Markov Model (HMM)</b> treats POS tags as "
            "hidden states and words as observations.", s),
        _spacer(0.3),
        _head("Two core independence assumptions", s),
        _bullet(
            "<b>Markov assumption:</b> the current tag depends only on the "
            "previous tag, not the full history.", s),
        _sub_bullet("P(t\u1d62 | t\u2081…t\u1d62\u208b\u2081) = P(t\u1d62 | t\u1d62\u208b\u2081)", s),
        _bullet(
            "<b>Output independence:</b> the word depends only on its own tag.", s),
        _sub_bullet("P(w\u1d62 | w\u2081…w\u2099, t\u2081…t\u2099) = P(w\u1d62 | t\u1d62)", s),
        _spacer(0.3),
        _head("Joint probability of a sentence", s),
        Paragraph(
            "P(w\u2081:\u2099, t\u2081:\u2099) = P(t\u2081|START)  \u00d7  "
            "\u220f P(t\u1d62|t\u1d62\u208b\u2081)  \u00d7  \u220f P(w\u1d62|t\u1d62)",
            s["formula"],
        ),
        _spacer(0.3),
        _head("Parameter estimation", s),
        _bullet(
            "Maximum-Likelihood Estimation (MLE) from labelled training data", s),
        _bullet(
            "Laplace (add-\u03b1) smoothing (α = 1.0) to handle unseen events "
            "and out-of-vocabulary words", s),
    ]
    return story


def _slide_hmm_parameters(s) -> list:
    """Slide 4 – HMM Parameters & Smoothing."""
    rows = [
        ["Parameter", "Symbol", "Formula (smoothed)"],
        ["Initial probability",
         "π(t) = P(t | START)",
         "( Count(START→t) + α ) / ( Count(START→*) + α·|T| )"],
        ["Transition probability",
         "A(tᵢ, tⱼ) = P(tⱼ | tᵢ)",
         "( Count(tᵢ→tⱼ) + α ) / ( Count(tᵢ→*) + α·|T| )"],
        ["Emission probability",
         "B(t, w) = P(w | t)",
         "( Count(t,w) + α ) / ( Count(t) + α·(|V|+1) )"],
    ]
    col_w = [3.2 * cm, 5.0 * cm, CONTENT_W - 8.4 * cm]
    param_table = Table(rows, colWidths=col_w)
    param_table.setStyle(_table_style(font_size=9, header_font_size=10))

    story = [
        _spacer(0.15),
        param_table,
        _spacer(0.45),
        _head("Laplace (add-α) smoothing — why it matters", s),
        _bullet("Prevents zero probabilities for unseen word–tag pairs", s),
        _bullet("Out-of-vocabulary words mapped to a special <UNK> token", s),
        _bullet("α = 1.0 adds one pseudo-count to every possible event", s),
        _spacer(0.4),
        _head("Corpus statistics", s),
        Table(
            [["350 training sentences", "827 unique words", "21 POS tags",
              "50 test sentences"],
             ["Manually labelled", "Penn Treebank style", "Including <UNK>",
              "283 test tokens"]],
            colWidths=[CONTENT_W / 4] * 4,
            style=TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0), C_LIGHT),
                ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, -1), 10),
                ("FONTSIZE",     (0, 1), (-1, 1), 8),
                ("TEXTCOLOR",    (0, 0), (-1, 0), C_DARK),
                ("TEXTCOLOR",    (0, 1), (-1, 1), C_GREY),
                ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING",   (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
                ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#C5CAE9")),
            ]),
        ),
    ]
    return story


def _slide_viterbi(s) -> list:
    """Slide 5 – Viterbi Decoding Algorithm."""
    story = [
        _spacer(0.15),
        _body(
            "Given a trained HMM, find the <b>most-probable tag sequence</b> for "
            "an input sentence using dynamic programming.", s),
        _spacer(0.25),
        _head("Recurrence (log-space to avoid underflow)", s),
        Paragraph(
            "\u03b4\u209c(j)  =  max\u1d62 [ \u03b4\u209c\u208b\u2081(i) "
            "+ log A(i,j) ]  +  log B(j, w\u209c)",
            s["formula"],
        ),
        _spacer(0.25),
        _head("Three passes", s),
        _bullet(
            "<b>Initialisation</b> (t = 0): δ₀(j) = log π(j) + log B(j, w₀)", s),
        _bullet(
            "<b>Recursion</b> (t = 1…n–1): fill the Viterbi table and store "
            "backpointers", s),
        _bullet(
            "<b>Backtrace</b>: recover the optimal path by following backpointers "
            "from the best final state", s),
        _spacer(0.35),
        _head("Complexity", s),
        Table(
            [["Time", "Space", "Exact?"],
             ["O(n × |T|²)", "O(n × |T|)", "Yes — globally optimal"]],
            colWidths=[CONTENT_W / 3] * 3,
            style=TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), C_MID),
                ("TEXTCOLOR",   (0, 0), (-1, 0), C_WHITE),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND",  (0, 1), (-1, 1), C_LIGHT),
                ("FONTNAME",    (0, 1), (-1, 1), "Courier-Bold"),
                ("TEXTCOLOR",   (0, 1), (-1, 1), C_DARK),
                ("FONTSIZE",    (0, 0), (-1, -1), 10),
                ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#C5CAE9")),
                ("TOPPADDING",  (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",(0,0), (-1, -1), 5),
            ]),
        ),
    ]
    return story


def _slide_pipeline(s) -> list:
    """Slide 6 – Project Pipeline."""
    rows = [
        ["Stage", "Module", "What it does"],
        ["1 – Train",
         "src/train.py\nsrc/hmm_model.py",
         "Parse word/TAG corpus → count transitions & emissions → "
         "compute smoothed log-probabilities."],
        ["2 – Decode",
         "src/viterbi.py",
         "Run Viterbi on each test sentence to find the most-probable "
         "tag sequence in O(n·|T|²)."],
        ["3 – Evaluate",
         "src/evaluate.py",
         "Token accuracy, per-tag precision / recall / F1, "
         "confusion matrix."],
        ["4 – Visualise & Report",
         "src/visualize.py\nreport/generate_report.py\nreport/generate_slides.py",
         "5 analysis figures (heat-maps, bar charts) + PDF report + "
         "this slide deck."],
    ]
    col_w = [2.8 * cm, 4.2 * cm, CONTENT_W - 7.2 * cm]
    t = Table(rows, colWidths=col_w)
    t.setStyle(_table_style(font_size=9, header_font_size=10))
    story = [
        _spacer(0.2),
        t,
        _spacer(0.5),
        _body(
            "Run the full pipeline with a single command:", s),
        Paragraph("python main.py", s["code"]),
    ]
    return story


def _slide_dataset(s, model) -> list:
    """Slide 7 – Dataset Overview."""
    tag_desc = {
        "CC": "Coordinating conjunction",
        "CD": "Cardinal number",
        "DT": "Determiner",
        "IN": "Preposition / subordinating conj.",
        "JJ": "Adjective",
        "JJR": "Adjective, comparative",
        "NN": "Noun, singular",
        "NNP": "Proper noun, singular",
        "NNS": "Noun, plural",
        "PRP": "Personal pronoun",
        "PRP$": "Possessive pronoun",
        "RB": "Adverb",
        "RP": "Particle",
        "TO": "Infinitival to",
        "VB": "Verb, base form",
        "VBD": "Verb, past tense",
        "VBG": "Verb, gerund",
        "VBN": "Verb, past participle",
        "VBP": "Verb, present non-3rd",
        "VBZ": "Verb, present 3rd-person singular",
        ".":  "Sentence-final punctuation",
    }
    # Split tags into two columns
    tag_list = [(tag, tag_desc.get(tag, "")) for tag in sorted(model.tags)
                if tag in tag_desc]
    half = math.ceil(len(tag_list) / 2)
    col1 = tag_list[:half]
    col2 = tag_list[half:]

    def _tag_rows(col):
        return [["Tag", "Description"]] + [[t, d] for t, d in col]

    left_rows = _tag_rows(col1)
    right_rows = _tag_rows(col2)
    # Pad to equal length
    while len(right_rows) < len(left_rows):
        right_rows.append(["", ""])

    col_w_tag = 1.3 * cm
    col_w_desc = (CONTENT_W / 2 - 0.4 * cm) - col_w_tag
    gap = 0.4 * cm

    left_t = Table(left_rows,
                   colWidths=[col_w_tag, col_w_desc])
    left_t.setStyle(_table_style(font_size=8, header_font_size=9))

    right_t = Table(right_rows,
                    colWidths=[col_w_tag, col_w_desc])
    right_t.setStyle(_table_style(font_size=8, header_font_size=9))

    wrapper = Table([[left_t, Spacer(gap, 1), right_t]],
                    colWidths=[CONTENT_W / 2, gap, CONTENT_W / 2])
    wrapper.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                  ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))

    story = [
        _spacer(0.1),
        _body(
            "Penn Treebank–style word/TAG corpus.  "
            "<b>350 training sentences</b> | <b>50 test sentences</b>  "
            "(no overlap).", s),
        _spacer(0.2),
        wrapper,
    ]
    return story


def _slide_accuracy(s, results) -> list:
    """Slide 8 – Headline Accuracy."""
    acc = results["overall_accuracy"] * 100
    correct = results["correct_tokens"]
    total = results["total_tokens"]

    story = [
        _spacer(0.5),
        Paragraph(f"<b>{acc:.2f}%</b>",
                  ParagraphStyle("big_acc", fontName="Helvetica-Bold",
                                 fontSize=72, textColor=C_DARK,
                                 alignment=TA_CENTER)),
        Paragraph("token-level accuracy on the held-out test set",
                  ParagraphStyle("big_acc_sub", fontName="Helvetica",
                                 fontSize=14, textColor=C_GREY,
                                 alignment=TA_CENTER)),
        _spacer(0.5),
        Table(
            [["Metric", "Value"],
             ["Total tokens", str(total)],
             ["Correctly tagged", str(correct)],
             ["Incorrectly tagged", str(total - correct)]],
            colWidths=[5 * cm, 4 * cm],
            style=TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), C_DARK),
                ("TEXTCOLOR",   (0, 0), (-1, 0), C_WHITE),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT, C_WHITE]),
                ("FONTSIZE",    (0, 0), (-1, -1), 10),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#C5CAE9")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING",  (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",(0,0), (-1, -1), 5),
                ("FONTNAME",    (0, 3), (-1, 3), "Helvetica-Bold"),
                ("TEXTCOLOR",   (1, 3), (-1, 3), C_RED),
            ]),
        ),
        _spacer(0.35),
        Paragraph(
            "Trained from <b>scratch</b> using only counts and Laplace smoothing "
            "— no neural networks, no pre-trained embeddings.",
            ParagraphStyle("note", fontName="Helvetica-Oblique",
                           fontSize=10, textColor=C_GREY,
                           alignment=TA_CENTER),
        ),
    ]
    return story


def _slide_per_tag_table(s, results) -> list:
    """Slide 9 – Per-Tag Metrics Table."""
    def _fmt(v):
        return f"{v:.4f}"

    n_tags = len(results["per_tag_precision"])
    macro_p = sum(results["per_tag_precision"].values()) / n_tags
    macro_r = sum(results["per_tag_recall"].values()) / n_tags
    macro_f = sum(results["per_tag_f1"].values()) / n_tags

    rows = [["Tag", "Precision", "Recall", "F1"]]
    for tag in sorted(results["per_tag_precision"]):
        p = results["per_tag_precision"][tag]
        r = results["per_tag_recall"][tag]
        f = results["per_tag_f1"][tag]
        rows.append([tag, _fmt(p), _fmt(r), _fmt(f)])
    rows.append(["Macro avg.", _fmt(macro_p), _fmt(macro_r), _fmt(macro_f)])

    # Split into two side-by-side tables
    data_rows = rows[1:-1]  # exclude header and macro-avg
    header = rows[0]
    macro = rows[-1]
    half = math.ceil(len(data_rows) / 2)
    left_data = [header] + data_rows[:half]
    right_data = [header] + data_rows[half:] + [macro]

    col_w_single = [1.8 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]
    gap = 0.4 * cm

    def _make_half_table(data, highlight_last=False):
        t = Table(data, colWidths=col_w_single)
        ts = _table_style(font_size=9, header_font_size=9)
        if highlight_last:
            ts.add("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#C5CAE9"))
            ts.add("FONTNAME",   (0, -1), (-1, -1), "Helvetica-Bold")
        t.setStyle(ts)
        return t

    left_t = _make_half_table(left_data)
    right_t = _make_half_table(right_data, highlight_last=True)

    wrapper = Table([[left_t, Spacer(gap, 1), right_t]],
                    colWidths=[sum(col_w_single), gap, sum(col_w_single)])
    wrapper.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                  ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story = [
        _spacer(0.15),
        wrapper,
        _spacer(0.3),
        _body(
            "<b>Note:</b> Tags with 0.0 precision / recall / F1 "
            "(NNP, NNS, VBG, VBZ) do not appear in the test set in sufficient "
            "quantity for reliable estimation.", s),
    ]
    return story


def _slide_figure(path: str, caption: str) -> list:
    """Generic figure slide – image + caption."""
    max_w = CONTENT_W
    max_h = CONTENT_H - 1.2 * cm   # leave room for caption
    img = _safe_img(path, max_w, max_h)
    cap_style = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=9,
                               textColor=C_GREY, alignment=TA_CENTER)
    return [
        _spacer(0.1),
        img,
        _spacer(0.15),
        Paragraph(caption, cap_style),
    ]


def _slide_discussion(s, results) -> list:
    """Slide 15 – Key Findings & Discussion."""
    acc = results["overall_accuracy"] * 100

    # Tags with perfect F1
    perfect = [t for t, v in results["per_tag_f1"].items() if v == 1.0]
    # Tags that struggled
    zero_f1 = [t for t, v in results["per_tag_f1"].items() if v == 0.0]

    story = [
        _spacer(0.2),
        _head("Strengths", s),
        _bullet(
            f"<b>{acc:.2f}%</b> token accuracy — strong baseline from a simple "
            "statistical model with no neural components.", s),
        _bullet(
            f"Perfect F1 = 1.00 for: "
            f"<b>{', '.join(sorted(perfect)) if perfect else '–'}</b>. "
            "These tags have highly distinctive vocabulary.", s),
        _bullet(
            "Log-space Viterbi is numerically stable even for long sentences.", s),
        _spacer(0.25),
        _head("Limitations & error sources", s),
        _bullet(
            "Common confusions: NN ↔ NNP (singular vs. proper noun), "
            "VBD ↔ VBN (past tense vs. past participle). "
            "Both pairs share lexical forms.", s),
        _bullet(
            f"Tags {', '.join(sorted(zero_f1)) if zero_f1 else '–'} "
            "score F1 = 0.00 — rare in the small test set.", s),
        _bullet(
            "First-order Markov assumption: only the immediately previous tag "
            "is considered. Longer-range syntax is ignored.", s),
        _bullet(
            "Small corpus (350 training sentences) limits coverage.", s),
    ]
    return story


def _slide_conclusions(s) -> list:
    """Slide 16 – Conclusions & Future Work."""
    story = [
        _spacer(0.2),
        _head("Key take-aways", s),
        _bullet(
            "A first-order HMM + Viterbi is a complete, interpretable POS "
            "tagger achievable in pure Python from scratch.", s),
        _bullet(
            "Laplace smoothing is essential for robustness to unseen words.", s),
        _bullet(
            "The Viterbi algorithm guarantees the globally optimal tag sequence "
            "in polynomial time.", s),
        _spacer(0.3),
        _head("Possible extensions", s),
        Table(
            [["Extension", "Expected benefit"],
             ["Higher-order HMM (bigram/trigram tags)",
              "Captures longer-range syntactic dependencies"],
             ["Morphological OOV features (suffixes, capitalisation)",
              "Better handling of unknown words"],
             ["Discriminative model (CRF, BiLSTM-CRF)",
              "State-of-the-art accuracy (~97–98%)"],
             ["Pre-trained language model fine-tuning (BERT)",
              "Near-human performance on standard benchmarks"],
             ["Beam search decoding",
              "Faster approximate decoding for large tag sets"]],
            colWidths=[7 * cm, CONTENT_W - 7.2 * cm],
        ),
    ]
    t = story[-1]
    t.setStyle(_table_style(font_size=9, header_font_size=9))
    return story


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_pdf_slides(model, results: dict, figures_dir: str,
                        output_path: str):
    """
    Build and save the PDF slide deck.

    Parameters
    ----------
    model : HMMModel
    results : dict   – from evaluate_model()
    figures_dir : str
    output_path : str
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"\nGenerating PDF slides → {output_path}")

    s = _make_styles()

    # ---- Slide metadata ----
    slides_meta = [
        ("HMM Viterbi POS Tagger", _slide_title_page(s, results)),
        ("What is POS Tagging?",   _slide_pos_tagging(s)),
        ("Hidden Markov Model",    _slide_hmm_approach(s)),
        ("HMM Parameters & Smoothing", _slide_hmm_parameters(s)),
        ("Viterbi Decoding Algorithm",  _slide_viterbi(s)),
        ("Project Pipeline",       _slide_pipeline(s)),
        ("Dataset Overview",       _slide_dataset(s, model)),
        ("Results – Overall Accuracy", _slide_accuracy(s, results)),
        ("Results – Per-Tag Metrics",  _slide_per_tag_table(s, results)),
        ("Analysis – Per-Tag F1",
         _slide_figure(
             os.path.join(figures_dir, "per_tag_f1.png"),
             "Per-Tag F1 Score.  Bars are coloured red (low) → green (high).  "
             "Dashed line marks overall token accuracy.",
         )),
        ("Analysis – Precision / Recall / F1",
         _slide_figure(
             os.path.join(figures_dir, "precision_recall_f1.png"),
             "Precision, Recall, and F1 per POS Tag.  "
             "Grouped bars allow direct metric comparison.",
         )),
        ("Analysis – Confusion Matrix",
         _slide_figure(
             os.path.join(figures_dir, "confusion_matrix.png"),
             "Row-normalised confusion matrix.  "
             "Diagonal = per-tag recall.  "
             "Off-diagonal cells reveal systematic confusions.",
         )),
        ("Analysis – Transition Probabilities",
         _slide_figure(
             os.path.join(figures_dir, "transition_matrix.png"),
             "HMM Transition Probability Matrix.  "
             "Cell (i,j) = smoothed P(tag\u2c7c | tag\u1d62).",
         )),
        ("Analysis – Top Emission Words",
         _slide_figure(
             os.path.join(figures_dir, "top_emissions.png"),
             "Top-8 emission probabilities per tag.  "
             "Each sub-plot shows the words most associated with a given POS.",
         )),
        ("Key Findings & Discussion", _slide_discussion(s, results)),
        ("Conclusions & Future Work",  _slide_conclusions(s)),
    ]
    total = len(slides_meta)

    # ---- Build one PageTemplate per slide ----
    # We use a single-frame layout.  The chrome is drawn by the page callback.
    content_y = FOOTER_H + 0.2 * cm
    content_x = MARGIN

    page_templates = []
    for slide_idx, (title, _) in enumerate(slides_meta, start=1):
        chrome = _SlideCanvas(title, total, slide_idx)
        frame = Frame(
            content_x, content_y, CONTENT_W, CONTENT_H,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
            id=f"slide_{slide_idx}",
        )
        page_templates.append(
            PageTemplate(
                id=f"t{slide_idx}",
                frames=[frame],
                onPage=chrome,
                pagesize=landscape(A4),
            )
        )

    doc = BaseDocTemplate(
        output_path,
        pagesize=landscape(A4),
        leftMargin=0, rightMargin=0,
        topMargin=0, bottomMargin=0,
        showBoundary=False,
    )
    doc.addPageTemplates(page_templates)

    # ---- Assemble story with NextPageTemplate directives ----
    story = []
    for slide_idx, (title, content) in enumerate(slides_meta, start=1):
        if slide_idx == 1:
            story.append(NextPageTemplate(f"t{slide_idx}"))
        else:
            story.append(NextPageTemplate(f"t{slide_idx}"))
            story.append(FrameBreak())
        story.extend(content)

    doc.build(story)
    print(f"PDF slides saved to: {output_path}")
