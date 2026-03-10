"""
main.py – Entry point for the HMM Viterbi POS Tagger.

Usage
-----
    python main.py [--train TRAIN_FILE] [--test TEST_FILE]
                   [--figures FIGURES_DIR] [--report REPORT_FILE]
                   [--slides SLIDES_FILE] [--smoothing SMOOTHING]

Defaults
--------
    --train   data/train.txt
    --test    data/test.txt
    --figures figures/
    --report  report/report.pdf
    --slides  report/slides.pdf
    --smoothing 1.0
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="HMM Viterbi Part-of-Speech Tagger"
    )
    parser.add_argument(
        "--train",
        default=os.path.join("data", "train.txt"),
        help="Path to the training corpus (default: data/train.txt)",
    )
    parser.add_argument(
        "--test",
        default=os.path.join("data", "test.txt"),
        help="Path to the test corpus (default: data/test.txt)",
    )
    parser.add_argument(
        "--figures",
        default="figures",
        help="Directory to save generated figures (default: figures/)",
    )
    parser.add_argument(
        "--report",
        default=os.path.join("report", "report.pdf"),
        help="Output path for the PDF report (default: report/report.pdf)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Laplace smoothing constant (default: 1.0)",
    )
    parser.add_argument(
        "--slides",
        default=os.path.join("report", "slides.pdf"),
        help="Output path for the PDF slide deck (default: report/slides.pdf)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Step 1: Train the HMM
    # ------------------------------------------------------------------
    from src.train import train_model
    model = train_model(args.train, smoothing=args.smoothing)

    # ------------------------------------------------------------------
    # Step 2: Evaluate on the test set
    # ------------------------------------------------------------------
    from src.evaluate import evaluate_from_files
    results = evaluate_from_files(model, args.test)

    # ------------------------------------------------------------------
    # Step 3: Generate figures
    # ------------------------------------------------------------------
    from src.visualize import generate_all_figures
    generate_all_figures(model, results, args.figures)

    # ------------------------------------------------------------------
    # Step 4: Generate the PDF report
    # ------------------------------------------------------------------
    from report.generate_report import generate_pdf_report
    generate_pdf_report(model, results, args.figures, args.report)

    # ------------------------------------------------------------------
    # Step 5: Generate the PDF slide deck
    # ------------------------------------------------------------------
    from report.generate_slides import generate_pdf_slides
    generate_pdf_slides(model, results, args.figures, args.slides)

    print(f"\nDone!  Report saved to: {args.report}")
    print(f"        Slides saved to: {args.slides}")


if __name__ == "__main__":
    main()
