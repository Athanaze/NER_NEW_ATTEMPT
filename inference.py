#!/usr/bin/env python3
"""
Inference script for the finetuned GLiNER legal NER model
"""

import argparse
from gliner import GLiNER


def main():
    parser = argparse.ArgumentParser(description="Run inference with finetuned GLiNER model")
    parser.add_argument("--model_path", type=str, default="./gliner-legal-finetuned",
                        help="Path to the finetuned model")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to analyze (if not provided, uses example)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for entity detection")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = GLiNER.from_pretrained(args.model_path)

    # Define labels
    labels = ["doctrine", "jurisprudence", "articles de loi"]

    # Use provided text or example
    if args.text:
        text = args.text
    else:
        text = """Entscheid vom 12. April 2021 Besetzung lic.iur. Gion Tomaschett,
        Vizepräsident Dr.med. Urs Gössi, Richter Dr.med. Pierre Lichtenhahn, Richter
        MLaw Tanja Marty, a.o. Gerichtsschreiberin Parteien A. Art. 32 Abs. 2 SuG als
        verjährt betrachtet. Die Rechtslage hinsichtlich der zurückbehaltenen Rest-
        beträge betreffen."""

    print(f"\nAnalyzing text:\n{text}\n")
    print(f"Looking for entities: {', '.join(labels)}")
    print(f"Threshold: {args.threshold}\n")

    # Predict entities
    entities = model.predict_entities(text, labels, threshold=args.threshold)

    # Display results
    if entities:
        print(f"Found {len(entities)} entities:")
        print("-" * 80)
        for entity in entities:
            print(f"Text: {entity['text']}")
            print(f"Label: {entity['label']}")
            print(f"Score: {entity['score']:.4f}")
            print(f"Span: ({entity['start']}, {entity['end']})")
            print("-" * 80)
    else:
        print("No entities found.")

    return entities


if __name__ == "__main__":
    main()
