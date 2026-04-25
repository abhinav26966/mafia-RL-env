#!/usr/bin/env python3
"""Run N games with a specified model checkpoint, dump JSON metrics + transcripts.

Phase 5 deliverable. Used for baseline-vs-trained comparison.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="'baseline' | 'trained' | HF model id")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--server", default="http://localhost:8000")
    args = parser.parse_args()
    raise NotImplementedError(
        f"eval_model — Phase 5. Args: model={args.model}, games={args.games}, out={args.out}"
    )


if __name__ == "__main__":
    main()
