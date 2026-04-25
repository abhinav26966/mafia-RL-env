#!/usr/bin/env python3
"""Curate hand-picked transcript pairs for the README demo section.

Phase 5 deliverable. Picks games where the baseline lost as werewolf and
the trained model won — best storytelling material.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--out", default="demo/transcripts/")
    args = parser.parse_args()
    raise NotImplementedError(
        f"make_demo_assets — Phase 5. baseline={args.baseline} trained={args.trained} out={args.out}"
    )


if __name__ == "__main__":
    main()
