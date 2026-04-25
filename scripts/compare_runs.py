#!/usr/bin/env python3
"""Stack baseline vs trained metrics into the demo plots.

Phase 5 deliverable. Generates PNGs into demo/plots/.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to baseline eval JSON")
    parser.add_argument("--trained", required=True, help="Path to trained eval JSON")
    parser.add_argument("--out", required=True, help="Output dir for PNGs")
    args = parser.parse_args()
    raise NotImplementedError(
        f"compare_runs — Phase 5. baseline={args.baseline} trained={args.trained} out={args.out}"
    )


if __name__ == "__main__":
    main()
