"""Truncate a Viola-Jones cascade checkpoint to keep only the first N stages.

Usage:
    python tools/truncate_checkpoint.py weights/19/cvj_weights_1778796130.pkl 5
    # → saves weights/19/cvj_weights_1778796130_trunc5.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from violajones import ViolaJones


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="Path to .pkl checkpoint")
    ap.add_argument("keep_stages", type=int,
                    help="Number of stages to keep (e.g. 5)")
    ap.add_argument("--output", default=None,
                    help="Output path (default: <input>_truncN.pkl)")
    args = ap.parse_args()

    clf = ViolaJones.load(args.checkpoint)
    total = len(clf.clfs)
    if args.keep_stages >= total:
        print(f"Checkpoint has {total} stages, requested {args.keep_stages}. "
              f"Nothing to truncate.")
        return
    if args.keep_stages < 1:
        ap.error("keep_stages must be >= 1")

    removed = clf.clfs[args.keep_stages:]
    clf.clfs = clf.clfs[:args.keep_stages]

    out = args.output
    if out is None:
        p = Path(args.checkpoint)
        stem = p.stem if not p.stem.endswith(".pkl") else p.stem[:-4]
        out = str(p.parent / f"{stem}_trunc{args.keep_stages}.pkl")
    # clf.save appends .pkl, so strip it if present
    save_path = out[:-4] if out.endswith(".pkl") else out
    clf.save(save_path)
    print(f"Truncated: {total} → {args.keep_stages} stages")
    print(f"Removed stages {args.keep_stages+1}–{total} "
          f"({sum(len(s.clfs) for s in removed)} weak classifiers discarded)")
    print(f"Saved → {save_path}.pkl")


if __name__ == "__main__":
    main()
