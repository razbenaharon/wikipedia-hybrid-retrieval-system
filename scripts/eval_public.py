"""Self-evaluation on the 50 public queries (mean NDCG@10)."""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from eval import evaluate_run, load_query_file
from main import run
from utils import PUBLIC_QUERIES_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=PUBLIC_QUERIES_PATH,
        help="Path to public query JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_query_file(args.queries_path)
    queries = [r["query"] for r in rows]
    ground_truth = [r["relevant_page_ids"] for r in rows]

    t0 = time.perf_counter()
    stats = evaluate_run(queries, ground_truth, run)
    elapsed = time.perf_counter() - t0

    print(f"queries_path={args.queries_path}")
    print(f"public_queries={len(queries)}")
    print(f"mean_ndcg@10={stats['mean_ndcg@10']:.4f}")
    print(f"query_phase_time={elapsed:.2f}s")


if __name__ == "__main__":
    main()
