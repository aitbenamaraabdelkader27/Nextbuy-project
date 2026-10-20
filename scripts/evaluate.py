"""Run the frozen evaluation outside Streamlit and write reviewable metrics."""

import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nextbuy_core.analytics import association_pairs, kpis, product_statistics
from nextbuy_core.data import ROOT, available_sources, load_tables, merge_tables
from nextbuy_core.modeling import run_experiment


def main() -> None:
    report = {}
    for source, directory in available_sources().items():
        start = time.perf_counter()
        frame = merge_tables(load_tables(directory))
        result = run_experiment(frame)
        report[source] = {
            "kpis": kpis(frame),
            "sizes": result["sizes"],
            "scores": result["scores"].to_dict("records"),
            "reports": {name: detail["report"] for name, detail in result["details"].items()},
            "test_prevalence": result["test_prevalence"],
            "top_products": product_statistics(frame).nlargest(5, "items").to_dict("records"),
            "top_pairs": association_pairs(frame).head(5).to_dict("records"),
            "elapsed_seconds": round(time.perf_counter() - start, 2),
        }
        print(source, result["sizes"], "\n", result["scores"].to_string(index=False), flush=True)
    report["peak_process_rss_mib"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
    )
    (ROOT / "docs/evaluation.json").write_text(json.dumps(report, indent=2, default=int))


if __name__ == "__main__":
    main()
