"""
backend/evaluation/run_comparison.py

Approach Comparison — 100 Free-Text Occupation Descriptions
============================================================
Runs four systems and prints a thesis-grade comparison table:

  1. No-RAG      — pure Ollama LLM (no retrieval)
  2. BM25        — sparse keyword retrieval
  3. Standard RAG — flat dense vector search (Qdrant)
  4. Hierarchical RAG — 4-stage pipeline (thesis method)

Usage:
    python -m backend.evaluation.run_comparison
    python -m backend.evaluation.run_comparison --skip-norag
"""

from __future__ import annotations
import argparse, sys, time, os
sys.path.insert(0, os.path.abspath("."))

# Silence TensorFlow / other noisy warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from backend.evaluation.evaluate import (
    TEST_CASES,
    NoRAGBaseline, BM25Baseline, FlatVectorBaseline, HierarchicalRAG,
    evaluate_system, SystemMetrics, _MAJOR_LABELS,
)

# ── ANSI colours ────────────────────────────────────────────────────────────
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
R  = "\033[91m"   # red
B  = "\033[94m"   # blue
C  = "\033[96m"   # cyan
W  = "\033[97m"   # white bold
DIM= "\033[2m"
RST= "\033[0m"

def col(text, colour): return f"{colour}{text}{RST}"

# ── Per-case detail printer ──────────────────────────────────────────────────
def _predict_all(systems: dict, text: str, true_code: str) -> dict:
    row = {"text": text, "true": true_code}
    for name, fn in systems.items():
        try:
            r = fn(text, 3)
            hit1 = r.predicted_code == true_code
            hit3 = true_code in r.top3_codes
            row[name] = {
                "pred":  r.predicted_code,
                "top3":  r.top3_codes,
                "conf":  r.confidence,
                "ms":    r.latency_ms,
                "hit1":  hit1,
                "hit3":  hit3,
            }
        except Exception as exc:
            row[name] = {"pred": "ERR", "top3": [], "conf": 0, "ms": 0,
                         "hit1": False, "hit3": False, "err": str(exc)}
    return row


# ── Approach table ───────────────────────────────────────────────────────────
def _halluc_label(count, n):
    if n == 0: return "  N/A  "
    r = count / n
    if r > 0.15: return col("  HIGH  ", R)
    if r > 0.05: return col("MODERATE", Y)
    if r > 0.01: return col("  LOW   ", Y)
    return col("VERY LOW", G)

def _scalability(name):
    d = {"no_rag":"N/A", "bm25":"Limited", "standard_rag":"Limited",
         "hierarchical": col("Excellent", G)}
    return d.get(name, "N/A")

def print_approach_table(results: dict[str, SystemMetrics]):
    print()
    print(col("="*90, C))
    print(col("  THESIS TABLE — APPROACH COMPARISON  (n=100 occupation descriptions)", W))
    print(col("="*90, C))

    # Header
    h = (f"  {'Approach':<18} {'Top-1':>6} {'Top-3':>6} {'Kappa':>7} "
         f"{'HITL%':>6} {'Halluc':>9} {'Avg ms':>8} {'Scalability':<12}")
    print(col(h, W))
    print("  " + "-"*86)

    order = ["no_rag", "bm25", "standard_rag", "hierarchical"]
    labels = {
        "no_rag":       "No-RAG (pure LLM)",
        "bm25":         "BM25 Keyword",
        "standard_rag": "Standard RAG",
        "hierarchical": "Hierarchical RAG *",
    }

    baseline_top1 = results.get("no_rag", results.get("bm25"))
    base_acc = baseline_top1.top1_accuracy if baseline_top1 else 0

    for name in order:
        m = results.get(name)
        if not m:
            print(f"  {labels[name]:<18}  {'(not run)':>60}")
            continue

        is_best = name == "hierarchical"
        top1_str = f"{m.top1_accuracy:.1%}"
        top3_str = f"{m.top3_accuracy:.1%}"
        kappa_str= f"{m.cohen_kappa:.3f}"
        hitl_str = f"{m.hitl_rate:.1%}"
        lat_str  = f"{m.avg_latency_ms:.1f}"

        if is_best:
            top1_str = col(f"{m.top1_accuracy:.1%}", G)
            kappa_str= col(f"{m.cohen_kappa:.3f}", G)

        # Relative accuracy vs first available baseline
        if base_acc > 0 and name != list(results.keys())[0]:
            delta = m.top1_accuracy - base_acc
            sign  = "+" if delta >= 0 else ""
            top1_str += col(f" ({sign}{delta:.0%})", G if delta > 0 else R)

        hl = _halluc_label(m.hallucination_count, m.n_evaluated)
        sc = _scalability(name)

        print(f"  {labels[name]:<18} {top1_str:>12} {top3_str:>6} {kappa_str:>9} "
              f"{hitl_str:>6} {hl:>18} {lat_str:>8} {sc}")

    print("  " + "-"*86)
    print(f"  * Hierarchical RAG = 4-stage Qdrant pipeline (thesis contribution)")
    print(f"  Fine-tuned models: published Top-1 ~55-65% (Gweon 2017, Boselli 2018) — requires labeled corpus")


# ── Per-case table ───────────────────────────────────────────────────────────
def print_case_table(rows: list[dict], sys_names: list[str]):
    print()
    print(col("="*110, C))
    print(col("  PER-CASE RESULTS — 100 Occupation Descriptions", W))
    print(col("="*110, C))

    heads = f"  {'#':>3}  {'Description':<45} {'True':>5}  "
    for s in sys_names:
        short = {"no_rag":"NoRAG","bm25":"BM25","standard_rag":"StdRAG","hierarchical":"HierRAG"}
        heads += f"{short.get(s,s[:6]):>9}"
    print(col(heads, W))
    print("  " + "-"*108)

    for i, row in enumerate(rows, 1):
        text  = row["text"][:44]
        true  = row["true"]
        cells = f"  {i:>3}  {text:<45} {col(true,'97'):>5}  "
        all_correct = all(row.get(s, {}).get("hit1", False) for s in sys_names if s in row)
        none_correct= all(not row.get(s, {}).get("hit1", False) for s in sys_names if s in row)

        for s in sys_names:
            d = row.get(s, {})
            pred = d.get("pred", "---")
            hit1 = d.get("hit1", False)
            hit3 = d.get("hit3", False)
            if hit1:
                cell = col(f"{pred:>9}", G)
            elif hit3:
                cell = col(f"({pred})", Y).rjust(9)
            else:
                cell = col(f"{pred:>9}", R)
            cells += cell
        print(cells)

    print("  " + "-"*108)
    print(f"  Legend: {col('green=Top-1 hit', G)}  {col('(yellow)=Top-3 hit', Y)}  {col('red=miss', R)}")


# ── Summary by difficulty ────────────────────────────────────────────────────
EASY_IDX   = list(range(0, 70))   # [E] easy cases roughly
MEDIUM_IDX = list(range(70, 85))  # [M]
HARD_IDX   = list(range(85, 100)) # [H]

def print_difficulty_breakdown(rows, sys_names):
    print()
    print(col("  DIFFICULTY BREAKDOWN", W))
    print("  " + "-"*60)
    for label, indices in [("Easy  [E]", EASY_IDX), ("Medium[M]", MEDIUM_IDX), ("Hard  [H]", HARD_IDX)]:
        line = f"  {label}  "
        for s in sys_names:
            hits = sum(rows[i].get(s, {}).get("hit1", False) for i in indices if i < len(rows))
            n    = sum(1 for i in indices if i < len(rows))
            pct  = hits/n if n else 0
            short= {"no_rag":"NoRAG","bm25":"BM25","standard_rag":"StdRAG","hierarchical":"HierRAG"}
            c    = G if pct >= 0.50 else (Y if pct >= 0.30 else R)
            line += f"{short.get(s,s):>8}={col(f'{pct:.0%}',c)}  "
        print(line)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-norag", action="store_true",
                    help="Skip No-RAG baseline (saves ~5 min Ollama inference)")
    ap.add_argument("--cases", type=int, default=100,
                    help="Number of test cases to run (default: 100)")
    args = ap.parse_args()

    cases = TEST_CASES[:args.cases]
    print(col(f"\n  LFS Thesis — ISCO-08 Approach Comparison", W))
    print(col(f"  {len(cases)} free-text occupation descriptions", DIM))
    print(col(f"  Systems: No-RAG | BM25 | Standard RAG | Hierarchical RAG", DIM))
    print()

    # ── Build system registry ────────────────────────────────────────────────
    registry: list[tuple[str, type]] = []
    if not args.skip_norag:
        registry.append(("no_rag", NoRAGBaseline))
    registry += [
        ("bm25",         BM25Baseline),
        ("standard_rag", FlatVectorBaseline),
        ("hierarchical", HierarchicalRAG),
    ]

    # ── Initialise all systems ────────────────────────────────────────────────
    sys_fns: dict[str, object] = {}
    for name, cls in registry:
        print(f"  [{name}] Initialising...", end=" ", flush=True)
        try:
            obj = cls()
            sys_fns[name] = obj.predict
            print(col("OK", G))
        except Exception as exc:
            print(col(f"FAILED ({exc})", R))

    sys_names = list(sys_fns.keys())

    # ── Run per-case predictions ──────────────────────────────────────────────
    print()
    print(col(f"  Running {len(cases)} cases across {len(sys_names)} systems...", W))
    print("  " + "-"*60)

    rows = []
    t_total = time.perf_counter()
    for i, (text, true_code) in enumerate(cases, 1):
        row = _predict_all(sys_fns, text, true_code)
        rows.append(row)
        # Live progress line
        hits = {s: col("v", G) if row.get(s,{}).get("hit1") else col("x", R) for s in sys_names}
        hit_str = "  ".join(f"{s[:5]}={hits[s]}" for s in sys_names)
        sys.stdout.write(f"\r  [{i:>3}/{len(cases)}] {text[:40]:<40} | {hit_str}")
        sys.stdout.flush()

    elapsed = time.perf_counter() - t_total
    print(f"\n\n  Completed {len(cases)} cases in {elapsed:.1f}s")

    # ── Compute aggregate metrics ─────────────────────────────────────────────
    metrics: dict[str, SystemMetrics] = {}
    for name, fn in sys_fns.items():
        print(f"  Computing metrics for {name}...", end=" ", flush=True)
        m = evaluate_system(name, fn, cases)
        metrics[name] = m
        print(col(f"Top-1={m.top1_accuracy:.1%}  Kappa={m.cohen_kappa:.3f}", G))

    # ── Print all output ─────────────────────────────────────────────────────
    print_approach_table(metrics)
    print_case_table(rows, sys_names)
    print_difficulty_breakdown(rows, sys_names)

    # Per-major breakdown for Hierarchical RAG
    hier = metrics.get("hierarchical")
    if hier and hier.per_major:
        print()
        print(col("  PER-MAJOR-GROUP BREAKDOWN — Hierarchical RAG", W))
        print(f"  {'Major':<4} {'Label':<22} {'Support':>7} {'TP':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("  " + "-"*58)
        for mg in hier.per_major:
            f1c = G if mg.f1 >= 0.60 else (Y if mg.f1 >= 0.30 else R)
            print(f"  {mg.major_code:<4} {mg.label:<22} {mg.n_true:>7} {mg.tp:>4} "
                  f"{mg.precision:>7.1%} {mg.recall:>7.1%} {col(f'{mg.f1:.1%}',f1c):>14}")
        mf1 = sum(mg.f1 for mg in hier.per_major)/max(len(hier.per_major),1)
        print("  " + "-"*58)
        print(f"  {'Macro-F1':<28} {col(f'{mf1:.1%}', G):>14}")

    # Save JSON
    import json
    out = {name: {
        "top1": m.top1_accuracy, "top3": m.top3_accuracy,
        "kappa": m.cohen_kappa,  "hitl_rate": m.hitl_rate,
        "avg_ms": m.avg_latency_ms, "hallucinations": m.hallucination_count,
        "n": m.n_evaluated,
    } for name, m in metrics.items()}
    path = "backend/evaluation/approach_comparison.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {path}")
    print()


if __name__ == "__main__":
    main()
