"""
eval/legacy_thesis_ch6/

Thesis Chapter 6 evaluation framework: BM25 / flat-vector / hierarchical-
RAG comparison over a 100-item synthetic occupation-description corpus.
Predates the eval/ harness (run_eval.py, ablation_runner.py, etc.), which
extends rather than replaces it -- see
Documentation/Conference_I_Reviewer_2/EVALUATION_PROTOCOL.md for which
subsystem's numbers to cite where. Not dead code: eval/run_eval.py reuses
its BM25Baseline, eval/wisco_subsample_3system_comparison.py reuses its
full 3-system comparison, and eval/legacy_thesis_ch6/test_evaluate.py
covers it directly.

Moved here 2026-08-24 from backend/evaluation/ (a documentation-
completeness audit found zero real production coupling -- only one test
file imported it -- so it was relocated out of the application package
and under eval/, where every other evaluation-only module already
lives). `wisco/` (Module A's WISCO parsing pipeline) moved with it for
the same reason. Generated run outputs that used to sit next to this
code now live in `eval/results/legacy_thesis_ch6/` instead, consistent
with `eval/results/`'s existing convention for run/ablation output.
"""
