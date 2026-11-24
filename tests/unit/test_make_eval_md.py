"""EVAL.md generator: built from a real (fake-backed) run, every section must render."""
import json

from evals.configs import EvalConfig
from evals.make_eval_md import build_eval_md
from evals.run import run_all
from tests.unit.test_eval_runner import GOLDEN, SETTINGS, EvalLLM, NamespacedStore, backends, corpus


async def make_run(tmp_path, llm=None):
    configs = [EvalConfig("A", 40, 5, False, description="baseline"), EvalConfig("B", 40, 5, True, description="rerank"),
               EvalConfig("E", 40, 5, True, rerank_threshold=0.1, description="threshold")]
    paths = await run_all(backends(llm or EvalLLM(), NamespacedStore()), GOLDEN, configs, corpus(tmp_path),
                          SETTINGS, tmp_path / "out")
    return json.loads(paths["json"].read_text(encoding="utf-8")), paths


async def test_report_contains_every_section_with_real_numbers(tmp_path) -> None:
    run, _ = await make_run(tmp_path)
    selftest = {"n_questions": 4, "by_kind": {"correct": {"n": 4, "correct_verdict_ok": 1.0, "faithful_verdict_ok": 1.0},
                                              "wrong_fact": {"n": 4, "correct_verdict_ok": 1.0, "faithful_verdict_ok": 1.0},
                                              "unsupported_claim": {"n": 4, "faithful_verdict_ok": 1.0}}}
    human = {"correct": {"n": 20, "agreement": 0.9, "kappa": 0.7}, "faithful": {"n": 20, "agreement": 0.95, "kappa": None}}
    md = build_eval_md(run, GOLDEN, selftest, human, "evals/results/test")
    for heading in ["## 1. What was measured", "## 2. Judge reliability", "## 3. Configurations compared", "## 4. Results",
                    "## 5. Winner", "## 6. Effect of each design choice", "## 7. Refusal threshold calibration",
                    "## 8. Failure analysis", "## 9. Limitations", "## 10. Reproduce"]:
        assert heading in md, heading
    assert "| hit@5 | 1.000 | 1.000 | 1.000 |" in md                       # real numbers, three config columns
    assert "| A → B | add reranking |" in md and "| B → E |" in md
    assert "Agreement with a human" in md and "90%" in md and "undefined" in md
    assert "100%" in md and "numbers changed" in md
    assert "1 question cannot be answered" in md and "TODO" in md           # analysis left for the author


async def test_same_model_judge_is_flagged_and_missing_checks_are_called_out(tmp_path) -> None:
    run, _ = await make_run(tmp_path)
    run["meta"]["judge_model"] = "x (same as LLM)"
    md = build_eval_md(run, GOLDEN, None, None, "evals/results/test")
    assert "biased toward its own" in md
    assert "Self-test not run yet" in md and "Human agreement not measured yet" in md


async def test_ceiling_effect_and_empty_failures_are_explained(tmp_path) -> None:
    run, _ = await make_run(tmp_path)
    md = build_eval_md(run, GOLDEN, None, None, "r")
    assert "ceiling effect" in md and "does not exercise the system's weak points" in md


async def test_failures_are_listed_with_issues(tmp_path) -> None:
    run, _ = await make_run(tmp_path, EvalLLM(answer_unanswerable=True))
    md = build_eval_md(run, GOLDEN, None, None, "r")
    assert "hallucinated_on_unanswerable" in md and "Do you sell zzzboots?" in md


# ------------------------------------------------------------------ follow-up run section

from evals.make_eval_md import build_addendum  # noqa: E402


async def make_two_config_run(tmp_path, threshold):
    configs = [EvalConfig("B", 40, 5, True, description="rerank"),
               EvalConfig("E", 40, 5, True, rerank_threshold=threshold, description="threshold")]
    paths = await run_all(backends(EvalLLM(), NamespacedStore()), GOLDEN, configs, corpus(tmp_path),
                          SETTINGS, tmp_path / "out")
    return json.loads(paths["json"].read_text(encoding="utf-8"))


async def test_addendum_reports_b_to_e_with_real_numbers(tmp_path) -> None:
    run = await make_two_config_run(tmp_path, threshold=0.1)
    md = build_addendum(run, "Run 2: production LLM", "evals/results/x")
    assert md.startswith("## Run 2: production LLM")
    assert "| correct (judge) | 1.000 | 1.000 |" in md and "### Effect of the corrected refusal threshold" in md
    assert "Questions refused before any LLM call: 0 to 0" in md and "TODO" in md


async def test_addendum_counts_threshold_refusals_before_the_llm(tmp_path) -> None:
    run = await make_two_config_run(tmp_path, threshold=1.5)       # nothing can pass: every question refused early
    md = build_addendum(run, "Run 2", "r")
    assert "| refused before the LLM (threshold) | 0 | 3 |" in md
    assert "False refusals: 0.000 to 1.000" in md and "false_refusal" in md


async def test_cli_appends_without_touching_the_existing_report(tmp_path, monkeypatch) -> None:
    import sys
    from evals import make_eval_md as module
    run = await make_two_config_run(tmp_path, 0.1)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(json.dumps(run), encoding="utf-8")
    report = tmp_path / "EVAL.md"
    report.write_text("# Evaluation report\n\nORIGINAL\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", str(run_dir), "--append-to", str(report), "--title", "Run 2: test"])
    module.main()
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Evaluation report\n\nORIGINAL") and "## Run 2: test" in text
