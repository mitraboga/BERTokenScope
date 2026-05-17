import time

from ber_tokenscope.schemas import FinancialAnalysis, FinancialSignal
from observability.tracker import RunTracker


def test_run_tracker_persists_and_reads_artifacts(tmp_path):
    tracker = RunTracker(
        run_dir=tmp_path / "runs",
        database_url=f"sqlite:///{tmp_path / 'runs.db'}",
        max_history=10,
    )
    result = FinancialAnalysis(
        text="Revenue growth was strong.",
        sentiment=FinancialSignal(
            label="positive",
            score=0.8,
            explanation="Optimistic language dominates.",
        ),
        risk_score=0.0,
        uncertainty_score=0.0,
        optimism_score=1.0,
        matched_terms={"risk": [], "uncertainty": [], "optimism": ["growth"]},
    )

    record = tracker.track(
        task="financial-analysis",
        request={"text": result.text},
        result=result,
        started_at=time.perf_counter(),
        summary={"sentiment": result.sentiment.label},
    )
    artifact = tracker.read_artifact(record.run_id)

    assert record.status == "completed"
    assert artifact["result"]["sentiment"]["label"] == "positive"
    assert tracker.list_runs(limit=1)[0].run_id == record.run_id


def test_run_tracker_trims_history(tmp_path):
    tracker = RunTracker(
        run_dir=tmp_path / "runs",
        database_url=f"sqlite:///{tmp_path / 'runs.db'}",
        max_history=2,
    )

    for index in range(3):
        tracker.track(
            task="test",
            request={"index": index},
            result={"ok": True},
            started_at=time.perf_counter(),
        )

    assert len(tracker.list_runs(limit=10)) == 2
