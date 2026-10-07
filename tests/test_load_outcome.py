"""chemparseplot loads an eOn outcome without taking a trajectory."""

import json

from chemparseplot.contracts.load_outcome import load_outcome


def test_load_outcome_reads_job_type_and_status():
    raw = json.dumps(
        {"jobType": "minimization", "statusCode": 0, "statusText": "good"}
    )
    assert load_outcome(raw) == {
        "jobType": "minimization",
        "statusCode": 0,
        "statusText": "good",
    }
