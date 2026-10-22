"""Pin the default (non loss-space) QTLS0 and waterfall fit outputs.

``data/flag_off_golden.json`` was generated from commit 9641041, before the
loss-space option existed, by running this file as a script:

    PYTHONPATH=tests python tests/test_flag_off_regression.py tests/data/flag_off_golden.json

With ``loss_space`` left at its default, every fit must reproduce those numbers.
"""

import contextlib
import io
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from synthetic import (
    overcoupled_qtls0_device,
    overcoupled_waterfall_device,
    waterfall_params,
)
from rrfit.QTLS0fit import fit_qtls0
from rrfit.waterfall import Fit_QIntVsTemp, fitIterated

GOLDEN = Path(__file__).parent / "data" / "flag_off_golden.json"


def _params_dump(params):
    return {name: [p.value, p.stderr] for name, p in params.items()}


def _qtls0_case(weighted, **kwargs):
    device = overcoupled_qtls0_device()
    fit_qtls0(device, weighted=weighted, **kwargs)
    return dict(
        params=_params_dump(device.best_params),
        qtls0=device.qtls0,
        qtls0_err=device.qtls0_err,
        redchi=device.qtls0_fit_result.redchi,
    )


def _waterfall_case(consistent, **kwargs):
    device = overcoupled_waterfall_device()
    fit_params, _ = Fit_QIntVsTemp(
        device,
        waterfall_params(Q_TLS0=2e7, tc=1.1),
        consistent=consistent,
        makePlot=False,
        **kwargs,
    )
    return dict(
        params=_params_dump(fit_params),
        redchi=device.waterfall_fit_result.redchi,
        summary=device.waterfall_best_values,
    )


def _fit_iterated_case(**kwargs):
    device = overcoupled_waterfall_device()
    random.seed(1)
    bounds = {"Q_TLS0": (1e7, 5e7), "tc": (1.0, 1.4)}
    init, final, red_chi2, _ = fitIterated(
        device, bounds, 3, consistent=False, makePlot=False,
        init_params=waterfall_params(), **kwargs,
    )
    return dict(
        init={k: v.tolist() for k, v in init.items()},
        final={k: v.tolist() for k, v in final.items()},
        chi=red_chi2.tolist(),
        best=_params_dump(device.best_params),
    )


CASES = {
    "qtls0_weighted=True": lambda **kw: _qtls0_case(True, **kw),
    "qtls0_weighted=False": lambda **kw: _qtls0_case(False, **kw),
    "wf_consistent=False": lambda **kw: _waterfall_case(False, **kw),
    "wf_consistent=True": lambda **kw: _waterfall_case(True, **kw),
    "fitIterated": _fit_iterated_case,
}


def _run(case, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return CASES[case](**kwargs)


def _assert_matches(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_matches(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_matches(a, e)
    elif expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected, rel=1e-7, abs=0)


@pytest.mark.parametrize("case", list(CASES))
def test_default_fit_outputs_match_golden(case):
    expected = json.loads(GOLDEN.read_text())[case]
    _assert_matches(json.loads(json.dumps(_run(case))), expected)


@pytest.mark.parametrize("case", list(CASES))
def test_explicit_loss_space_false_matches_golden(case):
    expected = json.loads(GOLDEN.read_text())[case]
    _assert_matches(json.loads(json.dumps(_run(case, loss_space=False))), expected)


if __name__ == "__main__":
    golden = {case: _run(case) for case in CASES}
    Path(sys.argv[1]).write_text(json.dumps(golden, indent=1) + "\n")
