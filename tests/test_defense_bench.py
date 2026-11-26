"""The test bench on its synthetic dataset: no data files are read.

Run from the repository root: python3 -m pytest tests/test_defense_bench.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import defense_bench as DB  # noqa: E402
import joint_defense_common as C  # noqa: E402

EXAMPLE = f"{REPO_ROOT / 'examples' / 'defense_example.py'}:Dither"
EXAMPLE_CELLS = 0.5
RECOVER_REPORTS = 50
RECOVER_SEED = 3
FINE_BITS = "6/4"
GAIN_TOL = 1e-9


@pytest.fixture(scope="module")
def selftest() -> dict[str, dict]:
    return DB.selftest()


def test_none_is_identified(selftest):
    none = selftest["none"]
    assert none["observers"]["scaled|T=1"]["mean"] > DB.SELFTEST_NONE_FACTOR * none["chance"]
    assert not none["verdict"]["private"]
    assert not none["verdict"]["passes"]


def test_rotation_per_group_passes(selftest):
    rotation = selftest["rotation_group"]
    assert rotation["verdict"] == dict(private=True, keeps_gain=True, passes=True)
    assert rotation["gain_first_stream"]["mean"] >= DB.SELFTEST_MIN_GAIN
    assert set(rotation["observers"]) == {f"{o}|T={T}" for o in DB.OBSERVERS for T in DB.T_REPORT}


def test_example_defense_runs_through_the_loader():
    defense = DB.load_defense(EXAMPLE, DB.parse_params([f"cells={EXAMPLE_CELLS}"]))
    summary = DB.run_bench(defense, seeds=DB.SELFTEST_SEEDS, alpha=DB.SELFTEST_ALPHA,
                           gain_reports=DB.SELFTEST_GAIN_REPORTS, synthetic=True)
    assert summary["defense"] == f"dither_c{EXAMPLE_CELLS:g}"
    assert not summary["verdict"]["private"]            # half a cell of dither hides nothing
    assert 0.0 < summary["gain_first_stream"]["mean"] < 1.0
    assert summary["loss_db"] > 0.0


def test_rotation_secret_undoes_the_rotation():
    """With the secret, the access point steers as well as with unprotected feedback."""
    d = DB.synthetic_angles()
    angles = (d["phi"][:RECOVER_REPORTS], d["psi"][:RECOVER_REPORTS])
    targets = d["y"][:RECOVER_REPORTS]
    truth = C.givens_to_v(*angles)
    with DB.codebook(DB.CODEBOOKS["9/7"]):
        for name in DB.ROTATIONS:
            defense = DB.load_defense(name, {})
            reported, secret = defense.report(angles, targets, np.random.default_rng(RECOVER_SEED))
            kept = DB.G.gains(defense.recover(reported, secret), truth)[0].mean()
            blind = DB.G.gains(DB.Defense().recover(reported, None), truth)[0].mean()
            assert kept > DB.DEFAULT_BETA > blind, name


def test_codebook_is_in_force_and_restored():
    before = DB.codebook_bits()
    with DB.codebook(DB.CODEBOOKS[FINE_BITS]):
        assert DB.codebook_bits() == DB.CODEBOOKS[FINE_BITS]
        fine = DB.cell_widths()
    assert DB.codebook_bits() == before
    assert all(f < c for f, c in zip(fine, DB.cell_widths()))


def test_loader_rejects_unknown_and_non_defense(tmp_path):
    with pytest.raises(ValueError):
        DB.load_defense("no_such_defense", {})
    other = tmp_path / "other.py"
    other.write_text("thing = object()\n")
    with pytest.raises(TypeError):
        DB.load_defense(f"{other}:thing", {})


def test_bench_rejects_a_report_of_the_wrong_shape():
    class Truncating(DB.Defense):
        name = "truncating"

        def report(self, angles, targets, rng):
            return (angles[0][:, :1], angles[1][:, :1]), None

    with pytest.raises(ValueError):
        DB.run_bench(Truncating(), seeds=1, gain_reports=DB.SELFTEST_GAIN_REPORTS, synthetic=True)


def test_stat_upper_bound():
    s = DB.stat([0.1, 0.2, 0.3])
    assert s["mean"] == pytest.approx(0.2)
    assert s["ub95"] > s["mean"]
    assert DB.stat([0.5])["ub95"] == pytest.approx(0.5, abs=GAIN_TOL)
