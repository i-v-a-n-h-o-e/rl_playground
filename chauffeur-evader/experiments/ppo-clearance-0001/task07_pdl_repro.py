"""Task 07 cases from run 08 (tuning 22) and the short replay (P1)."""

import pytest
from chauffeur_evader.chauffeur_ppo import ChauffeurPPOEnv


def test_course_after_localized_curved_pdl():
    before_crossing = (0.19190723123703982, 0.5167969937666678)
    courses = []
    with ChauffeurPPOEnv(clearance=0.001) as env:
        for next_turn in (-1, 1):
            env.reset(options={"state": before_crossing})
            *_, hit = env.step(1)
            assert hit["event"] == "curved_PDL"
            *_, after = env.step(next_turn)
            assert after["phi"] == next_turn
            courses.append(after["psi"])

    # At this right-hand dispersal line the two outgoing characteristics
    # head toward contact (negative steering) and SE (positive steering).
    assert courses == pytest.approx([-0.46, 1.55], abs=0.02)


def test_defined_course_after_se_near_protective_barrier():
    # P1, smoke seed 43 at 2048 decisions, immediately before arrival at SE.
    start = (0.19358403378625447, 0.3219735390860646)
    turn = 0.036021292209625244
    with ChauffeurPPOEnv(clearance=0.001) as env:
        env.reset(options={"state": start})
        *_, hit = env.step(turn)
        assert hit["event"] == "SE"
        *_, after = env.step(turn)

    assert after["psi"] == pytest.approx(0.039126, abs=0.001)
    assert after["dt"] > 0
