import pytest
import yaml

from eval.schema import DEFAULT_GOLDENS_PATH, load_goldens

VALID = {
    "id": "q-001",
    "question": "What is a Pod?",
    "intent": "rag",
    "golden_sources": ["concepts__workloads__pods.html"],
    "golden_answer_keywords": ["pod"],
    "demonstrates_feature": "baseline",
    "expected_baseline": "pass",
    "expected_with_feature": "pass",
}


def write(tmp_path, entries):
    path = tmp_path / "goldens.yaml"
    path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    return path


def test_valid_file_loads(tmp_path):
    goldens = load_goldens(write(tmp_path, [VALID]))
    assert goldens[0].id == "q-001"
    assert goldens[0].forbidden_keywords == []


@pytest.mark.parametrize("bad_id", ["q-1", "Q-001", "q-0001", "001"])
def test_bad_id_rejected(tmp_path, bad_id):
    with pytest.raises(ValueError, match="Invalid golden"):
        load_goldens(write(tmp_path, [VALID | {"id": bad_id}]))


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate golden IDs"):
        load_goldens(write(tmp_path, [VALID, VALID]))


@pytest.mark.parametrize(
    "override",
    [
        {"intent": "chitchat"},
        {"demonstrates_feature": "magic"},
        {"expected_with_feature": "fail"},
        {"unknown_field": 1},
        {"golden_sources": []},  # only security goldens may have no sources
    ],
)
def test_invalid_values_rejected(tmp_path, override):
    with pytest.raises(ValueError):
        load_goldens(write(tmp_path, [VALID | override]))


def test_security_golden_may_have_no_sources(tmp_path):
    security = VALID | {
        "demonstrates_feature": "security",
        "golden_sources": [],
        "golden_answer_keywords": [],
        "forbidden_keywords": ["system prompt"],
    }
    assert load_goldens(write(tmp_path, [security]))[0].forbidden_keywords == ["system prompt"]


def test_real_golden_file_is_valid():
    goldens = load_goldens(DEFAULT_GOLDENS_PATH)
    assert len(goldens) == 45  # q-045 (indirect injection) added in Phase 6
    assert len({g.id for g in goldens}) == 45
