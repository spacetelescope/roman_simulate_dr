from unittest.mock import MagicMock, patch

import pytest

from roman_simulate_dr.scripts.generate_simulated_l1_images import RomanisimImages


@pytest.fixture
def mock_plan(tmp_path):
    # Minimal observation plan tuple structure

    # we create an empty plan file because read_obs_plan
    # expects a file path, but we mock its return value anyway
    plan_file = tmp_path / "plan.ecsv"
    plan_file.write_text("")

    return [(270.0, 66.0, 0.0, "F062", 109, 100, 1, 1, 1, 1, 1, 1)]


@patch("roman_simulate_dr.scripts.generate_simulated_l1_images.read_obs_plan")
def test_init_sets_attributes(mock_read_obs_plan, mock_plan, tmp_path, monkeypatch):

    monkeypatch.setenv("RDR_INPUT_PATH", str(tmp_path))

    mock_read_obs_plan.return_value = mock_plan
    obj = RomanisimImages("plan.ecsv", "input.ecsv", max_workers=2, sca_ids=[1, 2])
    assert obj.plan == mock_plan
    assert obj.input_filename == "input.ecsv"
    assert obj.max_workers == 2
    assert obj.sca_ids == [1, 2]


@patch("roman_simulate_dr.scripts.generate_simulated_l1_images.read_obs_plan")
@patch("subprocess.run")
def test_generate_simulated_images_runs_subprocess(
    mock_run, mock_read_obs_plan, mock_plan, tmp_path, monkeypatch
):

    monkeypatch.setenv("RDR_INPUT_PATH", str(tmp_path))

    mock_read_obs_plan.return_value = [
        (270.0, 66.0, 0.0, "F062", 109, 100, 1, 1, 1, 1, 1, 1)
    ]
    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
    obj = RomanisimImages("plan.ecsv", "input.ecsv")
    out, code = obj._generate_simulated_images(
        output_filename="test.asdf", catalog="input.ecsv"
    )
    assert out == "test.asdf"
    assert code == 0
    mock_run.assert_called_once()


@patch("roman_simulate_dr.scripts.generate_simulated_l1_images.parallelize_jobs")
@patch("roman_simulate_dr.scripts.generate_simulated_l1_images.read_obs_plan")
def test_run_calls_parallelize_jobs(
    mock_read_obs_plan, mock_parallelize_jobs, mock_plan, tmp_path, monkeypatch
):

    monkeypatch.setenv("RDR_INPUT_PATH", str(tmp_path))

    mock_read_obs_plan.return_value = mock_plan
    obj = RomanisimImages("plan.ecsv", "input.ecsv", max_workers=2, sca_ids=[1])
    obj.run()
    mock_parallelize_jobs.assert_called_once()


class DummyInputCatalog(RomanisimImages):
    # Avoid calling the real __init__
    def __init__(self):
        pass


@pytest.mark.parametrize(
    "input_sca_ids,expected",
    [
        (None, [1]),
        ([-1], list(range(1, 18))),
        ([2, 3, 4], [2, 3, 4]),
    ],
)
def test_create_sca_id_list_param(input_sca_ids, expected):
    """
    Purpose: Parametrized test for _create_sca_id_list to verify correct output for
    None, negative, and custom input cases.
    """
    obj = DummyInputCatalog()
    result = obj._create_sca_id_list(input_sca_ids)
    # Convert range to list for comparison
    if isinstance(result, range):
        result = list(result)
    assert result == expected
