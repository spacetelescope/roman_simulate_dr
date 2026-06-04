from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from astropy import units as u

from roman_simulate_dr.scripts.generate_input_catalog import InputCatalog


@pytest.fixture
def mock_plan(tmp_path):
    # Minimal plan with RA and DEC columns

    # we create an empty plan file because read_obs_plan 
    # expects a file path, but we mock its return value anyway
    plan_file = tmp_path / "plan.ecsv"
    plan_file.write_text("")

    return {"RA": [10.0, 20.0], "DEC": [30.0, 40.0]}


ROMAN_PHOTOZ_FLUX_PATH = (
    Path(__file__).parent / "../data/roman_simulated_catalog.parquet"
).resolve().as_posix()


@patch("roman_simulate_dr.scripts.generate_input_catalog.read_obs_plan")
def test_init_sets_attributes(mock_read_obs_plan, mock_plan, tmp_path, monkeypatch):
    """
    Purpose: Verify that InputCatalog initializes its attributes correctly
    when provided with explicit arguments and a mocked observation plan.
    """

    monkeypatch.setenv("RDR_INPUT_PATH", str(tmp_path))

    mock_read_obs_plan.return_value = mock_plan
    obj = InputCatalog(
        "plan.ecsv",
        output_catalog_filename="out.ecsv",
        ra=1.0,
        dec=2.0,
        radius=0.5,
        flux_catalog_filename=ROMAN_PHOTOZ_FLUX_PATH,
    )
    assert obj.plan == mock_plan
    assert str(obj.catalog_filename) == "out.ecsv"
    assert obj.ra == 1.0
    assert obj.dec == 2.0
    assert obj.radius == 0.5
    assert str(obj.flux_catalog_filename) == ROMAN_PHOTOZ_FLUX_PATH


@patch("roman_simulate_dr.scripts.generate_input_catalog.vstack")
@patch("roman_simulate_dr.scripts.generate_input_catalog.make_stars")
@patch("roman_simulate_dr.scripts.generate_input_catalog.make_gaia_stars")
@patch("roman_simulate_dr.scripts.generate_input_catalog.make_cosmos_galaxies")
@patch("roman_simulate_dr.scripts.generate_input_catalog.read_obs_plan")
def test_generate_catalog_calls_components(
    mock_read_obs_plan,
    mock_make_cosmos_galaxies,
    mock_make_gaia_stars,
    mock_make_stars,
    mock_vstack,
    mock_plan,
    tmp_path,
    monkeypatch,
):
    """
    Purpose: Ensure that _generate_catalog calls all required component
    functions and writes the catalog using the correct filename and format.
    """
    import numpy as np
    from astropy.table import Table

    monkeypatch.setenv("RDR_INPUT_PATH", str(tmp_path))

    mock_read_obs_plan.return_value = mock_plan
    mock_make_cosmos_galaxies.return_value = MagicMock()
    mock_make_gaia_stars.return_value = MagicMock()
    mock_make_stars.return_value = MagicMock()
    # Create a mock catalog with the required column and at least one row
    mock_catalog = Table()
    mock_catalog["F213"] = np.array([1.0])
    mock_vstack.return_value = mock_catalog
    obj = InputCatalog(
        "plan.ecsv",
        output_catalog_filename="out.ecsv",
        flux_catalog_filename=ROMAN_PHOTOZ_FLUX_PATH,
    )
    obj._generate_catalog(filter_list=["f062"])


@patch.object(InputCatalog, "_generate_catalog")
@patch("roman_simulate_dr.scripts.generate_input_catalog.read_obs_plan")
def test_run_calls_generate_catalog(
    mock_read_obs_plan, mock_generate_catalog, mock_plan, tmp_path, monkeypatch
):
    """
    Purpose: Verify that the run() method of InputCatalog triggers
    _generate_catalog exactly once.
    """

    monkeypatch.setenv("RDR_INPUT_PATH", str(tmp_path))

    mock_read_obs_plan.return_value = mock_plan
    obj = InputCatalog(
        "plan.ecsv",
        output_catalog_filename="out.ecsv",
        flux_catalog_filename=ROMAN_PHOTOZ_FLUX_PATH,
    )
    obj.run()
    mock_generate_catalog.assert_called_once()


@patch("roman_simulate_dr.scripts.generate_input_catalog.Table")
@patch("roman_simulate_dr.scripts.generate_input_catalog.logger")
@patch("roman_simulate_dr.scripts.generate_input_catalog.read_obs_plan")
def test_update_catalog_fluxes_success(
    mock_read_obs_plan, mock_logger, mock_table, mock_plan, tmp_path, monkeypatch
):
    """
    Purpose: Test update_catalog_fluxes when everything succeeds.
    """
    import numpy as np
    from astropy.table import Table

    monkeypatch.setenv("RDR_INPUT_PATH", str(tmp_path))

    mock_read_obs_plan.return_value = mock_plan
    obj = InputCatalog(
        "plan.ecsv",
        output_catalog_filename="out.ecsv",
        flux_catalog_filename=ROMAN_PHOTOZ_FLUX_PATH,
    )
    fake_catalog = MagicMock()
    fake_catalog.copy.return_value = fake_catalog
    # Provide a non-empty fake_flux_catalog with required column
    fake_flux_catalog = Table()
    fake_flux_catalog["segment_f213_flux"] = np.array([1.0]) * u.mgy
    fake_flux_catalog["label"] = np.array([1])
    fake_flux_catalog["redshift_true"] = np.array([1.0])
    mock_table.read.return_value = fake_flux_catalog
    # Patch at the source module since update_catalog_fluxes uses a local import
    with patch(
        "roman_photoz.update_romanisim_catalog_fluxes.update_fluxes"
    ) as mock_update_fluxes, patch(
        "roman_photoz.update_romanisim_catalog_fluxes.create_random_catalog"
    ) as mock_create_random:
        mock_create_random.return_value = fake_flux_catalog
        mock_update_fluxes.return_value = fake_catalog
        result = obj.update_catalog_fluxes(fake_catalog)
        assert result is fake_catalog


@patch("roman_simulate_dr.scripts.generate_input_catalog.read_obs_plan")
def test_update_catalog_fluxes_no_flux_catalog(mock_read_obs_plan, mock_plan, tmp_path, monkeypatch):
    """
    Purpose: Test update_catalog_fluxes raises if no flux_catalog_filename is set.
    """

    monkeypatch.setenv("RDR_INPUT_PATH", str(tmp_path))

    mock_read_obs_plan.return_value = mock_plan
    obj = InputCatalog("plan.ecsv", output_catalog_filename="out.ecsv")
    with pytest.raises(RuntimeError):
        obj.update_catalog_fluxes(MagicMock())
