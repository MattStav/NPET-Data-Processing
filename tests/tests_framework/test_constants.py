from NPET_DP.framework import constants


def test_app_name() -> None:
    """Test the App name."""
    assert constants.APP_NAME == "NPET_Data_Processing"
    assert isinstance(constants.APP_NAME, str)


def test_package_name() -> None:
    """Test the package name."""
    assert constants.PACKAGE_NAME == "NPET_DP"
    assert isinstance(constants.PACKAGE_NAME, str)


def test_appdata_dir_name() -> None:
    """Test the dirname where this app stores generated data.

    This name must match the one used by NPET_communication_FW; it must NOT change!
    """
    assert constants.APPDATA_DIR_NAME == "NPET"
    assert isinstance(constants.APPDATA_DIR_NAME, str)


def test_femto() -> None:
    """Test the femto constant."""
    assert constants.FEMTO == 1_000_000_000_000_000
    assert isinstance(constants.FEMTO, int)
