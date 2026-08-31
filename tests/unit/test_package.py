"""Package import tests."""

import contextos


def test_package_version() -> None:
    assert contextos.__version__ == "0.3.0"
