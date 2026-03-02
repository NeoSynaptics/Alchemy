"""Alchemy test configuration."""

import pytest


@pytest.fixture
def display_num():
    """Default display number for shadow desktop tests."""
    return 99


@pytest.fixture
def vnc_port():
    """Default VNC port."""
    return 5900


@pytest.fixture
def novnc_port():
    """Default noVNC port."""
    return 6080
