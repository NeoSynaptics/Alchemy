"""Voice test configuration — mock webrtcvad for Python 3.12+ compatibility."""

import sys
from unittest.mock import MagicMock

# webrtcvad uses pkg_resources which is broken on Python 3.12+.
# Mock it before any voice module imports it.
if "webrtcvad" not in sys.modules:
    mock_webrtcvad = MagicMock()
    mock_vad_instance = MagicMock()
    mock_vad_instance.is_speech = MagicMock(return_value=False)
    mock_webrtcvad.Vad = MagicMock(return_value=mock_vad_instance)
    sys.modules["webrtcvad"] = mock_webrtcvad
