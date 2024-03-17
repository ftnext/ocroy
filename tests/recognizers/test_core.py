from pathlib import Path
from unittest.mock import MagicMock, patch

from ocroy.recognizers.base import ContentRecognizable
from ocroy.recognizers.core import OcrRecognizer, OcrRequest


class TestOcrRecognizer:
    @patch("ocroy.recognizers.core.read_image")
    def test_can_recognize(self, read_image: MagicMock) -> None:
        recognizable = MagicMock(spec=ContentRecognizable)
        sut = OcrRecognizer(recognizable)

        request = OcrRequest(Path("path/to/image.png"))
        actual = sut(request)

        assert actual == recognizable.recognize.return_value
        read_image.assert_called_once_with(Path("path/to/image.png"))
        recognizable.recognize.assert_called_once_with(read_image.return_value)
