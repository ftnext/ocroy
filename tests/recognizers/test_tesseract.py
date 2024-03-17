from unittest.mock import MagicMock, patch

from ocroy.recognizers.core import OcrRequest
from ocroy.recognizers.tesseract import ImageRecognizer, recognize

MODULE_UNDER_TEST = "ocroy.recognizers.tesseract"


@patch(f"{MODULE_UNDER_TEST}.BytesIO")
@patch(f"{MODULE_UNDER_TEST}.RemoveWhitespaceNormalizer")
@patch("PIL.Image")
@patch("pytesseract.image_to_string")
class TestImageRecognizer:
    def test_can_recognize(
        self,
        image_to_string: MagicMock,
        Image: MagicMock,
        RemoveWhitespaceNormalizer: MagicMock,
        BytesIO: MagicMock,
    ) -> None:
        image = Image.open.return_value
        normalizer = RemoveWhitespaceNormalizer.return_value
        sut = ImageRecognizer()

        image_content = MagicMock(spec=bytes)
        actual = sut.recognize(image_content)

        assert actual == normalizer.normalize.return_value
        BytesIO.assert_called_once_with(image_content)
        Image.open.assert_called_once_with(BytesIO.return_value)
        image_to_string.assert_called_once_with(image, lang="jpn")
        RemoveWhitespaceNormalizer.assert_called_once_with()
        normalizer.normalize.assert_called_once_with(
            image_to_string.return_value
        )


@patch(f"{MODULE_UNDER_TEST}.OcrRecognizer")
@patch(f"{MODULE_UNDER_TEST}.ImageRecognizer")
def test_recognize(
    ImageRecognizer: MagicMock, OcrRecognizer: MagicMock
) -> None:
    tesseract_recognizer = ImageRecognizer.return_value
    recognizer = OcrRecognizer.return_value
    request = MagicMock(spec=OcrRequest)

    actual = recognize(request)

    assert actual == recognizer.return_value
    ImageRecognizer.assert_called_once_with()
    OcrRecognizer.assert_called_once_with(tesseract_recognizer)
    recognizer.assert_called_once_with(request)
