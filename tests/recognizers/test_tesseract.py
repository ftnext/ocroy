from unittest.mock import MagicMock, patch

from ocroy.recognizers.tesseract import recognize


@patch("ocroy.recognizers.tesseract.BytesIO")
@patch("ocroy.recognizers.tesseract.RemoveWhitespaceNormalizer")
@patch("PIL.Image")
@patch("pytesseract.image_to_string")
def test_recognize(
    image_to_string, Image, RemoveWhitespaceNormalizer, BytesIO
):
    image_content = MagicMock(spec=bytes)
    image = Image.open.return_value
    normalizer = RemoveWhitespaceNormalizer.return_value

    actual = recognize(image_content)

    assert actual == normalizer.normalize.return_value
    BytesIO.assert_called_once_with(image_content)
    Image.open.assert_called_once_with(BytesIO.return_value)
    image_to_string.assert_called_once_with(image, lang="jpn")
    RemoveWhitespaceNormalizer.assert_called_once_with()
    normalizer.normalize.assert_called_once_with(image_to_string.return_value)
