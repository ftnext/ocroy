from unittest.mock import patch

from ocroy.recognizers.tesseract import recognize


@patch("ocroy.recognizers.tesseract.RemoveWhitespaceNormalizer")
@patch("PIL.Image")
@patch("pytesseract.image_to_string")
def test_recognize(image_to_string, Image, RemoveWhitespaceNormalizer):
    image = Image.open.return_value
    normalizer = RemoveWhitespaceNormalizer.return_value

    actual = recognize("path/to/image.png")

    assert actual == normalizer.normalize.return_value
    Image.open.assert_called_once_with("path/to/image.png")
    image_to_string.assert_called_once_with(image, lang="jpn")
    RemoveWhitespaceNormalizer.assert_called_once_with()
    normalizer.normalize.assert_called_once_with(image_to_string.return_value)
