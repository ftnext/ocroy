from unittest.mock import patch

from ocroy.recognizers.tesseract import recognize


@patch("PIL.Image")
@patch("pytesseract.image_to_string")
def test_recognize(image_to_string, Image):
    image = Image.open.return_value

    actual = recognize("path/to/image.png")

    assert actual == image_to_string.return_value
    Image.open.assert_called_once_with("path/to/image.png")
    image_to_string.assert_called_once_with(image, lang="jpn")
