from unittest.mock import mock_open, patch

from ocroy.recognizers.google_vision_api import recognize


@patch("google.cloud.vision.Image")
@patch("google.cloud.vision.ImageAnnotatorClient")
def test_recognize(ImageAnnotatorClient, Image):
    image = Image.return_value
    client = ImageAnnotatorClient.return_value
    response = client.document_text_detection.return_value

    with patch(
        "ocroy.recognizers.google_vision_api.open",
        mock_open(read_data=b"image_content"),
    ) as m:
        actual = recognize("path/to/image.png")

    assert actual == response.text_annotations[0].description
    m.assert_called_once_with("path/to/image.png", "rb")
    Image.assert_called_once_with(content=b"image_content")
    ImageAnnotatorClient.assert_called_once_with()
    client.document_text_detection.assert_called_once_with(image=image)
