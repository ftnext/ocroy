from unittest.mock import MagicMock, patch

from ocroy.recognizers.google_vision_api import recognize


@patch("google.cloud.vision.Image")
@patch("google.cloud.vision.ImageAnnotatorClient")
def test_recognize_as_document(ImageAnnotatorClient, Image):
    image_content = MagicMock(spec=bytes)
    image = Image.return_value
    client = ImageAnnotatorClient.return_value
    response = client.document_text_detection.return_value

    actual = recognize(image_content, as_document=True)

    assert actual == response.text_annotations[0].description
    Image.assert_called_once_with(content=image_content)
    ImageAnnotatorClient.assert_called_once_with()
    client.document_text_detection.assert_called_once_with(image=image)


@patch("google.cloud.vision.Image")
@patch("google.cloud.vision.ImageAnnotatorClient")
def test_recognize_not_document(ImageAnnotatorClient, Image):
    image_content = MagicMock(spec=bytes)
    image = Image.return_value
    client = ImageAnnotatorClient.return_value
    response = client.text_detection.return_value

    actual = recognize(image_content, as_document=False)

    assert actual == response.text_annotations[0].description
    Image.assert_called_once_with(content=image_content)
    ImageAnnotatorClient.assert_called_once_with()
    client.text_detection.assert_called_once_with(image=image)
