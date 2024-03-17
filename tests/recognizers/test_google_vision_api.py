from unittest.mock import MagicMock, patch

import pytest
from google.cloud import vision

from ocroy.recognizers.google_vision_api import (
    DocumentRecognizer,
    GoogleVisionApiRecognizer,
    ImageRecognizer,
    recognize,
)


@pytest.fixture
def client() -> vision.ImageAnnotatorClient:
    return MagicMock(spec=vision.ImageAnnotatorClient)


class TestImageRecognizer:
    @patch("google.cloud.vision.Image")
    def test_can_recognize(
        self, Image: MagicMock, client: vision.ImageAnnotatorClient
    ) -> None:
        response = client.text_detection.return_value
        sut = ImageRecognizer(client)

        image_content = MagicMock(spec=bytes)
        actual = sut.recognize(image_content)

        assert actual == response.text_annotations[0].description
        Image.assert_called_once_with(content=image_content)
        client.text_detection.assert_called_once_with(image=Image.return_value)


class TestDocumentRecognizer:
    @patch("google.cloud.vision.Image")
    def test_can_recognize(
        self, Image: MagicMock, client: vision.ImageAnnotatorClient
    ) -> None:
        response = client.document_text_detection.return_value
        sut = DocumentRecognizer(client)

        image_content = MagicMock(spec=bytes)
        actual = sut.recognize(image_content)

        assert actual == response.text_annotations[0].description
        Image.assert_called_once_with(content=image_content)
        client.document_text_detection.assert_called_once_with(
            image=Image.return_value
        )


@patch("google.cloud.vision.ImageAnnotatorClient")
class TestGoogleVisionApiRecognizer:
    def test_create_image_recognizer(
        self, ImageAnnotatorClient: MagicMock
    ) -> None:
        actual = GoogleVisionApiRecognizer(handle_document=False)

        assert isinstance(actual, ImageRecognizer)
        assert actual.client == ImageAnnotatorClient.return_value
        ImageAnnotatorClient.assert_called_once_with()

    def test_create_document_recognizer(
        self, ImageAnnotatorClient: MagicMock
    ) -> None:
        actual = GoogleVisionApiRecognizer(handle_document=True)

        assert isinstance(actual, DocumentRecognizer)
        assert actual.client == ImageAnnotatorClient.return_value
        ImageAnnotatorClient.assert_called_once_with()


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
