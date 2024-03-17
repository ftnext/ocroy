from unittest.mock import MagicMock, patch

import pytest
from google.cloud import vision

from ocroy.recognizers.core import OcrRequest
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


@patch("ocroy.recognizers.google_vision_api.OcrRecognizer")
@patch("ocroy.recognizers.google_vision_api.GoogleVisionApiRecognizer")
def test_recognize(
    GoogleVisionApiRecognizer: MagicMock, OcrRecognzier: MagicMock
) -> None:
    google_recognizer = GoogleVisionApiRecognizer.return_value
    recognizer = OcrRecognzier.return_value
    request = MagicMock(spec=OcrRequest)
    handle_document = MagicMock(spec=bool)

    actual = recognize(request, handle_document=handle_document)

    assert actual == recognizer.return_value
    GoogleVisionApiRecognizer.assert_called_once_with(
        handle_document=handle_document
    )
    OcrRecognzier.assert_called_once_with(google_recognizer)
    recognizer.assert_called_once_with(request)
