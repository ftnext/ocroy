from pathlib import Path


def recognize(image_path: str | Path) -> str:
    from google.cloud import vision

    with open(image_path, "rb") as fb:
        content = fb.read()
    image = vision.Image(content=content)
    client = vision.ImageAnnotatorClient()
    response = client.document_text_detection(image=image)
    annotations = response.text_annotations
    return annotations[0].description
