from __future__ import annotations

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    args = parser.parse_args()

    text = recognize(args.image_path)

    print(text)
