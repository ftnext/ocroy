def recognize(content: bytes, *, as_document: bool) -> str:
    from google.cloud import vision

    image = vision.Image(content=content)
    client = vision.ImageAnnotatorClient()
    if as_document:
        response = client.document_text_detection(image=image)
    else:
        response = client.text_detection(image=image)
    annotations = response.text_annotations
    return annotations[0].description


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from ocroy.reader import read_image

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--as-document", action="store_true")
    args = parser.parse_args()

    content = read_image(args.image_path)
    text = recognize(content, as_document=args.as_document)

    print(text)
