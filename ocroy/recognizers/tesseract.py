from io import BytesIO

from ocroy.normalize import RemoveWhitespaceNormalizer


class ImageRecognizer:
    def recognize(self, content: bytes) -> str:
        import pytesseract
        from PIL import Image

        image = Image.open(BytesIO(content))
        result = pytesseract.image_to_string(image, lang="jpn")

        normalizer = RemoveWhitespaceNormalizer()
        return normalizer.normalize(result)


def recognize(content: bytes) -> str:
    import pytesseract
    from PIL import Image

    image = Image.open(BytesIO(content))
    result = pytesseract.image_to_string(image, lang="jpn")

    normalizer = RemoveWhitespaceNormalizer()
    return normalizer.normalize(result)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from ocroy.reader import read_image

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    args = parser.parse_args()

    content = read_image(args.image_path)
    text = recognize(content)

    print(text)
