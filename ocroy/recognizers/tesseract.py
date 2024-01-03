from __future__ import annotations

from pathlib import Path

from ocroy.normalize import RemoveWhitespaceNormalizer


def recognize(image_path: str | Path) -> str:
    import pytesseract
    from PIL import Image

    image = Image.open(image_path)
    result = pytesseract.image_to_string(image, lang="jpn")

    normalizer = RemoveWhitespaceNormalizer()
    return normalizer.normalize(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    args = parser.parse_args()

    text = recognize(args.image_path)

    print(text)
