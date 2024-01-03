from pathlib import Path


def recognize(image_path: str | Path) -> str:
    import pytesseract
    from PIL import Image

    image = Image.open(image_path)
    return pytesseract.image_to_string(image, lang="jpn")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    args = parser.parse_args()

    text = recognize(args.image_path)

    print(text)
