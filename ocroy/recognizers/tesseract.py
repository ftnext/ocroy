from pathlib import Path


def recognize(image_path: str | Path) -> str:
    import pytesseract
    from PIL import Image

    image = Image.open(image_path)
    return pytesseract.image_to_string(image, lang="jpn")
