# ref: https://nikkie-ftnext.hatenablog.com/entry/remove-whitespace-in-text-with-regex  # NOQA: E501
import re


class RemoveWhitespaceNormalizer:
    """
    >>> normalizer = RemoveWhitespaceNormalizer()
    >>> normalizer.normalize("アルゴリズム C")
    'アルゴリズムC'
    >>> normalizer.normalize("アルゴ B リズム C")
    'アルゴBリズムC'
    >>> normalizer.normalize("アイ の 歌声 を 聴か せ て")
    'アイの歌声を聴かせて'
    >>> normalizer.normalize("ういっす ういっす ういっすー✌️")
    'ういっすういっすういっすー✌️'
    >>> normalizer.normalize("検索 エンジン 自作 入門 を 買い ました ！！！")
    '検索エンジン自作入門を買いました！！！'
    >>> normalizer.normalize("Algorithm C")
    'Algorithm C'
    >>> normalizer.normalize("Coding the Matrix")
    'Coding the Matrix'
    """

    basic_latin = "\u0000-\u007F"
    blocks = "".join(
        (
            "\u4E00-\u9FFF",  # CJK UNIFIED IDEOGRAPHS
            "\u3040-\u309F",  # HIRAGANA
            "\u30A0-\u30FF",  # KATAKANA
            "\u3000-\u303F",  # CJK SYMBOLS AND PUNCTUATION
            "\uFF00-\uFFEF",  # HALFWIDTH AND FULLWIDTH FORMS
        )
    )

    def __init__(self) -> None:
        pattern1 = re.compile("([{}]) ([{}])".format(self.blocks, self.blocks))
        pattern2 = re.compile(
            "([{}]) ([{}])".format(self.blocks, self.basic_latin)
        )
        pattern3 = re.compile(
            "([{}]) ([{}])".format(self.basic_latin, self.blocks)
        )
        self.patterns = (pattern1, pattern2, pattern3)

    def normalize(self, text: str) -> str:
        for pattern in self.patterns:
            while pattern.search(text):
                text = pattern.sub(r"\1\2", text)
        return text
