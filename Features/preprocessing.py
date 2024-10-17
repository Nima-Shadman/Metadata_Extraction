import string
from Features.variables import metadata_keywords, months_seasons
import re
from functools import reduce

PUNCTUATIONS = string.punctuation + "–"
NUMERICS = "0123456789٠۰١۱٢۲٣۳٤۴٥۵٦۶٧۷٨۸٩۹"


def basic_preproccesing(text: str):
    remove_chars = [
        "ـ",
        "\u0651",
        "\u064B",
        "\u064D",
        "\u064C",
        "\u0621",
        "\u064E",
        "\u064F",
        "\u0650",
        "\u0652",
        " ۗ",
        " ۚ",
        "\u06D5",
        "ٰ",
        "ٓ",
        "ۭ",
    ]
    replace_chars = {
        "\u0643": "ک",
        "ڪ": "ک",
        "\u064A": "ی",
        "ى": "ی",
        "ے": "ی",
        "أ": "ا",
        "إ": "ا",
        "ة": "ه",
        "ؤ": "و",
        "ﺓ": "ه",
        "٠": "۰",
        "١": "۱",
        "٢": "۲",
        "٣": "۳",
        "٤": "۴",
        "٥": "۵",
        "٦": "۶",
        "٧": "۷",
        "٨": "۸",
        "٩": "۹",
        "0": "۰",
        "1": "۱",
        "2": "۲",
        "3": "۳",
        "4": "۴",
        "5": "۵",
        "6": "۶",
        "7": "۷",
        "8": "۸",
        "9": "۹",
    }

    for char in remove_chars:
        if char in text:
            text = text.replace(char, "")

    for char in replace_chars.keys():
        if char in text:
            text = text.replace(char, replace_chars[char])

    return text


def remove_duplicates(blocks: list):
    unique_paragraphs = []
    unique_texts = []
    filtered_blocks = []
    for block in blocks:
        for paragraph in block:
            cleaned_text = basic_preproccesing(
                paragraph.text.strip().replace("\u200c", "")
            )
            if cleaned_text not in unique_texts:
                unique_paragraphs.append(paragraph)
                unique_texts.append(cleaned_text)
        if len(unique_paragraphs) != 0:
            filtered_blocks.append(unique_paragraphs)
            unique_paragraphs = []

    return filtered_blocks


def remove_short_paragraphs(blocks, threshold=3):
    c = 0
    long_paragraphs = []
    filtered_blocks = []
    for block in blocks:
        for paragraph in block:
            cleaned_text = basic_preproccesing(
                paragraph.text.strip().replace("\u200c", "")
            )
            if len(cleaned_text) <= threshold and cleaned_text not in metadata_keywords:
                # print(get_display(arabic_reshaper.reshape(paragraph)))
                c += 1
            else:
                long_paragraphs.append(paragraph)
        if len(long_paragraphs) != 0:
            filtered_blocks.append(long_paragraphs)
            long_paragraphs = []

    return filtered_blocks


def remove_numerical_punctuation_ratio(blocks, threshold=0.5):
    saved_paragraphs = []
    filtered_blocks = []
    for block in blocks:
        for paragraph in block:
            cleaned_text = basic_preproccesing(
                paragraph.text.strip().replace("\u200c", "")
            )
            count = 0
            for char in cleaned_text:
                if char in PUNCTUATIONS + NUMERICS:
                    count += 1
            ratio = count / len(cleaned_text)
            is_persian_year = (
                cleaned_text.strip(PUNCTUATIONS).isdigit()
                and int(cleaned_text.strip(PUNCTUATIONS)) <= 1450
                and int(cleaned_text.strip(PUNCTUATIONS)) >= 1350
            )
            if (
                ratio < threshold
                or any(month in cleaned_text for month in months_seasons)
                or is_persian_year
            ):
                saved_paragraphs.append(paragraph)
        if len(saved_paragraphs) != 0:
            filtered_blocks.append(saved_paragraphs)
            saved_paragraphs = []
    return filtered_blocks


def remove_regex_matches(blocks):
    pattern = r"^(شکل|نمودار|جدول)?\s*\d+([-.]\d+)+[-.:]?(\s)*[\u0600-\u06FF\sA-Za-z]+[-.\s]*\d+\s*$"
    removal = 0
    saved_paragraphs = []
    filtered_blocks = []
    for block in blocks:
        for paragraph in block:
            cleaned_text = basic_preproccesing(paragraph.text.replace("\u200c", ""))
            if not re.match(pattern, cleaned_text):
                saved_paragraphs.append(paragraph)

        if len(saved_paragraphs) != 0:
            filtered_blocks.append(saved_paragraphs)
            saved_paragraphs = []

    return filtered_blocks


def metadata_preprocessing(document):
    processing_steps = [
        remove_duplicates,
        remove_short_paragraphs,
        remove_numerical_punctuation_ratio,
        remove_regex_matches,
    ]

    return reduce(lambda doc, func: func(doc), processing_steps, document)
