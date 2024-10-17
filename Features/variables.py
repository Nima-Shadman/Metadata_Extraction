from scipy.stats import mode
import numpy as np


Punctuation = ["،", ".", "؛", "(", ")", "|", "«", "»", "؟", "?", ":", "/", "\\", "-"]

numbers = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "۰",
    "۱",
    "۲",
    "۳",
    "۴",
    "۵",
    "۶",
    "۷",
    "۸",
    "۹",
]

first_chapter_words = ["فصل", "اول", "مقدمه", "دیباچه", "پیشگفتار"]

months_seasons = [
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
    "بهار",
    "تابستان",
    "پاییز",
    "زمستان",
    "سال",
]

keyword = ["کلمات", "کلیدی", "واژه", "واژگان", "کلید", "واژهها"]

author = ["دانشجو", "پژوهشگر", "پژوهش", "نگارش", "تهیه", "تدوین", "نگارنده", "مولف"]

title = ["عنوان", "موضوع"]

professor = [
    "پروفسور",
    "استادان",
    "اساتید",
    "استاد",
    "راهنما",
    "مشاور",
    "دکتر",
    "مهندس",
    "حجتالاسلام",
]

professor_rate = ["", "", "", ""]

other_keywords = [
    "صورتجلسه",
    "تاییدیه",
    "دفاع",
    "سوگندنامه",
    "آییننامه",
    "تعهدنامه",
    "اظهارنامه",
]

acknowledgment = [
    "تقدیر",
    "تشکر",
    "سپاس",
    "قدردانی",
    "تقدیم",
    "مادر",
    "پدر",
    "خواهر",
    "برادر",
    "همسر",
    "دختر",
    "پسر",
    "فرزند",
]

toc = ["فهرست", "صفحه"]

university = [
    "دانشگاه",
    "پژوهشگاه",
    "موسسهیآموزشعالی",
    "موسسهآموزشعالی",
    "واحد",
    "مرکز",
]

faculty = [
    "دانشکده",
    "پژوهشکده",
    "آموزشکده",
    "پردیس",
]  # "مجتمعآموزشعالی","موسسهآموزشعالی"

parsa_type = ["پایاننامه", "رساله", "پروپوزال"]

field = ["رشته", "گرایش"]

degree = ["کارشناسی", "کارشناسیارشد", "دکتری", "دکترا"]

abstract = ["چکیده", "خلاصه"]

metadata_keywords = (
    months_seasons
    + keyword
    + author
    + title
    + professor
    + other_keywords
    + acknowledgment
    + toc
    + university
    + faculty
    + parsa_type
    + field
    + degree
    + abstract
)

boundary_labels = {"metadata": 0, "body": 1}

metadata_labels_10 = {
    "other": 0,
    "type": 0,
    "department": 0,
    "toc": 0,
    "university": 1,
    "faculty": 2,
    "degree & field": 3,
    "date": 4,
    "title": 5,
    "professor": 6,
    "author": 7,
    "abstract": 8,
    "keywords": 9,
}  # 10 classes


def font_size(default, paragraph):
    # print(type(paragraph))
    if type(paragraph) == list:

        font_list = []
        for para in paragraph:
            fonts = []
            for run in para.runs:
                if run.text != "":
                    if run.font.size == None:
                        fonts.append(default)
                    else:
                        fonts.append(run.font.size.pt)
            if len(fonts) == 0:
                font = default
            else:
                font = mode(np.array(fonts))[0]
            font_list.append(font)
        return font_list
    else:
        fonts = []
        runs = [run for run in paragraph.runs if run.text.strip()]
        if runs:
            for run in paragraph.runs:
                if run.text != "":
                    if run.font.size == None:
                        fonts.append(default)
                    else:
                        fonts.append(run.font.size.pt)
        else:
            try:
                fonts.append(paragraph.style.font.size.pt)
            except AttributeError:
                fonts.append(default)
        font = mode(np.array(fonts))[0]
        return font


def is_bold(para):
    bold = []
    for run in para.runs:
        if run.text != "":
            if run.bold:
                bold.append(1)
            else:
                bold.append(0)
    if len(bold) == 0:
        para_bold = 0
    else:
        para_bold = mode(np.array(bold))[0]
    return para_bold


def is_italic(para):
    italic = []
    for run in para.runs:
        if run.text != "":
            if run.italic:
                italic.append(1)
            else:
                italic.append(0)
    if len(italic) == 0:
        para_italic = 0
    else:
        para_italic = mode(np.array(italic))[0]
    return para_italic


def font_name(para, font_default):
    font = []
    for run in para.runs:
        if run.text != "":
            if run.font.name == None:
                font.append(font_default)
            else:
                font.append(run.font.name)
    if len(font) == 0:
        para_font_name = font_default
    else:
        unique, counts = np.unique(font, return_counts=True)
        para_font_name = unique[np.argmax(counts)]
    return para_font_name
