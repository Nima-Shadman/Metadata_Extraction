from docx import Document
from Features.preprocessing import basic_preproccesing
import csv
import os
from Features.preprocessing import metadata_preprocessing
from metadata_classification.features_for_meta_classification import (
    read_boundary,
    get_blocks,
    cut_blocks,
)
import json
import arabic_reshaper
from bidi.algorithm import get_display
import string
import re


def abstract_postprocessing(abstracts):
    complete_abstract = []
    for abstract in abstracts:
        cleaned_abstract = "".join(
            char for char in abstract if char not in [" "] + list(string.punctuation)
        )
        if (
            not cleaned_abstract.startswith(tuple(["چکیده", "خلاصه"]))
            and len(cleaned_abstract) > 10
        ):
            complete_abstract.append(abstract)
    return "\n".join(complete_abstract)


def keywords_postprocessing(keywords):
    complete_keywords = " ".join(keywords).replace("\u200c", " ")
    if ":" in complete_keywords:
        complete_keywords = complete_keywords.split(":")[1]
    return complete_keywords


def supervisor_advisor_postprocessing(professors):
    supervisors = []
    advisors = []
    is_advisor = False
    is_supervisor = False
    for professor in professors:
        if is_supervisor and "مشاور" not in professor:
            supervisors.append(professor)
        elif is_advisor and "راهنما" not in professor:
            advisors.append(professor)
        if "راهنما" in professor:
            is_supervisor = True
            is_advisor = False
        if "مشاور" in professor:
            is_advisor = True
            is_supervisor = False
    return "، ".join(supervisors) + "\n------\n" + "، ".join(advisors)


def student_postprocessing(students):
    complete_students = " ".join(students).replace("\u200c", " ")
    if ":" in complete_students:
        split = complete_students.split(":")
        if len(split[1].strip()) != 0:
            complete_students = split[1]
    elif len(students) == 2:
        complete_students = students[1]

    return complete_students


def title_postprocessing(titles):
    complete_titles = (
        " ".join(titles).replace("\u200c", " ").replace(":", "").replace("\n", " ")
    )
    for keyword in [
        "عنوان فارسی",
        "عنوان پایان نامه",
        "موضوع پایان نامه",
        "عنوان پایاننامه",
        "موضوع پایاننامه",
        "عنوان رساله",
        "موضوع رساله",
        "موضوع فارسی",
        "عنوان",
        "موضوع",
    ]:
        if complete_titles.strip().startswith(keyword):
            complete_titles = complete_titles.lstrip(keyword)
    return complete_titles


def date_postprocessing(dates):
    complete_dates = (
        " ".join(dates).replace("\u200c", " ").replace(":", "").replace("\n", " ")
    )
    return complete_dates


def faculty_postprocessing(faculties):
    complete_faculties = (
        " ".join(faculties).replace("\u200c", " ").replace(":", "").replace("\n", " ")
    )
    return complete_faculties


def university_postprocessing(universities):
    complete_universities = (
        " ".join(universities)
        .replace("\u200c", " ")
        .replace(":", "")
        .replace("\n", " ")
    )
    return complete_universities


def degree_field_postprocessing(degrees_fields):

    def cut_string_after_keyword(text, keyword):
        return text.split(keyword)[1]

    complete_degrees_fields = (
        " ".join(degrees_fields)
        .replace("\u200c", " ")
        .replace(":", "")
        .replace("\n", " ")
    )
    complete_degrees_fields = re.sub(r"\s+", " ", complete_degrees_fields)
    org = complete_degrees_fields
    degree = ""
    degrees = ["دکتری", "دکترای", "دکترا", "کارشناسی ارشد", "کارشناسیارشد", "کارشناسی"]
    for i in range(len(degrees)):
        if degrees[i] in complete_degrees_fields:
            complete_degrees_fields = complete_degrees_fields.replace(degrees[i], "")
            if i < 3:
                degree = "دکتری"
            elif i < 5:
                degree = "کارشناسی ارشد"
            else:
                degree = "کارشناسی"

    found_field = False

    for field_keyword in ["رشتهی", "رشته ی", "رشته"]:
        if field_keyword in complete_degrees_fields:
            field = cut_string_after_keyword(complete_degrees_fields, field_keyword)
            found_field = True
            break

    if not found_field:
        for keyword in [
            "جهت",
            "اخذ",
            "مدرک",
            "پایاننامه",
            "پایان نامه",
            "رساله",
            "درجه ی",
            "درجهی",
            "درجه",
            "رشتهی",
            "رشته ی",
            "رشته",
            "برای",
            "دریافت",
            " در ",
            "دورهی",
            "دوره ی",
            "دوره",
        ]:
            if keyword in complete_degrees_fields:
                complete_degrees_fields = complete_degrees_fields.replace(keyword, "")
        field = complete_degrees_fields

    return f"{org}:\n{degree}\n{field}"
