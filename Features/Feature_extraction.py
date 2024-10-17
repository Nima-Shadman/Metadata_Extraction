from PIL import ImageFont
from . import variables
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
import numpy as np


class Features:

    def __init__(
        self,
        para,
        text,
        count,
        para_font,
        pre_font_size,
        line_width,
        default_font_size,
        pre_label,
        two_pre_label,
        default_line_spacing,
        pre_para_italic=None,
        para_italic=None,
        pre_para_bold=None,
        para_bold=None,
        para_font_name=None,
        pre_font_name=None,
        space_count=None,
        image_flag=None,
        block_word_count=None,
        block_number=None,
        is_max_font=None,
        para_in_block_idx=None,
        block_other_count=None,
        block_meta_count=None,
        title_count=None,
        super_count=None,
        student_count=None,
    ):

        self.paragraph = para
        self.text = text
        self.count = count
        self.para_font = para_font
        self.pre_font_size = pre_font_size
        self.para_italic = para_italic
        self.pre_para_italic = pre_para_italic
        self.para_bold = para_bold
        self.pre_para_bold = pre_para_bold
        self.line_width = line_width
        self.default_font_size = default_font_size
        self.pre_label = pre_label
        self.two_pre_label = two_pre_label
        self.default_line_spacing = default_line_spacing
        self.line_length = 0
        self.para_font_name = para_font_name
        self.pre_font_name = pre_font_name
        self.space_count = space_count
        self.image_flag = image_flag
        self.line_count = 0
        self.block_word_count = block_word_count
        self.block_number = block_number
        self.is_max_font = is_max_font
        self.para_in_block_idx = para_in_block_idx
        self.block_other_count = block_other_count
        self.block_meta_count = block_meta_count
        self.title_count = title_count
        self.super_count = super_count
        self.student_count = student_count

    def abstract_feature(self):  # lexical feature

        abstract = variables.abstract
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        count = 0
        for word in abstract:
            if word in text:
                count += 1

        return count

    def acknowledgment_feature(self):  # lexical feature

        professor = variables.acknowledgment
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        count = 0
        for word in professor:
            if word in text:
                count += 1

        return count

    def author_features(self):  # lexical feature
        keywords = variables.author
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        count = 0
        for word in keywords:
            if word in text:
                count += 1

        return count

    def block_index(self):  # local feature
        return self.block_number

    def block_max_font(self):  # local feature
        return self.is_max_font

    def block_metadata_relative_count(self):  # local feature
        if self.para_in_block_idx == 0:
            return -1
        else:
            return self.block_meta_count / self.para_in_block_idx

    def block_other_relative_count(self):  # local feature
        if self.para_in_block_idx == 0:
            return -1
        else:
            return self.block_other_count / self.para_in_block_idx

    def block_para_index(self):  # local feature
        return self.para_in_block_idx

    def block_student_count(self):  # local feature
        return self.student_count

    def block_supervisor_count(self):  # local feature
        return self.super_count

    def block_title_count(self):  # local feature
        return self.title_count

    def block_word_relative_count(self):  # local feature
        return len(self.text.split(" ")) / self.block_word_count

    def bracket_count(self):  # heuristic feature

        count = self.text.count("[") + self.text.count("]")
        return count

    def bracket_relative_count(self):  # heuristic feature

        count = self.text.count("[") + self.text.count("]")
        return 0 if len(self.text) == 0 else count / len(self.text)

    def center_alignment(self):  # formatting features
        try:
            if (
                self.paragraph.paragraph_format.alignment
                == WD_PARAGRAPH_ALIGNMENT.CENTER
            ):
                return 1
            else:
                return 0
        except:
            return 0

    def chapter_title_feature(self):  # lexical feature
        keywords = variables.first_chapter_words
        text = self.text
        chars = [" ", "\u200c"]
        flag = 0
        for c in chars:
            text = text.replace(c, "")
        for keyword in keywords:
            if keyword in text:
                flag = 1
                break
        return flag

    def colon_count(self):  # heuristic feature

        count = self.text.count(":")
        return count

    def colon_relative_count(self):  # heuristic feature

        count = self.text.count(":")
        return 0 if len(self.text) == 0 else count / len(self.text)

    def comma_count(self):  # heuristic feature

        count = self.text.count(",") + self.text.count("،")
        return count

    def comma_relative_count(self):  # heuristic feature

        count = self.text.count(",") + self.text.count("،")
        return 0 if len(self.text) == 0 else count / len(self.text)

    def date_feature(self):  # lexical feature

        date = variables.months_seasons
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        flag = 0
        for word in date:
            if word in text:
                flag = 1

        return flag

    def degree_feature(self):  # lexical feature

        degree = variables.degree
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        count = 0
        for word in degree:
            if word in text:
                count += 1

        return count

    def digit_count(self):  # heuristic feature

        count = 0
        for word in self.text:
            if word.isnumeric():
                count += 1
        return count

    def digit_relative_count(self):  # heuristic feature

        count = 0
        for word in self.text:
            if word.isnumeric():
                count += 1

        return 0 if len(self.text) == 0 else count / len(self.text)

    def dot_count(self):  # heuristic feature

        count = self.text.count(".")
        return count

    def dot_relative_count(self):  # heuristic feature

        count = self.text.count(".")
        return 0 if len(self.text) == 0 else count / len(self.text)

    def faculty_feature(self):  # lexical feature

        faculty = variables.faculty
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        count = 0
        for word in faculty:
            if word in text:
                count += 1

        return count

    def field_feature(self):  # lexical feature

        field = variables.field
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        count = 0
        for word in field:
            if word in text:
                count += 1

        return count

    def font_size(self):  # formatting features
        return self.para_font

    def is_after_image(self):  # geometric feature
        return self.image_flag

    def is_bold(self):  # formatting features

        return self.para_bold

    def is_first_paragraph(self):  # heuristic feature

        if self.count == 0:
            return 1
        else:
            return 0

    def is_font_name_differ(self):  # sequential features

        if self.pre_font_name == -1:
            return 0
        if self.pre_font_name == self.para_font_name:

            return 0
        else:
            return 1

    def is_font_size_differ(self):  # sequential features

        if self.pre_font_size == -1:
            return 0
        else:
            if self.pre_font_size == self.para_font:
                return 0
            else:
                return 1

    def is_italic(self):  # formatting features

        return self.para_italic

    def is_numeric(self):  # heuristic feature

        if self.text.isnumeric():

            return 1
        else:
            return 0

    def is_sequence_bold(self):  # sequential features

        if self.pre_para_bold == -1:
            return 0

        if self.para_bold and self.pre_para_bold:
            return 1
        else:
            return 0

    def is_sequence_italic(self):  # sequential features

        if self.pre_para_italic == -1:
            return 0

        if self.para_italic and self.pre_para_italic:
            return 1
        else:
            return 0

    def is_single_word(self):  # heuristic feature

        if len(self.text.split(" ")) == 1:
            return 1
        else:
            return 0

    def keywords_features(self):  # lexical feature

        keywords = variables.keyword
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        count = 0
        for word in keywords:
            if word in text:
                count += 1

        return count

    def line_approximation(self):  # heuristic feature
        count = 1
        line = 0
        for run in self.paragraph.runs:
            if run.text != "":
                if run.bold:
                    font_path = "Features/Fonts/BZarBold.ttf"
                else:
                    font_path = "Features/Fonts/BZar.ttf"

                f = ImageFont.truetype(
                    font=font_path, size=int(self.para_font), encoding="unic"
                ).getlength(run.text)
                line += f
                if line > self.line_width:
                    count += 1
                    line = f

        self.line_length = line
        self.line_count = count

        return count

    def margin_left(self):  # geometric feature
        if self.line_count == 1:
            try:
                if (
                    self.paragraph.paragraph_format.alignment
                    == WD_PARAGRAPH_ALIGNMENT.CENTER
                ):
                    center = 1
                else:
                    center = 0
            except:
                center = 0
            if center:
                left_margin = (self.line_width - self.line_length) / 2
            else:
                left_margin = self.line_width - self.line_length
        elif self.line_count != 1:
            left_margin = 0

        return left_margin

    def margin_right(self):  # geometric feature
        if self.line_count == 1:
            try:
                if (
                    self.paragraph.paragraph_format.alignment
                    == WD_PARAGRAPH_ALIGNMENT.CENTER
                ):
                    center = 1
                else:
                    center = 0
            except:
                center = 0
            if center:
                right_margin = (self.line_width - self.line_length) / 2
            else:
                right_margin = 0
        elif self.line_count != 1:
            right_margin = 0

        return right_margin

    def other_keyword_feature(self):  # lexical feature

        other_key = variables.other_keywords
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        count = 0
        for word in other_key:
            if word in text:
                count += 1

        return count

    def paragraph_number(self):  # heuristic feature
        return self.count

    def parentheses_count(self):  # heuristic feature

        count = self.text.count(")") + self.text.count("(")
        return count

    def parentheses_relative_count(self):  # heuristic feature

        count = self.text.count(")") + self.text.count("(")
        return 0 if len(self.text) == 0 else count / len(self.text)

    def pre_label_feature(self):  # sequential features
        return self.pre_label

    def professor_feature(self):  # lexical feature

        professor = variables.professor
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        count = 0
        for word in professor:
            if word in text:
                count += 1

        return count

    def punc_relative_count(self):  # heuristic feature

        count = 0
        for c in self.text:
            if c in variables.Punctuation:
                count += 1

        return 0 if len(self.text) == 0 else count / len(self.text)

    def space_before(self):  # heuristic feature
        if self.space_count == 0:
            return 0
        else:
            return 1

    def space_relative_count(self):  # heuristic feature

        count = self.text.count(" ")
        return 0 if len(self.text) == 0 else count / len(self.text)

    def start_with_digit(self):  # heuristic feature
        if self.text.strip()[0].isnumeric():
            return 1
        else:
            return 0

    def title_feature(self):  # lexical feature

        title = variables.title
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        flag = 0
        for word in title:
            if word in text:
                flag = 1

        return flag

    def toc_feature(self):  # lexical feature

        toc = variables.toc  # table of content
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        count = 0
        for word in toc:
            if word in text:
                count += 1

        return count

    def two_pre_label_feature(self):  # sequential features
        return self.two_pre_label

    def type_feature(self):  # lexical feature

        type = variables.parsa_type
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        count = 0
        for word in type:
            if word in text:
                count += 1

        return count

    def university_feature(self):  # lexical feature

        university = variables.university
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        count = 0
        for word in university:
            if word in text:
                count += 1

        return count

    def width_height_ratio(self):  # geometric feature
        line_count = self.line_approximation()

        if self.paragraph.paragraph_format.line_spacing == None:
            line_spaceing = self.default_line_spacing
        else:
            line_spaceing = self.paragraph.paragraph_format.line_spacing

        if line_count == 1:
            if self.line_length == 0:
                ratio = (self.para_font + (2 * line_spaceing)) / 0.0001
            else:
                ratio = (self.para_font + (2 * line_spaceing)) / self.line_length
        else:
            ratio = (
                (self.para_font * line_count) + (line_spaceing * (line_spaceing + 1))
            ) / self.line_width

        return ratio

    def word_count(self):  # heuristic feature
        count = 0
        for word in self.text.split(" "):
            if word.strip() != "":
                flag = 1
                for ch in word:
                    if ch in variables.Punctuation or ch.isnumeric():
                        flag = 0
                if flag:
                    count += 1
        return count

    def word_length_mean(self):  # heuristic feature

        text = self.text.split(" ")
        count = 0
        for word in text:
            if word.strip() != "":
                count += len(word)

        return 0 if len(text) == 0 else count / len(text)

    def word_length_median(self):  # heuristic feature

        text = self.text.split(" ")
        length = []
        for word in text:
            if word.strip() != "":
                length.append(len(word))
        return np.median(length)

    def word_width_mean(self):  # geometric feature

        length = len(self.text.split(" "))
        count = 1
        text_width = 0
        for run in self.paragraph.runs:
            if run.text != "":
                if run.bold:
                    font_path = "Features/Fonts/BZarBold.ttf"
                else:
                    font_path = "Features/Fonts/BZar.ttf"

                text_width += ImageFont.truetype(
                    font=font_path, size=int(self.para_font), encoding="unic"
                ).getlength(run.text)

        return text_width / length

    def metadata_keywords_feature(self):
        metadata_keywords = variables.metadata_keywords
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            Words = text.replace(c, "")
        count = 0
        for word in metadata_keywords:
            if word in Words:
                return 1
        return 0

    def metadata_keywords_ratio(self):
        metadata_keywords = variables.metadata_keywords
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            Words = text.replace(c, "")
        count = 0
        for word in metadata_keywords:
            if word in Words:
                count += 1
                break
        word_count = self.word_count()
        if word_count > 0:
            return count / word_count
        else:
            return 0

    def faculty_index(self):
        faculties = variables.faculty
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        indices = []
        for faculty in faculties:
            start_index = text.find(faculty)
            if start_index != -1:
                indices.append(start_index)
        if len(indices) == 0:
            index = -1
        else:
            index = min(indices)
        return index

    def university_index(self):
        universities = variables.university
        text = self.text
        chars = [" ", "\u200c"]
        for c in chars:
            text = text.replace(c, "")
        indices = []
        for university in universities:
            start_index = text.find(university)
            if start_index != -1:
                indices.append(start_index)
        if len(indices) == 0:
            index = -1
        else:
            index = min(indices)
        return index
