from joblib import load
import numpy as np
from docx import Document
from Features.Feature_extraction import Features
from Features.variables import font_size, is_bold, is_italic
import boundary_detection.FeatureNames as FN
from Features.preprocessing import basic_preproccesing
from boundary import find_boundary
from docx import Document
from Features.Feature_extraction import Features
import numpy
from Features.variables import font_size, metadata_labels_10
from Features.preprocessing import basic_preproccesing
from metadata_classification import FeatureNames as FN
from Features.variables import font_size, is_bold, is_italic, font_name
from Features.preprocessing import metadata_preprocessing
from metadata_classification.features_for_meta_classification import (
    cut_blocks,
    get_blocks,
)
from itertools import chain
import arabic_reshaper
from bidi.algorithm import get_display


def extract_metadata(file_path, boundary=None):
    metadata = {
        "university": [],
        "faculty": [],
        "degree & field": [],
        "date": [],
        "title": [],
        "supervisor": [],
        "student": [],
        "abstract": [],
        "keyword": [],
    }
    clf = load("metadata_model.joblib")
    scaler = load("meta_normalized_model.joblib")
    if not boundary:
        boundary = find_boundary(file_path)

    label = []
    document = Document(file_path)
    # get font,font size and line spacing
    style = document.styles["Normal"]

    if style.font.size == None:
        default_font_size = 12
    else:
        default_font_size = style.font.size.pt

    if style.paragraph_format.line_spacing == None:
        default_line_spacing = 1.15
    else:
        default_line_spacing = style.paragraph_format.line_spacing

    if style.font.name == None:
        font_default = "Calibri"
    else:
        font_default = style.font.name

    # get page margins
    section = document.sections[0]
    p_width = section.page_width.pt
    l_margin = section.left_margin.pt
    r_margin = section.right_margin.pt
    line_width = p_width - (l_margin + r_margin)

    feature = []
    feature.append(FN.feature_names)
    index = 0
    is_max_font = 0
    pre_label = -1
    two_pre_label = -1
    pre_font_size = -1
    pre_para_italic = -1
    pre_para_bold = -1
    pre_font_name = -1
    space_count = 0
    image_flag = 0
    end = 0
    block_index = 0
    block_words_count = 0
    block_meta_count = 0
    block_other_count = 0

    blocks = get_blocks(document)
    blocks_cut = cut_blocks(blocks, boundary)
    preprocessed_blocks = metadata_preprocessing(blocks_cut)
    paragraph_index = len(list(chain.from_iterable(preprocessed_blocks)))
    for block in preprocessed_blocks:
        if end:
            break
        font_list = font_size(default_font_size, block)
        max_font = numpy.argwhere(font_list == numpy.amax(font_list)).flatten().tolist()
        para_in_block_idx = 0
        block_labels = []
        title_count = 0
        super_count = 0
        student_count = 0
        block_words_count = sum(
            [len(paragraph.text.split(" ")) for paragraph in block]
        )  # if this needs to be calculated first
        for paragraph in block:
            if "graphicData" in paragraph._p.xml:
                image_flag = 1

            if index < paragraph_index:
                temp = []
                if para_in_block_idx in max_font:
                    is_max_font = 1
                else:
                    is_max_font = 0

                para_bold = is_bold(paragraph)
                para_italic = is_italic(paragraph)
                para_font_name = font_name(paragraph, font_default)

                if para_in_block_idx == 0:
                    space_count = 1
                else:
                    space_count = 0

                text = basic_preproccesing(paragraph.text)
                f = Features(
                    paragraph,
                    text,
                    index,
                    font_list[para_in_block_idx],
                    pre_font_size,
                    line_width,
                    default_font_size,
                    pre_label,
                    two_pre_label,
                    default_line_spacing,
                    pre_para_italic,
                    para_italic,
                    pre_para_bold,
                    para_bold,
                    para_font_name,
                    pre_font_name,
                    space_count,
                    image_flag,
                    block_words_count,
                    block_index,
                    is_max_font,
                    para_in_block_idx,
                    block_other_count,
                    block_meta_count,
                    title_count,
                    super_count,
                    student_count,
                )
                for i in range(len(FN.feature_names) - 1):
                    if isinstance(getattr(f, FN.feature_names[i])(), np.ndarray):
                        print(FN.feature_names[i])
                    temp.append(getattr(f, FN.feature_names[i])())

                temp = np.array(temp).reshape(1, -1)
                temp_norm = scaler.transform(temp)
                predict = clf.predict(temp_norm)
                label.append(predict[0])

                if label[index] == 0:
                    block_other_count += 1
                else:
                    block_meta_count += 1
                if index > 0:
                    pre_label = label[index - 1]
                if index > 1:
                    two_pre_label = label[index - 2]
                block_labels.append(label[index])

                if metadata_labels_10["title"] in block_labels:
                    title_count += 1
                if metadata_labels_10["supervisor"] in block_labels:
                    super_count += 1
                if metadata_labels_10["student"] in block_labels:
                    student_count += 1

                pre_font_size = font_list[
                    para_in_block_idx
                ]  # save previous paragraph font
                pre_para_italic = para_italic  # save previous paragraph italic
                pre_para_bold = para_bold  # save previous paragraph bold
                pre_font_name = para_font_name  # save previous paragraph font name
                index += 1
                space_count = 0
                image_flag = 0
                para_in_block_idx += 1

                feature.append(temp)

                match predict[0]:
                    case 1:
                        metadata["university"].append(text)
                    case 2:
                        metadata["faculty"].append(text)
                    case 3:
                        metadata["degree & field"].append(text)
                    case 4:
                        metadata["date"].append(text)
                    case 5:
                        metadata["title"].append(text)
                    case 6:
                        metadata["supervisor"].append(text)
                    case 7:
                        metadata["student"].append(text)
                    case 8:
                        metadata["abstract"].append(text)
                    case 9:
                        metadata["keyword"].append(text)

            else:
                end = True

        block_index += 1
        block_words_count = 0
        block_meta_count = 0
        block_other_count = 0

    return metadata


metadata = extract_metadata("ETDs/test.docx")
for metadata, texts in metadata.items():
    print(f"{metadata}:")
    for text in texts:
        print(f"{get_display(arabic_reshaper.reshape(text))}")
