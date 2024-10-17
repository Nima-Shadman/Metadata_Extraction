from joblib import load
import numpy as np
from docx import Document
from Features.Feature_extraction import Features
from Features.variables import font_size, is_bold, is_italic, boundary_labels
import boundary_detection.FeatureNames as FN
from Features.preprocessing import basic_preproccesing
from boundary_detection.features_for_boundary_detection import get_paragraphs


def boundary_index(predict, threshold, min):
    predict = [int(x[0]) for x in predict]
    change_labels = np.where(np.diff(predict) != 0)[0]
    if len(change_labels) == 1:
        return change_labels[0] + 1

    if predict[0] == 0:
        subtraction = [y - x for x, y in zip(change_labels, change_labels[1:])][0::2]
    else:
        subtraction = [y - x for x, y in zip(change_labels, change_labels[1:])][1::2]

    for i in range(len(subtraction)):
        if subtraction[i] >= threshold:
            return change_labels[2 * i + (1 if predict[0] != 0 else 0)] + 1

    max_sub = max(subtraction)
    if max_sub > min:
        for i in range(len(subtraction)):
            if subtraction[i] == max_sub:
                return change_labels[2 * i + (1 if predict[0] != 0 else 0)] + 1
    else:
        if predict[-1] == 0:
            return len(predict)
        else:
            for i in range(len(predict) - 1, -1, -1):
                if predict[i] == 0:
                    return i + 1


def find_boundary(file_path, body_num=None):
    scaler = load("boundary_detection/normalized_model.joblib")
    clf = load("boundary_detection/boundary_detection_model.joblib")

    document = Document(file_path)
    all_paragraphs = get_paragraphs(document)
    paragraphs = [
        paragraph for paragraph in all_paragraphs if not paragraph.text.strip() == ""
    ]

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

    # get page margins
    section = document.sections[0]
    p_width = section.page_width.pt
    l_margin = section.left_margin.pt
    r_margin = section.right_margin.pt
    line_width = p_width - (l_margin + r_margin)
    # font name
    font_default = style.font.name

    predictions = []
    index = 0
    pre_label = -1
    two_pre_label = -1
    pre_font_size = 1
    texts = []
    for para in paragraphs:
        text = basic_preproccesing(para.text)
        if text.strip() == "":
            continue
        texts.append(text)
        temp = []
        # get paragraph font
        para_font = font_size(default_font_size, para)

        # is italic and(or) bold
        para_bold = is_bold(para)
        para_italic = is_italic(para)

        # create a Feature object
        f = Features(
            para,
            text,
            index,
            para_font,
            pre_font_size,
            line_width,
            default_font_size,
            pre_label,
            two_pre_label,
            default_line_spacing,
        )

        for i in range(len(FN.feature_names) - 1):
            temp.append(getattr(f, FN.feature_names[i])())

        temp = np.array(temp).reshape(1, -1)
        features_norm = scaler.transform(temp)
        predict = clf.predict(features_norm)
        predictions.append(predict)

        pre_font_size = para_font  # save pre paragraph font
        pre_para_italic = para_italic  # save pre paragraph italic
        pre_para_bold = para_bold  # save pre paragraph italic

        # save labels
        if index > 0:
            pre_label = predictions[index - 1]
        if index > 1:
            two_pre_label = predictions[index - 2]
        index += 1

    if body_num:
        labels = []
        for i in range(len(paragraphs)):
            text = basic_preproccesing(paragraphs[i].text)
            if text.strip() != "":
                if i < body_num:
                    labels.append(boundary_labels["metadata"])
                else:
                    labels.append(boundary_labels["body"])

        correct_count = 0
        if len(predictions) == len(labels):
            for i in range(len(predictions)):
                if predictions[i] == labels[i]:
                    correct_count += 1
        accuracy = correct_count / len(predictions)

    boundary = boundary_index(predictions, 5, 2)
    return boundary


find_boundary("ETDs/test.docx", 192)
