from docx import Document
from docx.oxml.table import CT_Tc
from docx.text.paragraph import Paragraph
from Features.Feature_extraction import Features
import numpy
from Features.variables import font_size, metadata_labels_10
from Features.preprocessing import basic_preproccesing
from metadata_classification import FeatureNames as FN
import csv
import os
from Features.variables import font_size, is_bold, is_italic, font_name
from Features.preprocessing import metadata_preprocessing


def read_boundary(file_name):
    with open("boundary_detection/docs_boundary_index_new.csv") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        counter = 0
        for line in reader:
            if line[0].strip() == str(file_name):
                body_num = int(line[1])
                return body_num


def get_blocks(document):
    body = document._body._body
    ps = body.xpath("//w:p")
    paragraphs = []
    for p in ps:
        try:
            paragraphs.append(Paragraph(p, p.getparent()))
        except:
            paragraphs.append(Paragraph(p, document._body))

    blocks = []
    block = []
    for paragraph in paragraphs:
        if paragraph.text.strip() != "":
            block.append(paragraph)
        elif len(block) != 0 and not isinstance(paragraph._p.getparent(), CT_Tc):
            blocks.append(block)
            block = []

    return blocks


def cut_blocks(two_d_list, index):
    # Flatten the 2D list into a 1D list
    flattened_list = [element for sublist in two_d_list for element in sublist]

    # Get only the elements up to the given index
    truncated_flattened_list = flattened_list[:index]

    # Reconstruct the 2D list with the same structure up to the index
    new_2d_list = []
    current_index = 0

    for sublist in two_d_list:
        new_sublist = []
        for element in sublist:
            if current_index < index:
                new_sublist.append(element)
                current_index += 1
            else:
                break
        if new_sublist:
            new_2d_list.append(new_sublist)
        if current_index >= index:
            break

    return new_2d_list


def save_features():
    address = "metadata_classification/metadata_labels/"

    for idx, csv_file_name in enumerate(sorted(os.listdir(address))):
        file_name = csv_file_name[:-4]
        label = []
        document = Document("ETDs/" + file_name + ".docx")
        print(f"extracting features from file {file_name}")
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

        with open(
            "metadata_classification/metadata_labels/" + file_name + ".csv",
            "r",
            encoding="utf-8",
        ) as file:
            reader = csv.reader(file)
            line_count = 0
            for line in reader:
                line_count += 1
                if line_count == 1:
                    continue
                label.append(int(line[1].strip()))

        paragraphs_index = len(label)
        paragraphs = []

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
        cutoff = read_boundary(file_name)
        blocks_cut = cut_blocks(blocks, cutoff)
        preprocessed_blocks = metadata_preprocessing(blocks_cut)
        for block in preprocessed_blocks:
            if end:
                break
            font_list = font_size(default_font_size, block)
            max_font = (
                numpy.argwhere(font_list == numpy.amax(font_list)).flatten().tolist()
            )
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

                if index < paragraphs_index:
                    paragraphs.append([paragraph.text.strip(), label[index]])
                    block_labels.append(label[index])
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

                    if index > 0:
                        pre_label = label[index - 1]
                    if index > 1:
                        two_pre_label = label[index - 2]

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
                        temp.append(getattr(f, FN.feature_names[i])())

                    temp.append(label[index])

                    if label[index] == 0:
                        block_other_count += 1
                    else:
                        block_meta_count += 1
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

                    if metadata_labels_10["title"] in block_labels:
                        title_count += 1
                    if metadata_labels_10["supervisor"] in block_labels:
                        super_count += 1
                    if metadata_labels_10["student"] in block_labels:
                        student_count += 1

                    feature.append(temp)
                else:
                    end = True

            block_index += 1
            block_words_count = 0
            block_meta_count = 0
            block_other_count = 0

        # save features
        with open(
            "metadata_classification/metadata_features_index/fm" + file_name + ".csv",
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            for row in range(len(feature)):
                writer.writerow(feature[row])
