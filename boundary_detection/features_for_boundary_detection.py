from docx import Document
from Features.Feature_extraction import Features
from Features.variables import font_size, is_bold, is_italic, boundary_labels
from boundary_detection import FeatureNames as FN
import csv
from Features.preprocessing import basic_preproccesing
from docx.text.paragraph import Paragraph


def get_paragraphs(document):
    body = document._body._body
    ps = body.xpath("//w:p")
    paragraphs = []
    for p in ps:
        try:
            paragraphs.append(Paragraph(p, p.getparent()))
        except:
            paragraphs.append(Paragraph(p, document._body))
    return paragraphs


def save_features():
    with open("boundary_detection/docs_boundary_index_new.csv") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        counter = 0
        for line in reader:
            if line[1].strip() != "":
                counter += 1
                print(counter, ") extracting features from file", line[0])
                file_name = line[0]
                body_num = int(line[1])
                document = Document("ETDs/" + file_name + ".docx")
                all_paragraphs = get_paragraphs(document)
                paragraphs = [
                    paragraph
                    for paragraph in all_paragraphs
                    if not paragraph.text.strip() == ""
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

                labels = []
                for i in range(len(paragraphs)):
                    if i < body_num:
                        labels.append(boundary_labels["metadata"])
                    else:
                        labels.append(boundary_labels["body"])

                feature = []
                feature.append(FN.feature_names)
                index = 0
                pre_label = -1
                two_pre_label = -1
                pre_font_size = 1
                pre_para_italic = -1
                pre_para_bold = -1
                pre_font_name = -1

                for para in paragraphs:
                    text = basic_preproccesing(para.text)
                    if text.strip() == "":
                        continue
                    temp = []
                    # get paragraph font
                    para_font = font_size(default_font_size, para)

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
                    temp.append(labels[index])

                    pre_font_size = para_font  # save pre paragraph font

                    # save labels
                    if index > 0:
                        pre_label = labels[index - 1]
                    if index > 1:
                        two_pre_label = labels[index - 2]
                    index += 1
                    feature.append(temp)

                # save features
                with open(
                    "boundary_detection/boundary_labels_features/fb"
                    + file_name
                    + ".csv",
                    "w",
                    newline="",
                ) as file:
                    writer = csv.writer(file)
                    for row in range(len(feature)):
                        writer.writerow(feature[row])
                print(f"features for file {file_name} saved.")
