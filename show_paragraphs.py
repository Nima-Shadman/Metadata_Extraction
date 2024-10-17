from docx import Document
from boundary_detection.features_for_boundary_detection import get_paragraphs
from metadata_classification.features_for_meta_classification import get_blocks
from itertools import chain
from Features.preprocessing import basic_preproccesing
from Features.preprocessing import metadata_preprocessing
import arabic_reshaper
from bidi.algorithm import get_display
import csv


def cut_blocks(two_d_list, index):
    flattened_list = [element for sublist in two_d_list for element in sublist]
    truncated_flattened_list = flattened_list[:index]

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


def read_boundary(i):
    with open("boundary_detection/docs_boundary_index.csv") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        for line in reader:
            if line[1].strip() != "":
                if line[0] == str(i):
                    body_num = int(line[1])
        return body_num


file_name = 147
start = 0
stop = 200
offset = 0
document = Document("ETDs/" + str(file_name) + ".docx")
phase = 1

if phase == 1:
    paragraphs = get_paragraphs(document)
    nonempty_paragraphs = [
        paragraph for paragraph in paragraphs if not paragraph.text.strip() == ""
    ]
    for i in range(start, stop):
        print(
            f"{i}:\n{get_display(arabic_reshaper.reshape(nonempty_paragraphs[i].text))}\n==========================================================================\n\n"
        )

elif phase == 2:
    boundary = read_boundary(file_name)
    blocks = get_blocks(document)
    blocks_cut = cut_blocks(blocks, boundary)
    preprocessed = metadata_preprocessing(blocks_cut)
    paragraphs = list(chain.from_iterable(preprocessed))
    for i in range(len(paragraphs) - offset):
        print(
            f"{i}:\n{get_display(arabic_reshaper.reshape(paragraphs[i].text))}\n==========================================================================\n\n"
        )
