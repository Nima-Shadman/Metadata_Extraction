from Features.Feature_extraction import Features

f = Features
feature_names = []
feature_names.append(f.abstract_feature.__name__)
feature_names.append(f.author_features.__name__)
feature_names.append(f.block_index.__name__)
feature_names.append(f.block_metadata_relative_count.__name__)
feature_names.append(f.block_other_relative_count.__name__)
feature_names.append(f.block_para_index.__name__)
feature_names.append(f.block_word_relative_count.__name__)
feature_names.append(f.center_alignment.__name__)
feature_names.append(f.colon_relative_count.__name__)
feature_names.append(f.comma_relative_count.__name__)
feature_names.append(f.date_feature.__name__)
feature_names.append(f.degree_feature.__name__)
feature_names.append(f.digit_relative_count.__name__)
feature_names.append(f.dot_relative_count.__name__)
feature_names.append(f.faculty_feature.__name__)
feature_names.append(f.field_feature.__name__)
feature_names.append(f.font_size.__name__)
feature_names.append(f.is_bold.__name__)
feature_names.append(f.is_single_word.__name__)
feature_names.append(f.keywords_features.__name__)
feature_names.append(f.line_approximation.__name__)
feature_names.append(f.margin_left.__name__)
feature_names.append(f.margin_right.__name__)
feature_names.append(f.paragraph_number.__name__)
feature_names.append(f.pre_label_feature.__name__)
feature_names.append(f.professor_feature.__name__)
feature_names.append(f.punc_relative_count.__name__)
feature_names.append(f.space_relative_count.__name__)
feature_names.append(f.title_feature.__name__)
feature_names.append(f.toc_feature.__name__)
feature_names.append(f.two_pre_label_feature.__name__)
feature_names.append(f.type_feature.__name__)
feature_names.append(f.university_feature.__name__)
feature_names.append(f.width_height_ratio.__name__)
feature_names.append(f.word_count.__name__)
feature_names.append(f.word_length_mean.__name__)
feature_names.append(f.university_index.__name__)
feature_names.append(f.faculty_index.__name__)

feature_names.append("label")

local_feature_names = [
    f.block_index.__name__,
    f.block_metadata_relative_count.__name__,
    f.block_other_relative_count.__name__,
    f.block_para_index.__name__,
    f.block_word_relative_count.__name__,
]

geometric_feature_names = [
    f.center_alignment.__name__,
    f.margin_left.__name__,
    f.margin_right.__name__,
    f.width_height_ratio.__name__,
]

sequential_feature_names = [
    f.paragraph_number.__name__,
    f.pre_label_feature.__name__,
    f.two_pre_label_feature.__name__,
]

lexical_feature_names = [
    f.abstract_feature.__name__,
    f.author_features.__name__,
    f.date_feature.__name__,
    f.degree_feature.__name__,
    f.faculty_feature.__name__,
    f.field_feature.__name__,
    f.keywords_features.__name__,
    f.professor_feature.__name__,
    f.title_feature.__name__,
    f.toc_feature.__name__,
    f.type_feature.__name__,
    f.university_feature.__name__,
    f.university_index.__name__,
    f.faculty_index.__name__,
]

formatting_feature_names = [
    f.font_size.__name__,
    f.is_bold.__name__,
]

heuristic_feature_names = [
    f.colon_relative_count.__name__,
    f.comma_relative_count.__name__,
    f.digit_relative_count.__name__,
    f.dot_relative_count.__name__,
    f.is_single_word.__name__,
    f.line_approximation.__name__,
    f.punc_relative_count.__name__,
    f.space_relative_count.__name__,
    f.word_count.__name__,
    f.word_length_mean.__name__,
]
