from Features.Feature_extraction import Features

f = Features
feature_names = []
feature_names.append(f.chapter_title_feature.__name__)  # 1 #lexical
feature_names.append(f.digit_relative_count.__name__)  # 2 #heuristic
feature_names.append(f.dot_relative_count.__name__)  # 3 #heuristic
feature_names.append(f.paragraph_number.__name__)  # 4 #sequential
feature_names.append(f.parentheses_relative_count.__name__)  # 5 #heuristic
feature_names.append(f.punc_relative_count.__name__)  # 6 #heuristic
feature_names.append(f.start_with_digit.__name__)  # 7 #heuristic
feature_names.append(f.width_height_ratio.__name__)  # 8 #geometric
feature_names.append(f.word_count.__name__)  # 9 #heuristic
feature_names.append(f.metadata_keywords_feature.__name__)  # 10 #lexical
feature_names.append(f.pre_label_feature.__name__)  # 11 sequential
feature_names.append(f.two_pre_label_feature.__name__)  # 12 sequential
feature_names.append(f.font_size.__name__)  # 13 formatting
feature_names.append(f.center_alignment.__name__)  # 14 geometric

feature_names.append("label")

sequential_feature_names = [
    f.paragraph_number.__name__,
    f.pre_label_feature.__name__,
    f.two_pre_label_feature.__name__,
]

heuristic_feature_names = [
    f.digit_relative_count.__name__,
    f.dot_relative_count.__name__,
    f.parentheses_relative_count.__name__,
    f.punc_relative_count.__name__,
    f.start_with_digit.__name__,
    f.word_count.__name__,
]

lexical_feature_names = [
    f.chapter_title_feature.__name__,
    f.metadata_keywords_feature.__name__,
]

geometric_feature_names = [f.width_height_ratio.__name__, f.center_alignment.__name__]

formatting_feature_names = [f.font_size.__name__]
