import os
import csv
import numpy
from metadata_classification import FeatureNames as FN

heu = FN.heuristic_feature_names
loc = FN.local_feature_names
lex = FN.lexical_feature_names
geo = FN.geometric_feature_names
form = FN.formatting_feature_names
seq = FN.sequential_feature_names
features_types = heu + loc + lex + geo + form + seq


def load_metadata_features(address="metadata_classification/metadata_features_index/"):
    X = []
    Y = []
    print("load preprocessed data...")
    for idx, feature_file in enumerate(sorted(os.listdir(address))):
        with open(address + feature_file, "r") as f:
            ignore_idx = []
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                if i == 0:
                    for j in range(len(line) - 1):
                        if line[j] not in features_types:
                            ignore_idx.append(j)
                    continue

                csv_row = []
                for s in range(len(line) - 1):
                    if s not in ignore_idx:
                        csv_row.append(float(line[s]))
                X.append(csv_row)
                Y.append(int(line[-1]))

    print("data size=", len(X))
    print("features count=", len(X[0]))
    return numpy.array(X), numpy.array(Y)
