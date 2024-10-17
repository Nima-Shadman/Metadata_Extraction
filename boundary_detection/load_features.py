import os
import csv
import numpy


def load_features():
    X = []
    Y = []
    seq = [3, 10, 11]
    form = [12]
    geo = [7, 13]
    lex = [0, 9]
    heu = [1, 2, 4, 5, 6, 8]
    feature_index = heu + seq + geo + lex + form
    metadata_count = 0
    body_count = 0
    address = "boundary_detection/boundary_labels_features/"
    for idx, feature_file in enumerate(sorted(os.listdir(address))):
        with open(address + feature_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                if i == 0 or len(line) == 0:
                    continue
                csv_row = []
                for s in range(len(line) - 1):
                    if s in feature_index:
                        csv_row.append(float(line[s]))
                X.append(csv_row)
                try:
                    Y.append(int(line[-1]))
                except IndexError:
                    print(i)
                if (int(line[-1])) == 0:
                    metadata_count += 1
                elif (int(line[-1])) == 1:
                    body_count += 1

    print("data size=", len(X))
    print("metadata size=", metadata_count)
    print("body size=", body_count)
    print("imbalance ratio=", metadata_count / body_count)
    print("features count=", len(X[0]))

    return numpy.array(X), numpy.array(Y)
