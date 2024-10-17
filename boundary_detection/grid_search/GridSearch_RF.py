from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import csv
from boundary_detection import load_features

X, Y, file_names = load_features.load_features()
skf = KFold(n_splits=10, shuffle=True)


def train_rf(train_index, test_index, sol):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Normalizing the data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # Unpack the solution (n_estimators and max_depth)
    n_estimators, max_depth = sol

    # Train the Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    clf.fit(X_train_norm, y_train)

    # Predict and calculate accuracy
    predict = clf.predict(X_test_norm)
    accuracy = accuracy_score(y_test, predict)

    return dict(error_rate=1 - accuracy)


def classification(sol):
    print(sol)
    out = Parallel(n_jobs=4, verbose=0, pre_dispatch="2*n_jobs")(
        delayed(train_rf)(train_index, test_index, sol)
        for train_index, test_index in skf.split(X, Y)
    )
    fitness = [d["error_rate"] for d in out]
    avg = np.average(fitness)
    with open("boundary_detection/fitness_surface.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(sol + [avg])
    return avg


def grid_search():
    start_time = time.time()
    n_estimators_range = list([100, 300, 500, 700, 900])
    max_depth_range = list([10, 30, 50, 70, None])

    fitness = [0.0] * (len(n_estimators_range) * len(max_depth_range))

    count = 0
    for n in n_estimators_range:
        for d in max_depth_range:
            fitness[count] = classification([n, d])
            print(fitness[count])
            count += 1
            with open("boundary_detection/gridsearch_rf.txt", "a") as file:
                file.write(
                    "iteration "
                    + str(count)
                    + "\n"
                    + "time="
                    + str((time.time() - start_time))
                    + "\n"
                    + "n_estimators="
                    + str(n)
                    + "\n"
                    + "max_depth="
                    + str(d)
                    + "\n"
                    + "Fitness="
                    + str(fitness)
                    + "\n"
                )

    best = np.min(fitness)
    print("best=", best)
