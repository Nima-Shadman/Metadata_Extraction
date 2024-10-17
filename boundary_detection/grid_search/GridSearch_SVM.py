from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import csv
from sklearn.metrics import accuracy_score
from numpy import average
from boundary_detection import load_features
from joblib import Parallel, delayed
import numpy
import time

X, Y, file_names = load_features.load()
skf = KFold(n_splits=10, shuffle=True)


def train(train_index, test_index, sol):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)

    c, kernel_num = sol

    clf = svm.SVC(C=c, kernel="rbf", gamma=kernel_num)
    clf.fit(X_train_norm, y_train)

    X_test_norm = scaler.transform(X_test)
    predict = clf.predict(X_test_norm)

    accuracy = accuracy_score(y_test, predict)

    return dict(error_rate=1 - accuracy)


def classification(sol):

    print(sol)
    out = Parallel(n_jobs=4, verbose=0, pre_dispatch="2*n_jobs")(
        delayed(train)(train_index, test_index, sol)
        for train_index, test_index in skf.split(X, Y)
    )
    fitness = [d["error_rate"] for d in out]
    avg = average(fitness)
    with open("boundary_detection/fitness_surface.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow(sol + [avg])
    return avg


start_time = time.time()
c_range = list(range(2, 10))
sigma_range = list(range(-7, 8))

fitness = [0.0] * (len(c_range) * len(sigma_range))

count = 0
for i in c_range:
    for j in sigma_range:
        fitness[count] = classification([2**i, 2**j])
        print(fitness[count])
        count += 1
        with open("boundary_detection/gridsearch_svm.txt", "a") as file:
            file.write(
                "iteration "
                + str(count)
                + "\n"
                + "times="
                + str((time.time() - start_time))
                + "\n"
                + "i="
                + str(i)
                + "\n"
                + "j="
                + str(j)
                + "\n"
                + "Fitness="
                + str(fitness)
                + "\n"
            )

best = numpy.min(fitness)
print("best=", best)
