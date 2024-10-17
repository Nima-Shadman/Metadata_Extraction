from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    classification_report,
    f1_score,
    confusion_matrix,
)
from boundary_detection.load_features import load_features
import numpy
import time
from joblib import dump
import pandas
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from libtsvm.estimators import LSTSVM, TSVM


def train_model(model):
    max_acc = 0
    start = time.time()
    path = "boundary_detection/"

    X, Y, file_names = load_features()

    k = 10
    kf = KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(X)

    labels = ["metadata", "body"]
    class_num = len(labels)

    avg_acc = 0
    avg_precision = 0
    avg_recall = 0
    avg_f1_score = 0
    avg_acc_per_class = 0
    cumulative_conf_matrix = numpy.zeros(
        (class_num, class_num)
    )  # Initialize cumulative confusion matrix

    counter = 1

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        test_name = file_names[test_index]

        print("in fold " + str(counter))

        print("Normalizing...")
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_norm = scaler.transform(X_train)

        print("training....")
        model.fit(X_train_norm, y_train)
        print("testing...")
        X_test_norm = scaler.transform(X_test)
        predict = model.predict(X_test_norm)

        classification_rep = classification_report(y_test, predict, target_names=labels)
        print(classification_rep)
        accuracy = accuracy_score(y_test, predict)
        precision = precision_score(y_test, predict, average=None)
        recall = recall_score(y_test, predict, average=None)
        f1score = f1_score(y_test, predict, average=None)

        conf_matrix = confusion_matrix(
            y_test, predict, labels=list(range(0, class_num))
        )
        conf = pandas.DataFrame(conf_matrix, labels, labels)
        cumulative_conf_matrix += conf_matrix

        # Normalize diagonal entries of confusion matrix
        conf_matrix = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, numpy.newaxis]
        )
        acc_per_class = conf_matrix.diagonal()

        with open(path + "evaluation_40.txt", "a") as file:
            file.write(
                "In fold"
                + str(counter)
                + "\n"
                + "confusion matrix:\n"
                + conf.to_string()
                + "\n\n"
                + "classification report:\n"
                + classification_rep
                + "\n"
                + "accuracy of each class:\n"
                + pandas.DataFrame(
                    numpy.array(acc_per_class).transpose(),
                    columns=["accuracy"],
                    index=labels,
                ).to_string()
                + "\n------------------------------------------------------------------\n"
            )
        print("accuracy=", accuracy)
        print("precision=", precision)
        print("recall=", recall)
        print("f1 score", f1score)

        print("number of mislabels=", numpy.sum(predict != y_test))

        avg_acc += accuracy
        avg_precision += precision
        avg_recall += recall
        avg_f1_score += f1score
        avg_acc_per_class += acc_per_class

        counter += 1

    avg_acc /= k
    avg_precision /= k
    avg_recall /= k
    avg_f1_score /= k
    avg_acc_per_class /= k
    print("average f1 score=", sum(avg_f1_score) / 2)
    df = pandas.DataFrame(
        numpy.array(avg_precision).transpose(),
        columns=["average precision"],
        index=labels,
    )
    df["average recall"] = pandas.Series(
        numpy.array(avg_recall).transpose(), index=df.index
    )
    df["average f1 score"] = pandas.Series(
        numpy.array(avg_f1_score).transpose(), index=df.index
    )
    df["average accuracy"] = pandas.Series(
        numpy.array(avg_acc_per_class).transpose(), index=df.index
    )

    with open(path + "evaluation.txt", "a") as file:
        file.write("average accuracy=" + str(avg_acc) + "\n" + df.to_string())

    end = time.time()
    print(end - start, "seconds")

    cumulative_conf_matrix = cumulative_conf_matrix.astype(int)
    plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cumulative_conf_matrix, display_labels=labels
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Cumulative Confusion Matrix")
    plt.savefig(f"{path}Cumulative_Confusion_Matrix.png")
    plt.clf()


def train_svm(C=16, kernel="rbf", gamma=2**-4):
    clf = svm.SVC(C=C, kernel=kernel, gamma=gamma)
    train_model(clf)


def train_random_forest(n_estimators=100, max_depth=None):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    train_model(clf)


def train_knn(n_neighbors=3):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    train_model(clf)


def train_naive_bayes():
    clf = GaussianNB()
    train_model(clf)


def train_decision_tree():
    clf = DecisionTreeClassifier()
    train_model(clf)


def train_logistic_regression(max_iter=1000):
    clf = LogisticRegression(max_iter=max_iter)
    train_model(clf)


def train_twin_svm(kernel="RBF", C1=16, C2=16, gamma=2**-4):
    clf = TSVM(kernel=kernel, C1=C1, C2=C2, gamma=gamma)
    train_model(clf)
