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
from metadata_classification import load_features
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from statistics import stdev
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from libtsvm.estimators import LSTSVM
from libtsvm.mc_scheme import OneVsOneClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    Input,
    BatchNormalization,
)
from sklearn.model_selection import KFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import itertools

labels = [
    "other",
    "university",
    "faculty",
    "degree & field",
    "date",
    "title",
    "professor",
    "author",
    "abstract",
    "keywords",
]


def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=2, activation="relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.29287))
    model.add(Dense(len(labels), activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=0.00044),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def neural_network():

    X, Y = load_features.load_metadata_features()
    NUM_FEATURES = X.shape[1] - 1

    # Split data for k-fold cross-validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(X)

    # Training the model using k-fold cross-validation
    path = "metadata_classification/deep_results/"
    all_accs = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    # Initialize arrays to store per-class metrics and cumulative confusion matrix
    class_precisions = []
    class_recalls = []
    class_f1s = []
    cumulative_conf_matrix = None

    max_acc = 0
    counter = 1

    start = time.time()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        print(f"Training fold {counter}...")

        # Normalize the features
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Reshape for CNN input (samples, timesteps, features)
        X_train = X_train.reshape(X_train.shape[0], NUM_FEATURES, 1)
        X_test = X_test.reshape(X_test.shape[0], NUM_FEATURES, 1)

        model = create_cnn_model((NUM_FEATURES, 1))

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # Evaluate the model
        accuracy = model.evaluate(X_test, y_test)[1]
        all_accs.append(accuracy)

        print(f"Fold {counter} accuracy: {accuracy}")

        # Predictions and evaluation
        y_pred = model.predict(X_test).argmax(axis=1)

        # Calculate macro-averaged precision, recall, f1-score
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        # Calculate per-class metrics
        class_precisions.append(precision_score(y_test, y_pred, average=None))
        class_recalls.append(recall_score(y_test, y_pred, average=None))
        class_f1s.append(f1_score(y_test, y_pred, average=None))

        # Calculate the confusion matrix for the current fold
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Accumulate the confusion matrix
        if cumulative_conf_matrix is None:
            cumulative_conf_matrix = conf_matrix
        else:
            cumulative_conf_matrix += conf_matrix

        conf_df = pd.DataFrame(conf_matrix, index=np.unique(Y), columns=np.unique(Y))

        # Save evaluation metrics
        with open(f"{path}evaluation.txt", "a") as file:
            file.write(f"In fold {counter}\n")
            file.write(f"Confusion matrix:\n{conf_df}\n")
            file.write(
                f"Classification report:\n{classification_report(y_test, y_pred, target_names=labels)}\n"
            )
            file.write("-" * 70 + "\n")

        counter += 1

    # Calculate macro average over all folds
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)
    avg_acc = np.mean(all_accs)

    print(f"Macro average precision: {avg_precision}")
    print(f"Macro average recall: {avg_recall}")
    print(f"Macro average F1-score: {avg_f1}")
    print(f"Average accuracy: {avg_acc}")
    print(f"Training time: {time.time() - start} seconds")

    # Calculate average precision, recall, and F1-score for each class across all folds
    avg_class_precisions = np.mean(class_precisions, axis=0)
    avg_class_recalls = np.mean(class_recalls, axis=0)
    avg_class_f1s = np.mean(class_f1s, axis=0)

    # Create a DataFrame for per-class metrics
    class_labels = np.unique(Y)  # Assuming Y contains the labels
    class_metrics_df = pd.DataFrame(
        {
            "Class": labels,
            "Precision": avg_class_precisions,
            "Recall": avg_class_recalls,
            "F1 Score": avg_class_f1s,
        },
        index=class_labels,
    )

    print("Per-class metrics:")
    print(class_metrics_df)

    with open(f"{path}evaluation.txt", "a") as file:
        file.write(f"Macro average precision: {avg_precision}\n")
        file.write(f"Macro average recall: {avg_recall}\n")
        file.write(f"Macro average F1-score: {avg_f1}\n")
        file.write(f"Average accuracy: {avg_acc}\n")
        file.write(f"Per-class average precision:\n{class_metrics_df['Precision']}\n")
        file.write(f"Per-class average recall:\n{class_metrics_df['Recall']}\n")
        file.write(f"Per-class average F1-score:\n{class_metrics_df['F1 Score']}\n")

    # Plot and save the cumulative confusion matrix
    plt.figure(figsize=(10, 7))
    plt.imshow(cumulative_conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Cumulative Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    cumulative_conf_matrix_normalized = (
        cumulative_conf_matrix.astype("float")
        / cumulative_conf_matrix.sum(axis=1)[:, np.newaxis]
    )
    cumulative_conf_matrix_normalized *= 100  # Convert to percentage

    # Plot the normalized confusion matrix
    plt.imshow(
        cumulative_conf_matrix_normalized, interpolation="nearest", cmap=plt.cm.Blues
    )
    plt.title("Cumulative Confusion Matrix (Percentage)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    thresh = cumulative_conf_matrix_normalized.max() / 2.0
    for i, j in itertools.product(
        range(cumulative_conf_matrix_normalized.shape[0]),
        range(cumulative_conf_matrix_normalized.shape[1]),
    ):
        plt.text(
            j,
            i,
            f"{cumulative_conf_matrix_normalized[i, j]:.1f}%",
            horizontalalignment="center",
            color=(
                "white" if cumulative_conf_matrix_normalized[i, j] > thresh else "black"
            ),
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f"{path}cumulative_confusion_matrix.png")
    plt.close()


def classic_machine_learning_model(model, path="metadata_classification/", k=5):
    all_accs = []
    all_recs = []
    all_pres = []
    all_f1s = []
    max_acc = 0
    start = time.time()

    X, Y = load_features.load_metadata_features()

    kf = KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(X)

    class_num = len(labels)

    avg_acc = 0
    avg_precision = 0
    avg_recall = 0
    avg_f1_score = 0
    avg_acc_per_class = 0
    cumulative_conf_matrix = np.zeros(
        (class_num, class_num)
    )  # Initialize cumulative confusion matrix

    counter = 1

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        print(f"In fold {counter}")

        print("Normalizing")

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_norm = scaler.transform(X_train)

        print("Training....")
        model.fit(X_train_norm, y_train)

        print("Testing...")
        X_test_norm = scaler.transform(X_test)
        predict = model.predict(X_test_norm)

        classification_rep = classification_report(y_test, predict, target_names=labels)
        print(classification_rep)

        accuracy = accuracy_score(y_test, predict)
        precision = precision_score(y_test, predict, average=None)
        recall = recall_score(y_test, predict, average=None)
        f1score = f1_score(y_test, predict, average=None)

        all_accs.append(accuracy)
        all_pres.append(sum(precision) / len(precision))
        all_recs.append(sum(recall) / len(recall))
        all_f1s.append(sum(f1score) / len(f1score))

        conf_matrix = confusion_matrix(
            y_test, predict, labels=list(range(0, class_num))
        )
        conf = pd.DataFrame(conf_matrix, labels, labels)
        cumulative_conf_matrix += conf_matrix

        # Get accuracy scores for each class from confusion matrix
        conf_matrix = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )
        acc_per_class = conf_matrix.diagonal()

        with open(path + "evaluation.txt", "a") as file:
            file.write(
                f"In fold {counter}\n"
                + "confusion matrix:\n"
                + conf.to_string()
                + "\n\n"
                + "classification report:\n"
                + classification_rep
                + "\n"
                + "accuracy of each class:\n"
                + pd.DataFrame(
                    np.array(acc_per_class).transpose(),
                    columns=["accuracy"],
                    index=labels,
                ).to_string()
                + "\n------------------------------------------------------------------\n"
            )

        avg_acc += accuracy
        avg_precision += precision
        avg_recall += recall
        avg_f1_score += f1score
        avg_acc_per_class += acc_per_class
        counter += 1

    # Calculate and display average performance metrics
    avg_acc /= k
    avg_precision /= k
    avg_recall /= k
    avg_f1_score /= k
    avg_acc_per_class /= k

    # Plot and save cumulative confusion matrix
    cumulative_conf_matrix = cumulative_conf_matrix.astype(
        int
    )  # Ensure integers for confusion matrix
    plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cumulative_conf_matrix, display_labels=labels
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.title("Cumulative Confusion Matrix")
    plt.savefig(f"{path}Cumulative_Confusion_Matrix.png")
    plt.clf()

    df = pd.DataFrame(
        np.array(avg_precision).transpose(), columns=["average precision"], index=labels
    )
    df["average recall"] = pd.Series(np.array(avg_recall).transpose(), index=df.index)
    df["average f1 score"] = pd.Series(
        np.array(avg_f1_score).transpose(), index=df.index
    )
    df["average accuracy"] = pd.Series(
        np.array(avg_acc_per_class).transpose(), index=df.index
    )
    print(df)

    print("average accuracy=", avg_acc)
    print("average precision=", sum(avg_precision) / len(avg_precision))
    print("average recall=", sum(avg_recall) / len(avg_recall))
    print("average f1 score=", sum(avg_f1_score) / len(avg_f1_score))

    print(f"accuracy standard deviation: {stdev(all_accs)}")
    print(f"precision standard deviation: {stdev(all_pres)}")
    print(f"recall standard deviation: {stdev(all_recs)}")
    print(f"f1 score standard deviation: {stdev(all_f1s)}")

    print("time=", time.time() - start, "seconds")

    with open(path + "evaluation.txt", "a") as file:
        file.write("average accuracy=" + str(avg_acc) + "\n" + df.to_string())


def train_random_forest(n_estimators=100, max_depth=None):
    classic_machine_learning_model(
        RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    )


def train_svm(C=16, kernel="rbf", gamma=2**-4):
    classic_machine_learning_model(svm.SVC(C=C, kernel=kernel, gamma=gamma))


def train_naive_bayes():
    classic_machine_learning_model(GaussianNB())


def train_twin_svm(kernel="RBF", C1=2**3, C2=2**3, gamma=2**-3):
    lstsvm_clf = LSTSVM(kernel=kernel, C1=C1, C2=C2, gamma=gamma)
    clf = OneVsOneClassifier(lstsvm_clf)
    classic_machine_learning_model(clf)
