import csv
import optuna
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
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from metadata_classification import load_features
import numpy as np
from sklearn.model_selection import KFold

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
X, Y, file_names = load_features.load_metadata_features()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_indices = list(kf.split(X))
NUM_FEATURES = X.shape[1] - 1
input_shape = (NUM_FEATURES, 1)


def objective(trial):
    filters_1 = trial.suggest_categorical("filter1", [64, 128, 256, 512])
    filters_2 = trial.suggest_categorical("filter2", [64, 128, 256, 512])
    kernel_size = trial.suggest_int("kernel_size", 1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.4)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

    fold_accuracies = []

    for train_index, val_index in fold_indices:
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]
        X_train = X_train[:, :-1]
        X_val = X_val[:, :-1]
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Conv1D(filters=filters_1, kernel_size=kernel_size, activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=filters_2, kernel_size=kernel_size, activation="relu"))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(len(labels), activation="softmax"))
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        val_accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
        fold_accuracies.append(val_accuracy)

    return np.mean(fold_accuracies)


def bayesian_optimization():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    with open("optuna_all_trials_results.csv", "w", newline="") as csvfile:
        fieldnames = [
            "trial_number",
            "mean_accuracy",
            "filters",
            "kernel_size",
            "dropout_rate",
            "learning_rate",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for trial in study.trials:
            writer.writerow(
                {
                    "trial_number": trial.number,
                    "mean_accuracy": trial.value,
                    "filters": trial.params["filters"],
                    "kernel_size": trial.params["kernel_size"],
                    "dropout_rate": trial.params["dropout_rate"],
                    "learning_rate": trial.params["learning_rate"],
                }
            )
