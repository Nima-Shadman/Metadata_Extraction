from sklearn.decomposition import PCA
from metadata_classification import load_features
from metadata_classification import FeatureNames
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from metadata_classification import FeatureNames as FN
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from metadata_classification.models import create_cnn_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def dim_reduction_plot():
    feature_names = FeatureNames.feature_names
    X, y = load_features.load_metadata_features()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Lists to store scores
    accuracy_scores = []
    macro_avg_scores = []

    # Stratified K-Fold for better cross-validation
    cv = StratifiedKFold(n_splits=5)

    # Train the model and reduce dimensions each time
    for n_components in range(38, 0, -1):
        print("reducing dimension to " + str(n_components))
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        X_reduced = X_reduced.reshape(-1, n_components, 1)  # Adjust dimensions for CNN
        print("dimensions reduced")

        # Wrap CNN model using KerasClassifier
        model = KerasClassifier(
            build_fn=create_cnn_model,
            input_shape=(n_components, 1),
            epochs=10,
            batch_size=32,
            verbose=0,
        )

        # Perform 5-fold cross-validation
        scoring = ["accuracy", "f1_macro"]
        cv_results = cross_validate(model, X_reduced, y, cv=cv, scoring=scoring)
        print("cross-validation done")

        # Store the mean scores
        accuracy_scores.append(cv_results["test_accuracy"].mean())
        macro_avg_scores.append(cv_results["test_f1_macro"].mean())

    # Plot Accuracy chart
    plt.figure(figsize=(12, 10))
    plt.plot(
        range(38, 0, -1), accuracy_scores, marker="o", color="blue", label="Accuracy"
    )
    plt.title("Accuracy vs. Number of Features")
    plt.xlabel("Number of Features")
    plt.ylabel("Average Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig("accuracy_plot.png")  # Save plot as PNG
    plt.show()

    # Plot Macro Average F1-Score chart
    plt.figure(figsize=(12, 10))
    plt.plot(
        range(38, 0, -1),
        macro_avg_scores,
        marker="o",
        color="green",
        label="Macro Average F1",
    )
    plt.title("Macro Average F1-Score vs. Number of Features")
    plt.xlabel("Number of Features")
    plt.ylabel("Macro Average F1-Score")
    plt.grid(True)
    plt.legend()
    plt.savefig("macro_avg_plot.png")  # Save plot as PNG
