from sklearn.decomposition import PCA
from boundary_detection import load_features
from boundary_detection import FeatureNames
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate


def pca():
    feature_names = FeatureNames.feature_names
    X, Y = load_features.load_features()
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)
    pca = PCA()
    pca.fit(X_norm)
    evr = pca.explained_variance_
    variance = pca.explained_variance_ratio_
    sv = pca.singular_values_
    for i in range(len(feature_names) - 7):
        print(i + 1)
        print("variance(%):", variance[i] * 100)
        print("eigenvalue:", evr[i])
        print("---------------------------")

    # number of components
    n_pcs = pca.components_.shape[0]

    # get the index of the most important feature on EACH component
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

    most_important_names = [feature_names[most_important[i]] for i in range(n_pcs)]

    dic = {"PC{}".format(i): most_important_names[i] for i in range(n_pcs)}

    important_features = pd.DataFrame(dic.items(), columns=["pc", "important feature"])
    important_features["variance"] = variance * 100
    important_features["eigenvalue"] = evr
    print(important_features)


def dim_reduction_plot():
    feature_names = FeatureNames.feature_names
    X, y = load_features.load_features()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Lists to store scores
    accuracy_scores = []
    macro_avg_scores = []

    # Train the model and reduce dimensions each time
    for n_components in range(14, 0, -1):
        print("reducing dimension to " + str(n_components))
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        print("dimensions reduced")

        # Random Forest model
        model = RandomForestClassifier(random_state=42)

        # Perform 10-fold cross-validation
        scoring = ["accuracy", "f1_macro"]
        cv_results = cross_validate(model, X_reduced, y, cv=10, scoring=scoring)
        print("cross-validation done")

        # Store the mean scores
        accuracy_scores.append(cv_results["test_accuracy"].mean())
        macro_avg_scores.append(cv_results["test_f1_macro"].mean())

    # Plot Accuracy chart
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(14, 0, -1), accuracy_scores, marker="o", color="blue", label="Accuracy"
    )
    plt.title("Accuracy vs. Number of Features (10-Fold CV)")
    plt.xlabel("Number of Features")
    plt.ylabel("Average Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig("boundary_detection/plots/accuracy_plot.png")  # Save plot as PNG
    plt.show()

    # Plot Macro Average F1-Score chart
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(14, 0, -1),
        macro_avg_scores,
        marker="o",
        color="green",
        label="Macro Average F1",
    )
    plt.title("Macro Average F1-Score vs. Number of Features (10-Fold CV)")
    plt.xlabel("Number of Features")
    plt.ylabel("Macro Average F1-Score")
    plt.grid(True)
    plt.legend()
    plt.savefig("boundary_detection/plots/macro_avg_plot.png")  # Save plot as PNG
    plt.show()
