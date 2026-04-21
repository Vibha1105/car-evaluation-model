"""
Car Evaluation Predictive Analysis Dashboard
Backend server using Flask - replicates the exact Colab notebook model training.
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    r2_score, mean_squared_error
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

app = Flask(__name__)

# ──────────────────────────────────────────────
# GLOBAL STATE - trained once at startup
# ──────────────────────────────────────────────
DATA = {}


def train_all_models():
    """
    Replicates the exact Colab notebook pipeline:
      1. Load cars.csv
      2. Fill NaN with mode (mirrors df.loc[0,'buying']=np.nan then fillna)
      3. LabelEncode every column
      4. Train/test split (80/20, random_state=42, stratify=y)
      5. Train KNN, Decision Tree, Naive Bayes, SVM, Random Forest
      6. Linear & Polynomial Regression
      7. K-Means Clustering
      8. PCA
    """
    csv_path = os.path.join(os.path.dirname(__file__), "cars.csv")
    df_original = pd.read_csv(csv_path)

    # -- keep a copy of original string data for the dashboard ---------------
    DATA["df_original"] = df_original.copy()
    DATA["feature_names"] = [c for c in df_original.columns if c != "class"]
    DATA["feature_options"] = {
        col: sorted(df_original[col].dropna().unique().tolist())
        for col in DATA["feature_names"]
    }
    DATA["class_distribution"] = df_original["class"].value_counts().to_dict()

    # -- replicate Colab: inject one NaN then fill with mode -----------------
    df = df_original.copy()
    df.loc[0, "buying"] = np.nan
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # -- LabelEncode (per-column encoders stored for prediction) -------------
    label_encoders = {}
    df_encoded = df.copy()
    for col in df.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    DATA["label_encoders"] = label_encoders
    DATA["df_encoded"] = df_encoded
    DATA["encoded_class_names"] = list(label_encoders["class"].classes_)

    # -- Features / Target ---------------------------------------------------
    X = df_encoded.drop("class", axis=1)
    y = df_encoded["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    DATA["X"] = X
    DATA["y"] = y
    DATA["X_train"] = X_train
    DATA["X_test"] = X_test
    DATA["y_train"] = y_train
    DATA["y_test"] = y_test

    # -- Train all classification models (exactly like Colab) ----------------
    classifiers = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    model_results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True,
                                       target_names=DATA["encoded_class_names"],
                                       zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        model_results[name] = {
            "accuracy": round(acc, 4),
            "confusion_matrix": cm,
            "report": report,
        }

    # Cross-validation for Random Forest (exactly like Colab)
    rf_model = classifiers["Random Forest"]
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="accuracy")
    model_results["Random Forest"]["cv_scores"] = [round(s, 4) for s in cv_scores]
    model_results["Random Forest"]["cv_mean"] = round(cv_scores.mean(), 4)
    model_results["Random Forest"]["cv_std"] = round(cv_scores.std(), 4)

    DATA["classifiers"] = classifiers
    DATA["model_results"] = model_results

    # -- Feature importance from Random Forest -------------------------------
    importances = rf_model.feature_importances_
    DATA["feature_importance"] = {
        col: round(float(imp), 4)
        for col, imp in zip(X.columns, importances)
    }

    # -- Linear & Polynomial Regression (exactly like Colab) -----------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    pr = LinearRegression()
    pr.fit(X_train_poly, y_train)
    y_pred_pr = pr.predict(X_test_poly)

    DATA["regression"] = {
        "linear": {
            "r2": round(r2_score(y_test, y_pred_lr), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_lr))), 4),
        },
        "polynomial": {
            "r2": round(r2_score(y_test, y_pred_pr), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_pr))), 4),
        },
    }

    # -- K-Means Clustering (exactly like Colab) -----------------------------
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(round(float(kmeans.inertia_), 2))
    DATA["kmeans_wcss"] = wcss

    kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(X)
    DATA["clusters"] = clusters.tolist()

    # -- PCA (exactly like Colab) --------------------------------------------
    X_onehot = pd.get_dummies(df.drop("class", axis=1))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_onehot)
    DATA["pca_points"] = X_pca.tolist()
    DATA["pca_variance"] = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    # -- Correlation matrix --------------------------------------------------
    corr = df_encoded.corr(method="pearson")
    DATA["correlation"] = {
        "columns": corr.columns.tolist(),
        "values": np.round(corr.values, 3).tolist(),
    }

    # -- Per-feature distributions -------------------------------------------
    feature_distributions = {}
    for col in DATA["feature_names"]:
        cross = pd.crosstab(df_original[col], df_original["class"])
        feature_distributions[col] = {
            "labels": cross.index.tolist(),
            "classes": cross.columns.tolist(),
            "data": cross.values.tolist(),
        }
    DATA["feature_distributions"] = feature_distributions

    # -- Descriptive stats ---------------------------------------------------
    DATA["desc_stats"] = {
        "mean": df_encoded.mean().round(3).to_dict(),
        "median": df_encoded.median().round(3).to_dict(),
        "std": df_encoded.std().round(3).to_dict(),
    }

    print("[OK] All models trained successfully!")


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard-data")
def dashboard_data():
    """Return all pre-computed analytics for the frontend."""
    return jsonify({
        "class_distribution": DATA["class_distribution"],
        "feature_options": DATA["feature_options"],
        "feature_names": DATA["feature_names"],
        "encoded_class_names": DATA["encoded_class_names"],
        "model_results": DATA["model_results"],
        "feature_importance": DATA["feature_importance"],
        "regression": DATA["regression"],
        "kmeans_wcss": DATA["kmeans_wcss"],
        "correlation": DATA["correlation"],
        "feature_distributions": DATA["feature_distributions"],
        "desc_stats": DATA["desc_stats"],
        "pca_variance": DATA["pca_variance"],
        "total_samples": len(DATA["df_original"]),
        "total_features": len(DATA["feature_names"]),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with feature values (original string labels).
    Encodes them using the stored LabelEncoders and predicts with Random Forest.
    """
    payload = request.get_json()
    le_dict = DATA["label_encoders"]

    try:
        encoded_row = []
        for col in DATA["feature_names"]:
            val = payload.get(col)
            if val is None:
                return jsonify({"error": f"Missing feature: {col}"}), 400
            encoded_val = le_dict[col].transform([val])[0]
            encoded_row.append(encoded_val)

        X_input = np.array(encoded_row).reshape(1, -1)

        # Predict with all classifiers
        predictions = {}
        for name, clf in DATA["classifiers"].items():
            pred_encoded = clf.predict(X_input)[0]
            pred_label = le_dict["class"].inverse_transform([pred_encoded])[0]
            predictions[name] = pred_label

            # Probabilities where available
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_input)[0]
                predictions[name + "_probs"] = {
                    le_dict["class"].inverse_transform([i])[0]: round(float(p), 4)
                    for i, p in enumerate(probs)
                }

        # Primary prediction from Random Forest
        rf_pred = predictions["Random Forest"]
        rf_probs = predictions.get("Random Forest_probs", {})

        return jsonify({
            "prediction": rf_pred,
            "probabilities": rf_probs,
            "all_models": {
                name: predictions[name]
                for name in DATA["classifiers"]
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/feature-distribution/<feature>")
def feature_distribution(feature):
    """Return distribution data for a specific feature vs class."""
    if feature in DATA["feature_distributions"]:
        return jsonify(DATA["feature_distributions"][feature])
    return jsonify({"error": "Feature not found"}), 404


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    train_all_models()
    app.run(debug=False, port=5000)
