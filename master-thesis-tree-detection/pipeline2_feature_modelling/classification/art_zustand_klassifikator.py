#!/usr/bin/env python3
"""
Trainingsskript für Baumart- und Zustandsklassifikator
Speichert Modelle, Skalierer und Featurelisten für Inferenz
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

# === Zentrales Ausgabeverzeichnis ===
OUTPUT_DIR = "/home/abrax/Desktop/convert_pkl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Daten laden und vorbereiten ===
df = pd.read_csv("/home/abrax/Training/results/full_features_per_polygon.csv")
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# === Erweiterte Features berechnen ===
df["spectral_diversity"] = df["band_1_std"] + df["band_2_std"] + df["band_3_std"] + df["band_4_std"]
df["vegetation_strength"] = (df["ndvi_mean"] + df["evi"] + df["savi"]) / 3
df["texture_complexity"] = (
    df["band1_contrast"] + df["band2_contrast"] +
    df["band3_contrast"] + df["band4_contrast"]
) / 4

# === Klassifikator 1: Baumart ("ba_text") ===
def train_baumart_classifier():
    print("\n🌲 Trainiere Baumartenklassifikator ...")
    target = "ba_text"
    drop_cols = ["fid", "tile", "ba", "ba_text", "zustand"]
    features = [col for col in df.columns if col not in drop_cols]

    X = df[features]
    y = df[target]

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [500],
        "max_depth": [25],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring="f1_weighted"
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    print(f"🏆 Beste Parameter (Baumart): {grid.best_params_}")

    # 📝 Featureliste speichern
    with open(f"{OUTPUT_DIR}/model_baumart_optimized_features.txt", "w") as f:
        for col in X.columns:
            f.write(col + "\n")

    y_pred = model.predict(X_test)
    print("📈 Accuracy:", accuracy_score(y_test, y_pred))
    print("📊 F1:", f1_score(y_test, y_pred, average="weighted"))
    print("📋 Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("📉 Recall:", recall_score(y_test, y_pred, average="weighted"))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    print("\n🎯 Wichtigste Features (Top 5):")
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(fi.head(5))

    joblib.dump(model, f"{OUTPUT_DIR}/model_baumart_optimized.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler_baumart.pkl")
    print("💾 Modell & Scaler gespeichert (Baumart)")

# === Klassifikator 2: Zustand ===
def train_zustand_classifier():
    print("\n🩺 Trainiere Zustandsklassifikator ...")
    target = "zustand"
    drop_cols = ["fid", "tile", "ba", "ba_text", "zustand"]
    features = [col for col in df.columns if col not in drop_cols]

    X = df[features]
    y = df[target]

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE für Balancing
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("📊 Klassenverteilung nach SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_balanced, y_train_balanced)

    # 📝 Featureliste speichern
    with open(f"{OUTPUT_DIR}/model_zustand_smote_features.txt", "w") as f:
        for col in X.columns:
            f.write(col + "\n")

    y_pred = model.predict(X_test)
    print("📈 Accuracy:", accuracy_score(y_test, y_pred))
    print("📊 F1:", f1_score(y_test, y_pred, average="weighted"))
    print("📋 Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("📉 Recall:", recall_score(y_test, y_pred, average="weighted"))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    print("\n🔗 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    for i, row in enumerate(cm):
        print(f"True {i+1}: {row}")

    print("\n🎯 Wichtigste Features (Top 5):")
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(fi.head(5))

    joblib.dump(model, f"{OUTPUT_DIR}/model_zustand_smote.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler_zustand.pkl")
    print("💾 Modell & Scaler gespeichert (Zustand)")

# === Hauptfunktion aufrufen ===
if __name__ == "__main__":
    train_baumart_classifier()
    train_zustand_classifier()

