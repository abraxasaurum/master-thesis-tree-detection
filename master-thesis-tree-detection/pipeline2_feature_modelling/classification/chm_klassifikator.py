#!/usr/bin/env python3
"""
Trainingsskript für Baumhöhenmodelle (chm_max & chm_mean)
Speichert Modelle, Skalierer und Feature-Liste für Inferenz
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import joblib
import os

# Zentrales Ausgabeverzeichnis
OUTPUT_DIR = "/home/abrax/Desktop/convert_pkl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# 📦 Daten laden und vorbereiten
# ================================
df = pd.read_csv("/home/abrax/Training/results/full_features_per_polygon.csv")
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ➕ Zusätzliche Features
df['spectral_diversity'] = df['band_1_std'] + df['band_2_std'] + df['band_3_std'] + df['band_4_std']
df['vegetation_strength'] = (df['ndvi_mean'] + df['evi'] + df['savi']) / 3
df['texture_complexity'] = (
    df['band1_contrast'] + df['band2_contrast'] +
    df['band3_contrast'] + df['band4_contrast']
) / 4

# ================================
# 📈 Trainingsfunktion
# ================================
def train_height_model(target_col):
    print(f"\n🔧 Trainiere Modell für: {target_col}")

    drop_cols = [
        "fid", "tile", "ba", "ba_text", "zustand",
        "chm_max", "chm_mean", "chm_std", "chm_skew", "chm_kurt",
        "ndvi_chm", "evi_chm"
    ]
    features = [col for col in df.columns if col not in drop_cols]

    X = df[features]
    y = df[target_col]

    # Skaliere Features robust gegen Ausreißer
    scaler = RobustScaler(quantile_range=(5, 95))
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Modell mit erweiterten Parametern
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    # Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"📊 Cross-Validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Training
    model.fit(X_train, y_train)

    # Dateibasis
    base_path = f"{OUTPUT_DIR}/model_{target_col}"

    # 📝 Featureliste speichern
    with open(f"{base_path}_features.txt", "w") as f:
        for col in X.columns:
            f.write(col + "\n")

    y_pred = model.predict(X_test)

    # Evaluation
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"✅ Modell für {target_col} trainiert")
    print(f"📈 R² Score: {r2:.3f}")
    print(f"📉 RMSE: {rmse:.2f} m")
    print(f"📊 MAE: {mae:.2f} m")
    print(f"📋 MAPE: {mape:.1f}%")

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🔝 Top 10 wichtigste Features:")
    print(feature_importance.head(10))

    # Modell + Skalierer speichern
    joblib.dump(model, f"{base_path}_model.pkl")
    joblib.dump(scaler, f"{base_path}_scaler.pkl")
    print(f"💾 Modell gespeichert: {base_path}_model.pkl")
    print(f"💾 Skalierer gespeichert: {base_path}_scaler.pkl")

# ================================
# 🔁 Modelle trainieren
# ================================
train_height_model("chm_max")
train_height_model("chm_mean")

