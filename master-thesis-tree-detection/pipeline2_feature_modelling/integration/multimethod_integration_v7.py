#!/usr/bin/env python3

"""
Multi-Method Integration Script v8 – TCD + Detectree2 + DeepTree
Kombiniert:
- TCD: Feature-Extraktion + ML-Prediction + Art-spezifische Allometrie + TP/FP
- Detectree2: Feature-Extraktion + ML-Prediction + Art-spezifische Allometrie + TP/FP
- DeepTree: Platzhalter für spätere Integration
- Online-Statistiken während des Laufs (speichereffizient)
- Vergleichende Statistiken über alle Methoden
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import mapping, shape
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
import joblib
import sqlite3
from shapely import wkt
import fiona
import pyproj.datadir
import math
import warnings
import json
from collections import defaultdict
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# PROJ Data Directory setzen
conda_proj_dir = "/home/abrax/miniconda3/envs/tcd_hsh_working_2/share/proj"
if os.path.exists(conda_proj_dir):
    pyproj.datadir.set_data_dir(conda_proj_dir)
    print("✅ PROJ Data Dir korrigiert")

# CRS-Test
try:
    import pyproj
    test_crs = pyproj.CRS.from_epsg(25833)
    print("✅ CRS EPSG:25833 funktioniert")
except Exception as e:
    print(f"⚠️ CRS-Test fehlgeschlagen: {e}")


class OnlineStatistics:
    """Effiziente Online-Statistik-Sammlung ohne RAM-intensive Listen"""
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_val = np.inf
        self.max_val = -np.inf
        self.values_for_quantiles = []  # für Quantile (nur bei Bedarf)
    
    def add(self, value):
        if value is None or np.isnan(value):
            return
        value = float(value)
        self.count += 1
        self.sum += value
        self.sum_sq += value ** 2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        # Nur erste 10000 Werte für Quantile (RAM-Limit)
        if len(self.values_for_quantiles) < 10000:
            self.values_for_quantiles.append(value)
    
    def mean(self):
        return self.sum / self.count if self.count > 0 else np.nan
    
    def std(self):
        if self.count < 2:
            return np.nan
        mean = self.mean()
        variance = (self.sum_sq / self.count) - mean ** 2
        return np.sqrt(max(0, variance))
    
    def quantile(self, q):
        if not self.values_for_quantiles:
            return np.nan
        return np.quantile(self.values_for_quantiles, q)
    
    def to_dict(self):
        return {
            "count": self.count,
            "mean": self.mean(),
            "std": self.std(),
            "min": self.min_val if self.count > 0 else np.nan,
            "max": self.max_val if self.count > 0 else np.nan,
            "q25": self.quantile(0.25),
            "q50": self.quantile(0.50),
            "q75": self.quantile(0.75)
        }


class MultiMethodIntegrator:
    """
    Integriert mehrere Tree Detection Methoden mit speichereffizienten Online-Statistiken.
    """
    
    def __init__(self, model_dir="/home/abrax/Desktop/convert_pkl"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_lists = {}
        
        self.model_configs = {
            "baumart": {
                "model": f"{model_dir}/model_baumart_optimized.pkl",
                "scaler": f"{model_dir}/scaler_baumart.pkl",
                "features": f"{model_dir}/model_baumart_optimized_features.txt"
            },
            "zustand": {
                "model": f"{model_dir}/model_zustand_smote.pkl",
                "scaler": f"{model_dir}/scaler_zustand.pkl",
                "features": f"{model_dir}/model_zustand_smote_features.txt"
            }
        }
        
        # Art-spezifische Allometrie-Parameter (aus GT-Daten geschätzt)
        self.ALLOMETRY_DBH_PARAMS = {
            "Beech": {
                "height_only": {"alpha": 0.9930, "beta_h": 0.624},
                "crown_only": {"alpha": 2.359, "beta_c": 0.137},
                "height_crown": {"alpha": 0.582, "beta_h": 0.668, "beta_c": 0.144}
            },
            "Douglas Fir": {
                "height_only": {"alpha": -1.306, "beta_h": 1.492},
                "crown_only": {"alpha": 2.311, "beta_c": 0.333},
                "height_crown": {"alpha": -0.116, "beta_h": 0.934, "beta_c": 0.193}
            },
            "Pine": {
                "height_only": {"alpha": -0.918, "beta_h": 1.313},
                "crown_only": {"alpha": 2.497, "beta_c": 0.240},
                "height_crown": {"alpha": 0.590, "beta_h": 0.675, "beta_c": 0.199}
            }
        }
        
        # Mapping Klassencode ↔ Baumart-String
        self.species_mapping = {
            0: "Pine",
            1: "Douglas Fir",
            2: "Beech",
            "Pine": "Pine",
            "Douglas Fir": "Douglas Fir",
            "Beech": "Beech"
        }
        
        # Mapping Zustandscode
        self.condition_mapping = {
            0: "healthy",
            1: "damaged",
            2: "dead",
            "healthy": "healthy",
            "damaged": "damaged",
            "dead": "dead"
        }
        
        self.load_trained_models()
    
    def load_trained_models(self):
        """Lade trainierte ML-Modelle"""
        print("🔄 Lade bereits trainierte Modelle (Baumart/Zustand)...")
        for name, config in self.model_configs.items():
            try:
                self.models[name] = joblib.load(config["model"])
                self.scalers[name] = joblib.load(config["scaler"])
                with open(config["features"], "r") as f:
                    self.feature_lists[name] = [line.strip() for line in f.readlines()]
                print(f"✅ {name}: geladen")
            except Exception as e:
                print(f"❌ Fehler beim Laden von {name}: {e}")
    
    def _calc_bhd_from_height_crown(self, height, crown_area, species, mode="height_crown"):
        """
        BHD-Schätzung (in cm) aus Höhe und/oder Kronenfläche
        using art-spezifische log-log Allometrie-Parameter.
        
        Args:
            height (float): Baumhöhe in m
            crown_area (float): Kronenfläche in m²
            species (str): Baumart ("Pine", "Douglas Fir", "Beech")
            mode (str): "height_only", "crown_only", oder "height_crown"
        
        Returns:
            float: BHD in cm
        """
        params_species = self.ALLOMETRY_DBH_PARAMS.get(
            species, self.ALLOMETRY_DBH_PARAMS["Douglas Fir"]
        )
        params = params_species.get(mode, params_species["height_crown"])
        
        h = max(float(height), 0.1)
        A = max(float(crown_area), 0.1)
        
        if mode == "height_only":
            y_log = params["alpha"] + params["beta_h"] * np.log(h)
        elif mode == "crown_only":
            y_log = params["alpha"] + params["beta_c"] * np.log(A)
        else:  # "height_crown"
            y_log = (params["alpha"]
                     + params["beta_h"] * np.log(h)
                     + params["beta_c"] * np.log(A))
        
        dbh_cm = float(np.exp(y_log))
        return dbh_cm
    
    def calculate_glcm(self, texture_data):
        """GLCM Texturmetriken"""
        if texture_data.size < 4:
            return {'contrast': 0, 'homogeneity': 0, 'energy': 0, 'correlation': 0}
        glcm = graycomatrix(texture_data.reshape(-1, 1), [1], [0], symmetric=True, normed=True)
        return {
            'contrast': graycoprops(glcm, 'contrast')[0, 0],
            'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
            'energy': graycoprops(glcm, 'energy')[0, 0],
            'correlation': graycoprops(glcm, 'correlation')[0, 0]
        }
    
    def load_polygons_robust(self, polygon_file, method_name=""):
        """Robustes Laden von Polygonen (Fiona/GeoPandas/SQLite)"""
        print(f"📥 Lade {method_name} Polygone aus: {os.path.basename(polygon_file)}")
        
        gdf = None
        try:
            # Versuch 1: GeoPandas direkt
            gdf = gpd.read_file(polygon_file)
            print(f"✅ {len(gdf)} Polygone mit GeoPandas geladen")
        except Exception as gpd_error:
            try:
                # Versuch 2: Fiona
                with fiona.open(polygon_file) as src:
                    features, geometries = [], []
                    for feature in src:
                        try:
                            geom = shape(feature['geometry'])
                            if geom.is_valid and geom.geom_type in ['Polygon', 'MultiPolygon']:
                                geometries.append(geom)
                                features.append(feature['properties'])
                        except Exception:
                            continue
                gdf = gpd.GeoDataFrame(features, geometry=geometries)
                gdf.crs = "EPSG:25833"
                print(f"✅ {len(gdf)} Polygone mit Fiona geladen")
            except Exception as fiona_error:
                try:
                    # Versuch 3: SQLite
                    conn = sqlite3.connect(polygon_file)
                    tables = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type='table'", conn
                    )
                    table_name = tables['name'].iloc[0]
                    df_poly = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    conn.close()
                    df_poly['geometry'] = df_poly['geometry'].apply(wkt.loads)
                    gdf = gpd.GeoDataFrame(df_poly, geometry='geometry')
                    gdf.crs = "EPSG:25833"
                    print(f"✅ {len(gdf)} Polygone aus SQLite geladen")
                except Exception as sqlite_error:
                    print(f"❌ Alle Lade-Methoden fehlgeschlagen: {sqlite_error}")
                    return None, None
        
        if gdf is None or len(gdf) == 0:
            print(f"❌ Keine Polygone geladen")
            return None, None
        
        # Stelle sicher CRS
        if gdf.crs is None:
            gdf.crs = "EPSG:25833"
        else:
            gdf = gdf.to_crs("EPSG:25833")
        
        # ID-Spalte bestimmen
        id_column = None
        for candidate in ['id', 'fid', 'FID', 'objectid', 'tree_id', 'ID']:
            if candidate in gdf.columns:
                id_column = candidate
                break
        
        if not id_column:
            gdf['auto_id'] = range(len(gdf))
            id_column = 'auto_id'
        
        print(f"🔑 Verwende ID-Spalte: {id_column}")
        
        # Repariere ungültige Geometrien
        invalid_geoms = ~gdf.geometry.is_valid
        if invalid_geoms.any():
            print(f"⚠️ {invalid_geoms.sum()} ungültige Geometrien gefunden, repariere...")
            gdf.loc[invalid_geoms, 'geometry'] = gdf.loc[invalid_geoms, 'geometry'].buffer(0)
        
        return gdf, id_column
    
    def extract_features_from_polygons(self, gdf, id_column, rgbnir_tif, chm_tif, method_name=""):
        """Vollständige Feature-Extraktion (RGB-NIR + CHM + Indices + Texturen)"""
        print(f"🔍 Extrahiere {method_name} Features aus {len(gdf)} Polygonen...")
        
        results = []
        
        with rasterio.open(rgbnir_tif) as src_rgbnir, rasterio.open(chm_tif) as src_chm:
            print(f"📊 RGB+NIR Bands: {src_rgbnir.count}, CHM Bands: {src_chm.count}")
            
            for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc=f"{method_name} Feature-Extraktion"):
                feature = {
                    'fid': row[id_column],
                    'polygon_area_m2': row.geometry.area
                }
                
                try:
                    geom = [mapping(row.geometry)]
                    
                    # RGB-NIR und CHM Daten extrahieren
                    rgbnir_data, _ = rasterio.mask.mask(src_rgbnir, geom, crop=True)
                    chm_data, _ = rasterio.mask.mask(src_chm, geom, crop=True)
                    
                    # Masken für gültige Pixel
                    mask = (rgbnir_data != src_rgbnir.nodata).all(axis=0)
                    chm_mask = (chm_data != src_chm.nodata)[0]
                    
                    if not mask.any():
                        continue
                    
                    # Band-Daten mit Clipping
                    red = np.clip(rgbnir_data[0][mask], 0, 42000)
                    green = np.clip(rgbnir_data[1][mask], 0, 42000)
                    blue = np.clip(rgbnir_data[2][mask], 0, 42000)
                    nir = np.clip(rgbnir_data[3][mask], 0, 42000)
                    chm = chm_data[0][chm_mask]
                    
                    # 1. Band-Statistiken mit Quantilen
                    for band_idx in range(rgbnir_data.shape[0]):
                        band_data = np.clip(rgbnir_data[band_idx][mask], 0, 42000)
                        if band_data.size == 0:
                            continue
                        feature.update({
                            f"band_{band_idx+1}_mean": np.nanmean(band_data),
                            f"band_{band_idx+1}_std": np.nanstd(band_data),
                            f"band_{band_idx+1}_q25": np.nanquantile(band_data, 0.25),
                            f"band_{band_idx+1}_q75": np.nanquantile(band_data, 0.75),
                            f"band_{band_idx+1}_skew": pd.Series(band_data).skew(),
                            f"band_{band_idx+1}_kurt": pd.Series(band_data).kurtosis()
                        })
                    
                    # 2. Vegetationsindizes
                    ndvi = (nir - red) / (nir + red + 1e-6)
                    feature.update({
                        "ndvi_mean": np.nanmean(ndvi),
                        "ndvi_std": np.nanstd(ndvi)
                    })
                    
                    gndvi = (nir - green) / (nir + green + 1e-6)
                    feature["gndvi"] = np.nanmean(gndvi)
                    
                    L = 0.5
                    savi = ((nir - red) / (nir + red + L + 1e-6)) * (1 + L)
                    feature["savi"] = np.nanmean(savi)
                    
                    vari = (green - red) / (green + red - blue + 1e-6)
                    feature["vari"] = np.nanmean(vari)
                    
                    denominator_evi = nir + 6*red - 7.5*blue + 1
                    evi = np.where(np.abs(denominator_evi) < 1e-9, 0, 2.5 * (nir - red) / denominator_evi)
                    feature["evi"] = np.nanmean(evi)
                    
                    msi = nir / (red + 1e-6)
                    feature["msi"] = np.nanmean(msi)
                    
                    ndwi = (green - nir) / (green + nir + 1e-6)
                    feature["ndwi"] = np.nanmean(ndwi)
                    
                    # 3. Texturmetriken (GLCM)
                    for i, band in enumerate([red, green, blue, nir], 1):
                        if band.size < 10:
                            continue
                        band_norm = ((band - np.nanmin(band)) /
                                     (np.nanmax(band) - np.nanmin(band) + 1e-6) * 255).astype(np.uint8)
                        texture = self.calculate_glcm(band_norm)
                        feature.update({
                            f"band{i}_contrast": texture['contrast'],
                            f"band{i}_homogeneity": texture['homogeneity'],
                            f"band{i}_energy": texture['energy'],
                            f"band{i}_correlation": texture['correlation']
                        })
                    
                    # 4. Höhenmetriken (CHM)
                    if chm.size > 0:
                        feature.update({
                            "chm_max": np.nanmax(chm),
                            "chm_mean": np.nanmean(chm),
                            "chm_std": np.nanstd(chm),
                            "chm_skew": pd.Series(chm).skew(),
                            "chm_kurt": pd.Series(chm).kurtosis()
                        })
                    
                    # 5. Kombinationsmerkmale
                    if "ndvi_mean" in feature and "chm_mean" in feature:
                        feature["ndvi_chm"] = feature["ndvi_mean"] * feature["chm_mean"]
                        feature["evi_chm"] = feature.get("evi", 0) * feature["chm_mean"]
                    
                    # 6. Band-Verhältnisse
                    feature["rg_ratio"] = np.nanmean(red / (green + 1e-6))
                    feature["nir_blue_ratio"] = np.nanmean(nir / (blue + 1e-6))
                    
                    results.append(feature)
                
                except Exception as e:
                    print(f"⚠️ Fehler bei Polygon {row[id_column]}: {e}")
                    continue
        
        df_results = pd.DataFrame(results)
        
        if len(df_results) > 0:
            # Abgeleitete Features
            df_results['spectral_diversity'] = (df_results.get('band_1_std', 0) + 
                                                df_results.get('band_2_std', 0) +
                                                df_results.get('band_3_std', 0) + 
                                                df_results.get('band_4_std', 0))
            
            df_results['vegetation_strength'] = ((df_results.get('ndvi_mean', 0) + 
                                                  df_results.get('evi', 0) +
                                                  df_results.get('savi', 0)) / 3)
            
            df_results['texture_complexity'] = ((df_results.get('band1_contrast', 0) + 
                                                 df_results.get('band2_contrast', 0) +
                                                 df_results.get('band3_contrast', 0) + 
                                                 df_results.get('band4_contrast', 0)) / 4)
            
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            print(f"✅ {len(df_results)} {method_name} Features erfolgreich extrahiert")
        
        return df_results
    
    def predict_attributes_hybrid(self, features_df, gdf, id_column, method_name=""):
        """
        Hybrid-Attribut-Vorhersagen (Baumart/Zustand) + Art-spezifische Allometrie
        mit Online-Statistiken
        """
        print(f"🔮 Führe {method_name} Hybrid-Attribut-Vorhersagen durch...")
        
        predictions = []
        
        # Initialisiere Online-Statistiken pro Art und TP/FP
        #stats_by_species_tpfp = defaultdict(lambda: defaultdict(OnlineStatistics))
        #overall_stats = {
        #    'bhd_crown_only': OnlineStatistics(),
        #    'bhd_height_crown': OnlineStatistics(),
        #    'chm_max': OnlineStatistics(),
        #    'chm_mean': OnlineStatistics(),
        #    'polygon_area': OnlineStatistics()
        #}
        
        stats_by_species_tpfp = defaultdict(
            lambda: defaultdict(
                lambda: {
                    'bhd_crown_only': OnlineStatistics(),
                    'bhd_height_crown': OnlineStatistics(),
                    'chm_max': OnlineStatistics(),
                    'chm_mean': OnlineStatistics(),
                    'polygon_area': OnlineStatistics(),
                }
            )
        )

        overall_stats = {
            'bhd_crown_only': OnlineStatistics(),
            'bhd_height_crown': OnlineStatistics(),
            'chm_max': OnlineStatistics(),
            'chm_mean': OnlineStatistics(),
            'polygon_area': OnlineStatistics()
        }
        
        
        
        species_counts = defaultdict(int)
        tpfp_counts = defaultdict(int)
        
        for idx, row in tqdm(features_df.iterrows(), total=len(features_df), desc=f"{method_name} Predictions"):
            fid = row['fid']
            
            # Hole zugehöriges Polygon für TP/FP und Fläche
            polygon_row = gdf[gdf[id_column] == fid]
            if len(polygon_row) == 0:
                continue
            
            polygon_row = polygon_row.iloc[0]
            is_tp = polygon_row.get('is_TP', False) if hasattr(polygon_row, 'get') else polygon_row.get('match_type') == 'TP'
            tpfp_str = 'TP' if is_tp else 'FP'
            polygon_area = polygon_row.geometry.area
            
            pred = {'fid': fid}
            
            try:
                # 1. Baumart-Vorhersage
                features_needed = self.feature_lists.get('baumart', [])
                X_baumart = features_df.loc[idx:idx, features_needed].fillna(0)
                
                if len(X_baumart) > 0:
                    X_baumart_scaled = self.scalers['baumart'].transform(X_baumart)
                    baumart_class = self.models['baumart'].predict(X_baumart_scaled)[0]
                    baumart_pred = self.species_mapping.get(baumart_class, "Douglas Fir")
                else:
                    baumart_pred = "Douglas Fir"
                
                pred['baumart_pred'] = baumart_pred
                species_counts[baumart_pred] += 1
                tpfp_counts[tpfp_str] += 1
                
                # 2. Zustand-Vorhersage
                features_needed_zustand = self.feature_lists.get('zustand', [])
                X_zustand = features_df.loc[idx:idx, features_needed_zustand].fillna(0)
                
                if len(X_zustand) > 0:
                    X_zustand_scaled = self.scalers['zustand'].transform(X_zustand)
                    zustand_class = self.models['zustand'].predict(X_zustand_scaled)[0]
                    zustand_pred = self.condition_mapping.get(zustand_class, "healthy")
                else:
                    zustand_pred = "healthy"
                
                pred['zustand_pred'] = zustand_pred
                
                # 3. Höhen- und Kronendaten
                chm_max = float(row.get('chm_max', np.nan))
                chm_mean = float(row.get('chm_mean', np.nan))
                
                pred['chm_max'] = chm_max
                pred['chm_mean'] = chm_mean
                pred['polygon_area_m2'] = polygon_area
                pred['tpfp'] = tpfp_str
                
                # 4. BHD-Vorhersagen über Allometrie
                # 4a. Crown-only
                bhd_crown_only = self._calc_bhd_from_height_crown(
                    height=1.0,  # Dummy-Höhe
                    crown_area=polygon_area,
                    species=baumart_pred,
                    mode="crown_only"
                )
                pred['bhd_crown_only_cm'] = round(bhd_crown_only, 1)
                
                # 4b. Height + Crown
                if not np.isnan(chm_mean) and chm_mean > 0:
                    bhd_height_crown = self._calc_bhd_from_height_crown(
                        height=chm_mean,
                        crown_area=polygon_area,
                        species=baumart_pred,
                        mode="height_crown"
                    )
                else:
                    bhd_height_crown = bhd_crown_only  # Fallback zu crown-only
                
                pred['bhd_height_crown_cm'] = round(bhd_height_crown, 1)
                
                # 5. Online-Statistiken sammeln
                stats_by_species_tpfp[baumart_pred][tpfp_str]['bhd_crown_only'].add(bhd_crown_only)
                stats_by_species_tpfp[baumart_pred][tpfp_str]['bhd_height_crown'].add(bhd_height_crown)
                stats_by_species_tpfp[baumart_pred][tpfp_str]['chm_max'].add(chm_max)
                stats_by_species_tpfp[baumart_pred][tpfp_str]['chm_mean'].add(chm_mean)
                stats_by_species_tpfp[baumart_pred][tpfp_str]['polygon_area'].add(polygon_area)
                
                overall_stats['bhd_crown_only'].add(bhd_crown_only)
                overall_stats['bhd_height_crown'].add(bhd_height_crown)
                overall_stats['chm_max'].add(chm_max)
                overall_stats['chm_mean'].add(chm_mean)
                overall_stats['polygon_area'].add(polygon_area)
                
                predictions.append(pred)
            
            except Exception as e:
                print(f"⚠️ Fehler bei FID {fid}: {e}")
                continue
        
        df_pred = pd.DataFrame(predictions)
        print(f"✅ {len(df_pred)} Vorhersagen erstellt")
        
        return df_pred, dict(stats_by_species_tpfp), overall_stats, species_counts, tpfp_counts
    
    def classify_tp_fp(self, gdf, id_column, tp_csv, tp_id_column):
        """Klassifiziere Trees als TP (True Positive) oder FP (False Positive)"""
        print(f"📊 Klassifiziere TP/FP basierend auf: {os.path.basename(tp_csv)}")
        
        try:
            df_tp = pd.read_csv(tp_csv)
            tp_ids_set = set(df_tp[tp_id_column].unique())
            
            gdf['is_TP'] = gdf[id_column].isin(tp_ids_set)
            gdf['match_type'] = gdf['is_TP'].map({True: 'TP', False: 'FP'})
            
            tp_count = gdf['is_TP'].sum()
            fp_count = len(gdf) - tp_count
            print(f"✅ TP/FP klassifiziert: {tp_count} TP, {fp_count} FP")
        
        except Exception as e:
            print(f"⚠️ TP/FP Klassifikation fehlgeschlagen: {e}")
            gdf['is_TP'] = False
            gdf['match_type'] = 'FP'
        
        return gdf
    
    def integrate_with_geometries(self, gdf, predictions_df, id_column):
        """Verbinde Vorhersagen mit Geometrien"""
        #gdf_result = gdf[[id_column, 'geometry', 'match_type']].copy()
        #gdf_result = gdf_result.merge(
        #    predictions_df,
        #    left_on=id_column,
        #    right_on='fid',
        #    how='inner'
        #)
        gdf_result = gdf[[id_column, 'geometry', 'match_type']].copy()
        gdf_result[id_column] = gdf_result[id_column].astype(int)

        predictions_df = predictions_df.copy()
        predictions_df['fid'] = predictions_df['fid'].astype(int)

        gdf_result = gdf_result.merge(
            predictions_df,
            left_on=id_column,
            right_on='fid',
            how='inner'
        )

        # Optional: interne ID anders nennen, um Konflikt zu vermeiden
        gdf_result.rename(columns={'fid': 'fid_det'}, inplace=True)

        
        
        
        
        return gdf_result
    
    def print_statistics(self, stats_by_species_tpfp, overall_stats, species_counts, tpfp_counts, method_name, tile_name):
        """Drucke detaillierte Online-Statistiken"""
        print(f"\n{'='*70}")
        print(f"📊 {method_name.upper()} STATISTIKEN – {tile_name}")
        print(f"{'='*70}")
        
        # Globale Zähler
        total_trees = sum(tpfp_counts.values())
        tp_count = tpfp_counts.get('TP', 0)
        fp_count = tpfp_counts.get('FP', 0)
        tp_ratio = tp_count / total_trees if total_trees > 0 else 0
        
        print(f"\n🌳 GESAMT-STATISTIKEN")
        print(f"   Gesamt Bäume: {total_trees}")
        print(f"   TP: {tp_count}, FP: {fp_count}")
        print(f"   TP-Ratio: {tp_ratio:.2%}")
        
        # Statistiken pro Baumart
        print(f"\n🌲 PRO BAUMART:")
        for species in sorted(species_counts.keys()):
            count = species_counts[species]
            print(f"\n   {species} (n={count}):")
            
            for tpfp in ['TP', 'FP']:
                if tpfp not in stats_by_species_tpfp[species]:
                    continue
                
                stats = stats_by_species_tpfp[species][tpfp]
                bhd_crown = stats['bhd_crown_only'].to_dict()
                bhd_hc = stats['bhd_height_crown'].to_dict()
                chm = stats['chm_mean'].to_dict()
                
                print(f"      {tpfp} (n={bhd_crown['count']}):")
                print(f"         BHD (crown-only):     {bhd_crown['mean']:.1f} ± {bhd_crown['std']:.1f} cm")
                print(f"         BHD (height+crown):  {bhd_hc['mean']:.1f} ± {bhd_hc['std']:.1f} cm")
                print(f"         Höhe (CHM):          {chm['mean']:.1f} ± {chm['std']:.1f} m")
        
        # Gesamt-Statistiken
        print(f"\n📈 GLOBALE STATISTIKEN (alle Bäume):")
        for key in ['bhd_crown_only', 'bhd_height_crown', 'chm_mean', 'polygon_area']:
            stats_dict = overall_stats[key].to_dict()
            unit = "cm" if "bhd" in key else ("m" if "chm" in key else "m²")
            print(f"   {key:20s}: {stats_dict['mean']:8.2f} ± {stats_dict['std']:7.2f} {unit}")
    
    def save_statistics_json(self, stats_by_species_tpfp, overall_stats, method_name, output_dir, tile_name):
        """Speichere Statistiken als JSON"""
        stats_dict = {}
        
        # Pro Baumart
        for species, tpfp_dict in stats_by_species_tpfp.items():
            stats_dict[species] = {}
            for tpfp, stats in tpfp_dict.items():
                stats_dict[species][tpfp] = {
                    'bhd_crown_only': stats['bhd_crown_only'].to_dict(),
                    'bhd_height_crown': stats['bhd_height_crown'].to_dict(),
                    'chm_max': stats['chm_max'].to_dict(),
                    'chm_mean': stats['chm_mean'].to_dict(),
                    'polygon_area': stats['polygon_area'].to_dict()
                }
        
        # Gesamt
        overall_dict = {key: val.to_dict() for key, val in overall_stats.items()}
        
        output_file = os.path.join(
            output_dir,
            f"{tile_name}_{method_name.lower()}_statistics.json"
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                'by_species_tpfp': stats_dict,
                'overall': overall_dict
            }, f, indent=2)
        
        print(f"💾 Statistiken gespeichert: {output_file}")
    
    def process_method(self, method_config, tile_name, output_dir):
        """Verarbeite eine einzelne Detektionsmethode komplett"""
        method_name = method_config['method']
        
        # 1. Lade Polygone
        gdf, id_column = self.load_polygons_robust(
            method_config['polygon_file'],
            method_name=method_name
        )
        if gdf is None:
            print(f"❌ {method_name} abgebrochen")
            return None, None
        
        # 2. Klassifiziere TP/FP
        gdf = self.classify_tp_fp(
            gdf, id_column,
            method_config['tp_csv'],
            method_config['tp_id_column']
        )
        
        # 3. Extrahiere Features
        features_df = self.extract_features_from_polygons(
            gdf, id_column,
            method_config['rgbnir_tif'],
            method_config['chm_tif'],
            method_name=method_name
        )
        
        if len(features_df) == 0:
            print(f"❌ {method_name} Keine Features extrahiert")
            return None, None
        
        # Speichere Features als CSV
        features_csv = os.path.join(output_dir, f"{tile_name}_{method_name.lower()}_features.csv")
        os.makedirs(output_dir, exist_ok=True)
        features_df.to_csv(features_csv, index=False)
        print(f"💾 Features gespeichert: {features_csv}")
        
        # 4. Vorhersagen + Online-Statistiken
        predictions_df, stats_by_sp, overall_st, sp_counts, tpfp_counts = self.predict_attributes_hybrid(
            features_df, gdf, id_column, method_name=method_name
        )
        
        # 5. Integriere mit Geometrien
        gdf_result = self.integrate_with_geometries(gdf, predictions_df, id_column)
        
        # 6. Drucke Statistiken
        self.print_statistics(stats_by_sp, overall_st, sp_counts, tpfp_counts, method_name, tile_name)
        
        # 7. Speichere Ergebnisse
        try:
            # CSV-Export
            output_csv = os.path.join(output_dir, f"{tile_name}_{method_name.lower()}_results.csv")
            gdf_result.drop('geometry', axis=1).to_csv(output_csv, index=False)
            print(f"💾 CSV gespeichert: {output_csv}")
            
            # GeoPackage-Export
            try:
                output_gpkg = os.path.join(output_dir, f"{tile_name}_{method_name.lower()}_results.gpkg")
                gdf_result.to_file(output_gpkg, driver="GPKG")
                print(f"💾 GeoPackage gespeichert: {output_gpkg}")
            except Exception as e:
                print(f"⚠️ GeoPackage-Export fehlgeschlagen: {e}")
        
        except Exception as e:
            print(f"❌ Fehler beim Speichern: {e}")
        
        # 8. Statistiken als JSON
        self.save_statistics_json(stats_by_sp, overall_st, method_name, output_dir, tile_name)
        
        return gdf_result, {
            'total_trees': len(gdf_result),
            'tp_count': tpfp_counts.get('TP', 0),
            'fp_count': tpfp_counts.get('FP', 0),
            'tp_ratio': tpfp_counts.get('TP', 0) / len(gdf_result) if len(gdf_result) > 0 else 0,
            'species_counts': dict(sp_counts),
            'method': method_name
        }


def main():
    """Main-Funktion mit Multi-Method Konfigurationen"""
    
    
    # Tile-Konfigurationen
    TILES = {
        "tile1": {
            "TCD": {
                "method": "TCD",
                "polygon_file": "/home/abrax/Desktop/Infer_stat_input/tcd/raw/result_tile1.sqlite",
                "rgbnir_tif": "/home/abrax/Desktop/tiles_NIR_norm/normalized_tile_1.tif",
                "chm_tif": "/home/abrax/Desktop/Tif_CHM/processed/Tif_CHM_1_nonneg_compressed.tif",
                "tp_csv": "/home/abrax/Desktop/Infer_stat_input/tcd/pipe1_TP/pipe1_TP_tile1.csv",
                "tp_id_column": "tcd_id",
                "output_dir": "/home/abrax/Desktop/Bitz_singular/output/multimethod/tile1/tcd"
            },
            "Detectree2": {
                "method": "Detectree2",
                "polygon_file": "/home/abrax/Desktop/Infer_stat_input/dt2/raw/result_tile1.gpkg",
                "rgbnir_tif": "/home/abrax/Desktop/tiles_NIR_norm/normalized_tile_1.tif",
                "chm_tif": "/home/abrax/Desktop/Tif_CHM/processed/Tif_CHM_1_nonneg_compressed.tif",
                "tp_csv": "/home/abrax/Desktop/Infer_stat_input/dt2/pipe1_TP/pipe1_TP_detected_trees_enhanced1.csv",
                "tp_id_column": "det_id",
                "output_dir": "/home/abrax/Desktop/Bitz_singular/output/multimethod/tile1/detectree2"
            },
            "DeepTree": {
                "method": "DeepTree",
                "polygon_file": "/home/abrax/Desktop/Infer_stat_input/deeptree/raw/result_tile1.gpkg",
                "rgbnir_tif": "/home/abrax/Desktop/tiles_NIR_norm/normalized_tile_1.tif",
                "chm_tif": "/home/abrax/Desktop/Tif_CHM/processed/Tif_CHM_1_nonneg_compressed.tif",
                "tp_csv": "/home/abrax/Desktop/Infer_stat_input/deeptree/pipe1_TP/pipe1_TP_DeepTree_train_tile1.csv",
                "tp_id_column": "det_id",  # ggf. anpassen (z.B. deeptree_id)
                "output_dir": "/home/abrax/Desktop/Bitz_singular/output/multimethod/tile1/deeptree"
            }
        },
        "tile2": {
            "TCD": {
                "method": "TCD",
                "polygon_file": "/home/abrax/Desktop/Infer_stat_input/tcd/raw/result_tile2.sqlite",
                "rgbnir_tif": "/home/abrax/Desktop/tiles_NIR_norm/normalized_tile_2.tif",
                "chm_tif": "/home/abrax/Desktop/Tif_CHM/processed/Tif_CHM_2_nonneg_compressed.tif",
                "tp_csv": "/home/abrax/Desktop/Infer_stat_input/tcd/pipe1_TP/pipe1_TP_tile2.csv",
                "tp_id_column": "tcd_id",
                "output_dir": "/home/abrax/Desktop/Bitz_singular/output/multimethod/tile2/tcd"
            },
            "Detectree2": {
                "method": "Detectree2",
                "polygon_file": "/home/abrax/Desktop/Infer_stat_input/dt2/raw/result_tile2.gpkg",
                "rgbnir_tif": "/home/abrax/Desktop/tiles_NIR_norm/normalized_tile_2.tif",
                "chm_tif": "/home/abrax/Desktop/Tif_CHM/processed/Tif_CHM_2_nonneg_compressed.tif",
                "tp_csv": "/home/abrax/Desktop/Infer_stat_input/dt2/pipe1_TP/pipe1_TP_detected_trees_enhanced2.csv",
                "tp_id_column": "det_id",
                "output_dir": "/home/abrax/Desktop/Bitz_singular/output/multimethod/tile2/detectree2"
            },
            "DeepTree": {
                "method": "DeepTree",
                "polygon_file": "/home/abrax/Desktop/Infer_stat_input/deeptree/raw/result_tile2.gpkg",
                "rgbnir_tif": "/home/abrax/Desktop/tiles_NIR_norm/normalized_tile_2.tif",
                "chm_tif": "/home/abrax/Desktop/Tif_CHM/processed/Tif_CHM_2_nonneg_compressed.tif",
                "tp_csv": "/home/abrax/Desktop/Infer_stat_input/deeptree/pipe1_TP/pipe1_TP_DeepTree_train_tile2.csv",
                "tp_id_column": "det_id",  # ggf. anpassen
                "output_dir": "/home/abrax/Desktop/Bitz_singular/output/multimethod/tile2/deeptree"
            }
        },
        "tile3": {
            "TCD": {
                "method": "TCD",
                "polygon_file": "/home/abrax/Desktop/Infer_stat_input/tcd/raw/result_tile3.sqlite",
                "rgbnir_tif": "/home/abrax/Desktop/tiles_NIR_norm/normalized_tile_3.tif",
                "chm_tif": "/home/abrax/Desktop/Tif_CHM/processed/Tif_CHM_3_nonneg_compressed.tif",
                "tp_csv": "/home/abrax/Desktop/Infer_stat_input/tcd/pipe1_TP/pipe1_TP_tile3.csv",
                "tp_id_column": "tcd_id",
                "output_dir": "/home/abrax/Desktop/Bitz_singular/output/multimethod/tile3/tcd"
            },
            "Detectree2": {
                "method": "Detectree2",
                "polygon_file": "/home/abrax/Desktop/Infer_stat_input/dt2/raw/result_tile3.gpkg",
                "rgbnir_tif": "/home/abrax/Desktop/tiles_NIR_norm/normalized_tile_3.tif",
                "chm_tif": "/home/abrax/Desktop/Tif_CHM/processed/Tif_CHM_3_nonneg_compressed.tif",
                "tp_csv": "/home/abrax/Desktop/Infer_stat_input/dt2/pipe1_TP/pipe1_TP_detected_trees_enhanced3.csv",
                "tp_id_column": "det_id",
                "output_dir": "/home/abrax/Desktop/Bitz_singular/output/multimethod/tile3/detectree2"
            },
            "DeepTree": {
                "method": "DeepTree",
                "polygon_file": "/home/abrax/Desktop/Infer_stat_input/deeptree/raw/result_tile3.gpkg",
                "rgbnir_tif": "/home/abrax/Desktop/tiles_NIR_norm/normalized_tile_3.tif",
                "chm_tif": "/home/abrax/Desktop/Tif_CHM/processed/Tif_CHM_3_nonneg_compressed.tif",
                "tp_csv": "/home/abrax/Desktop/Infer_stat_input/deeptree/pipe1_TP/pipe1_TP_DeepTree_train_tile3.csv",
                "tp_id_column": "det_id",  # ggf. anpassen
                "output_dir": "/home/abrax/Desktop/Bitz_singular/output/multimethod/tile3/deeptree"
            }
        }
    }

    
    
    # Tile-Konfigurationen
    #TILES = {
    #    "tile1": {
    #        "TCD": {
    #            "method": "TCD",
    #            "polygon_file": "/home/abrax/Desktop/Infer_stat_input/tcd/raw/result_tile1.sqlite",
    #            "rgbnir_tif": "/home/abrax/Desktop/tiles_NIR_norm/normalized_tile_1.tif",
    #            "chm_tif": "/home/abrax/Desktop/Tif_CHM/processed/Tif_CHM_1_nonneg_compressed.tif",
    #            "tp_csv": "/home/abrax/Desktop/Infer_stat_input/tcd/pipe1_TP/pipe1_TP_tile1.csv",
    #            "tp_id_column": "tcd_id",
    #            "output_dir": "/home/abrax/Desktop/Bitz_singular/output/multimethod/tile1/tcd"
    #        },
    #        "Detectree2": {
    #            "method": "Detectree2",
    #            "polygon_file": "/home/abrax/Desktop/Infer_stat_input/dt2/raw/result_tile1.gpkg",
    #            "rgbnir_tif": "/home/abrax/Desktop/tiles_NIR_norm/normalized_tile_1.tif",
    #            "chm_tif": "/home/abrax/Desktop/Tif_CHM/processed/Tif_CHM_1_nonneg_compressed.tif",
    #            "tp_csv": "/home/abrax/Desktop/Infer_stat_input/dt2/pipe1_TP/pipe1_TP_detected_trees_enhanced1.csv",
    #            "tp_id_column": "det_id",
    #            "output_dir": "/home/abrax/Desktop/Bitz_singular/output/multimethod/tile1/detectree2"
    #        }
    #    }
    #}
    
    integrator = MultiMethodIntegrator()
    
    for tile_name, methods in TILES.items():
        print(f"\n\n{'='*70}")
        print(f"🎯 VERARBEITE {tile_name.upper()}")
        print(f"{'='*70}\n")
        
        all_results = {}
        
        for method_name, method_config in methods.items():
            gdf_result, stats = integrator.process_method(
                method_config, tile_name,
                method_config['output_dir']
            )
            
            if stats:
                all_results[method_name] = stats
        
        # Vergleich aller Methoden pro Tile
        if len(all_results) > 1:
            print(f"\n{'='*70}")
            print(f"📊 METHODEN-VERGLEICH für {tile_name.upper()}")
            print(f"{'='*70}")
            
            comparison_df = pd.DataFrame(all_results).T
            print(comparison_df)
            
            comparison_csv = f"/home/abrax/Desktop/Bitz_singular/output/multimethod/{tile_name}_method_comparison.csv"
            os.makedirs(os.path.dirname(comparison_csv), exist_ok=True)
            comparison_df.to_csv(comparison_csv)
            print(f"💾 Vergleichstabelle gespeichert: {comparison_csv}")


if __name__ == "__main__":
    main()
