#!/usr/bin/env python3
"""
Bereinigte Feature-Extraktion für RGB-NIR + CHM
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import mapping
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops

def calculate_glcm(texture_data):
    """Berechnet GLCM-Texturmetriken für 8-bit Texturdaten"""
    glcm = graycomatrix(texture_data.reshape(-1,1), [1], [0], symmetric=True, normed=True)
    return {
        'contrast': graycoprops(glcm, 'contrast')[0,0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0,0],
        'energy': graycoprops(glcm, 'energy')[0,0],
        'correlation': graycoprops(glcm, 'correlation')[0,0]
    }

def extract_features_per_file(shapefile, rgbnir_tif, chm_tif):
    gdf = gpd.read_file(shapefile)
    results = []
    
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc=f"Verarbeite {os.path.basename(shapefile)}"):
        try:
            geom = [mapping(row["geometry"])]
            with rasterio.open(rgbnir_tif) as src_rgbnir:
                rgbnir_data, _ = rasterio.mask.mask(src_rgbnir, geom, crop=True)
            with rasterio.open(chm_tif) as src_chm:
                chm_data, _ = rasterio.mask.mask(src_chm, geom, crop=True)

            # Überprüfung
            if rgbnir_data.shape[0] < 4:
                raise ValueError("RGBNIR hat weniger als 4 Bänder")

            mask = (rgbnir_data != src_rgbnir.nodata).all(axis=0)
            chm_mask = (chm_data != src_chm.nodata)[0] if src_chm.nodata is not None else np.ones_like(chm_data[0], dtype=bool)

            red = np.clip(rgbnir_data[0][mask], 0, 42000)
            green = np.clip(rgbnir_data[1][mask], 0, 42000)
            blue = np.clip(rgbnir_data[2][mask], 0, 42000)
            nir = np.clip(rgbnir_data[3][mask], 0, 42000)
            chm = chm_data[0][chm_mask]

            if red.size == 0 or chm.size == 0:
                raise ValueError("Leere Daten nach Maskierung")

            feature = {'fid': row.get('fid', None)}

            for i, band_data in enumerate([red, green, blue, nir], start=1):
                feature.update({
                    f"band_{i}_mean": np.nanmean(band_data),
                    f"band_{i}_std": np.nanstd(band_data),
                    f"band_{i}_q25": np.nanquantile(band_data, 0.25),
                    f"band_{i}_q75": np.nanquantile(band_data, 0.75),
                    f"band_{i}_skew": pd.Series(band_data).skew(),
                    f"band_{i}_kurt": pd.Series(band_data).kurtosis()
                })

            ndvi = (nir - red) / (nir + red + 1e-6)
            gndvi = (nir - green) / (nir + green + 1e-6)
            L = 0.5
            savi = ((nir - red) / (nir + red + L + 1e-6)) * (1 + L)
            vari = (green - red) / (green + red - blue + 1e-6)
            denominator_evi = nir + 6*red - 7.5*blue + 1
            evi = np.where(np.abs(denominator_evi) < 1e-9, 0, 2.5 * (nir - red) / denominator_evi)
            msi = nir / (red + 1e-6)
            ndwi = (green - nir) / (green + nir + 1e-6)

            feature.update({
                "ndvi_mean": np.nanmean(ndvi),
                "ndvi_std": np.nanstd(ndvi),
                "gndvi": np.nanmean(gndvi),
                "savi": np.nanmean(savi),
                "vari": np.nanmean(vari),
                "evi": np.nanmean(evi),
                "msi": np.nanmean(msi),
                "ndwi": np.nanmean(ndwi)
            })

            for i, band in enumerate([red, green, blue, nir], 1):
                if band.size < 10:
                    continue
                norm_band = ((band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band) + 1e-6) * 255).astype(np.uint8)
                texture = calculate_glcm(norm_band)
                feature.update({
                    f"band{i}_contrast": texture['contrast'],
                    f"band{i}_homogeneity": texture['homogeneity'],
                    f"band{i}_energy": texture['energy'],
                    f"band{i}_correlation": texture['correlation']
                })

            feature.update({
                "chm_max": np.nanmax(chm),
                "chm_mean": np.nanmean(chm),
                "chm_std": np.nanstd(chm),
                "chm_skew": pd.Series(chm).skew(),
                "chm_kurt": pd.Series(chm).kurtosis(),
                "ndvi_chm": np.nanmean(ndvi) * np.nanmean(chm),
                "evi_chm": np.nanmean(evi) * np.nanmean(chm),
                "rg_ratio": np.nanmean(red / (green + 1e-6)),
                "nir_blue_ratio": np.nanmean(nir / (blue + 1e-6))
            })

            results.append(feature)

        except Exception as e:
            print(f"❌ Fehler bei Polygon ?: {str(e)}")
            continue

    return pd.DataFrame(results)

def extract_full_features(shapefile_paths, rgbnir_paths, chm_paths, output_csv):
    all_features = []
    for shp, rgbnir, chm in zip(shapefile_paths, rgbnir_paths, chm_paths):
        df = extract_features_per_file(shp, rgbnir, chm)
        all_features.append(df)
    pd.concat(all_features).replace([np.inf, -np.inf], np.nan).to_csv(output_csv, index=False)
    print(f"\n✅ Features gespeichert: {output_csv}")

if __name__ == "__main__":
    shapefile_paths = [
        "/home/abrax/Desktop/output_pipe1/martel.sqlite",
        "/home/abrax/Desktop/output_pipe1/track1.sqlite",
        "/home/abrax/Desktop/output_pipe1/track2.sqlite",
        "/home/abrax/Desktop/output_pipe1/x7.sqlite"
    ]
    rgbnir_tifs = [
        "/home/abrax/Desktop/FInal_sample/Martel_RGB_NIR.tif",
        "/home/abrax/Desktop/FInal_sample/trakt1_RGB_NIR.tif",
        "/home/abrax/Desktop/FInal_sample/trakt2_RGB_NIR.tif",
        "/home/abrax/Desktop/FInal_sample/x7_RGB_NIR.tif"
    ]
    chm_tifs = [
        "/home/abrax/Desktop/CHM_samples/CHM_Martel.tif",
        "/home/abrax/Desktop/CHM_samples/CHM_Trakt_1.tif",
        "/home/abrax/Desktop/CHM_samples/CHM_Trakt_2.tif",
        "/home/abrax/Desktop/CHM_samples/CHM_x7.tif"
    ]
    output_csv = "/home/abrax/Training/results/full_features_per_polygon_Samples.csv"

    extract_full_features(shapefile_paths, rgbnir_tifs, chm_tifs, output_csv)

