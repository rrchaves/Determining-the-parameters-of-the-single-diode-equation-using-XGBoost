"""
Feature Engineering Script for Solar Panel Parameters Prediction

This script implements the physical feature engineering for solar cell diode model parameters.
It processes I-V curve data and extracts relevant features for XGBoost model training.
"""

import pandas as pd
import numpy as np
import os


def create_features_from_curves(df_end, df_normalized=None):
    """
    Create physical features from I-V curve data for solar panels.
    
    Parameters:
    -----------
    df_end : pd.DataFrame
        DataFrame containing current and voltage measurements
    df_normalized : pd.DataFrame, optional
        If None, creates a copy from df_end
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features
    """
    
    if df_normalized is None:
        df_normalized = df_end.copy()
    
    # =====================================================================
    # 1. PHYSICAL FEATURES ENGINEERING
    # =====================================================================
    
    target = ['Iph', 'I0', 'Rs', 'Rsh', 'nNsVth']
    
    colums_current = [f'current_{i}' for i in range(181)]
    colums_voltage = [f'voltage_{i}' for i in range(181)]
    
    # Extract key points of the I-V curve
    df_normalized['Voc'] = df_normalized['voltage_180']
    df_normalized['Isc'] = df_normalized['current_0']
    
    # Find maximum power point (Vmp and Imp)
    power = df_normalized[colums_current].values * df_normalized[colums_voltage].values
    orf = np.apply_along_axis(np.argmax, axis=1, arr=power)
    df_normalized['Vmp'] = [df_normalized[colums_voltage].iloc[i, j] for i, j in enumerate(orf)]
    df_normalized['Imp'] = [df_normalized[colums_current].iloc[i, j] for i, j in enumerate(orf)]
    
    # Curve geometry - Fill Factor
    df_normalized['FF'] = (df_normalized['Vmp'] * df_normalized['Imp']) / (
        df_normalized['Voc'] * df_normalized['Isc'] + 1e-8
    )
    
    # =====================================================================
    # 2. SLOPE CALCULATIONS (Vectorized using polyfit)
    # =====================================================================
    
    # Slope at Voc region (near open circuit voltage)
    I_final = df_normalized[[f'current_{i}' for i in range(150, 163)]].values
    V_final = df_normalized[[f'voltage_{i}' for i in range(150, 163)]].values
    coef = [np.polyfit(V_final[i], I_final[i], 1)[0] for i in range(I_final.shape[0])]
    
    df_normalized['slope_voc'] = np.array(coef)
    df_normalized['inv_slope_voc'] = np.abs(1 / (df_normalized['slope_voc'] + 1e-8))
    
    # Slope at Isc region (near short circuit current)
    df_normalized['I_near_0'] = df_normalized[[f'current_{i}' for i in range(0, 15)]].mean(axis=1)
    df_normalized['slope_isc'] = (
        (df_normalized['current_5'] - df_normalized['current_0']) / 
        (df_normalized['voltage_5'] - df_normalized['voltage_0'] + 1e-8)
    )
    df_normalized['slope_isc_region'] = (
        (df_normalized['current_25'] - df_normalized['current_0']) / 
        (df_normalized['voltage_25'] - df_normalized['voltage_0'] + 1e-8)
    )
    df_normalized['inv_slope_isc'] = np.abs(1 / (df_normalized['slope_isc_region'] + 1e-8))
    
    # =====================================================================
    # 3. ANALYTICAL PROXIES (Physics-based)
    # =====================================================================
    
    # nNsVth proxy from single-diode model equation
    denominador_ln = np.log(
        np.clip(
            df_normalized['Isc'] / (df_normalized['Isc'] - df_normalized['Imp'] + 1e-8),
            1e-5, None
        )
    )
    df_normalized['nNsVth_proxy'] = np.clip(
        (df_normalized['Voc'] - df_normalized['Vmp']) / (denominador_ln + 1e-8),
        0.1, 50.0
    )
    
    # I0 proxy calculation
    expoente_proxy = df_normalized['Voc'] / (df_normalized['nNsVth_proxy'] * 2.302585)  # 2.302585 is ln(10)
    df_normalized['log_I0_proxy'] = np.clip(
        np.log10(np.clip(df_normalized['Isc'], 1e-8, None)) - expoente_proxy,
        -20.0, -2.0
    )
    
    # Other proxy features
    df_normalized['Iph_proxy'] = df_normalized['Isc'] * 1.001
    df_normalized['log_Rsh_proxy'] = np.log10(
        np.clip(df_normalized['inv_slope_isc'], 1.0, 100000.0)
    )
    df_normalized['log_Rs_proxy'] = np.log10(
        np.clip(df_normalized['inv_slope_voc'], 0.0001, 100.0)
    )
    
    # Fill Factor loss relative to ideal
    voc_norm = df_normalized['Voc'] / df_normalized['nNsVth_proxy']
    FF_ideal = (voc_norm - np.log(np.clip(voc_norm + 0.72, 1e-5, None))) / (voc_norm + 1.0)
    df_normalized['FF_loss'] = FF_ideal - df_normalized['FF']
    
    # Protection against NaN in features
    df_normalized.fillna(0, inplace=True)
    
    return df_normalized, colums_current, colums_voltage


def normalize_targets(df, target_list=['Iph', 'I0', 'Rs', 'Rsh', 'nNsVth']):
    """
    Apply logarithmic transformation and StandardScaler normalization to target variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing target columns
    target_list : list
        List of target column names
    
    Returns:
    --------
    tuple
        Normalized DataFrame and dictionary of scalers
    """
    
    df_norm = df.copy()
    
    # Apply log transformation (these variables have very small scales)
    df_norm['I0'] = np.log10(np.clip(df_norm['I0'], 1e-15, None))
    df_norm['Rs'] = np.log10(np.clip(df_norm['Rs'], 1e-4, None))
    df_norm['Rsh'] = np.log10(np.clip(df_norm['Rsh'], 1.0, None))
    df_norm['nNsVth'] = np.log10(df_norm['nNsVth'])
    df_norm['Iph'] = np.log10(df_norm['Iph'])
    
    
    return df_norm


def select_best_features():
    """
    Return the list of best features for XGBoost model.
    
    Returns:
    --------
    list
        List of feature names selected for model training
    """
    
    best_features = [
        'Voc', 'Isc', 'Vmp', 'Imp', 'FF', 'FF_loss', 
        'slope_voc', 'inv_slope_voc', 'I_near_0', 
        'slope_isc', 'slope_isc_region', 'inv_slope_isc',
        'nNsVth_proxy', 'log_I0_proxy', 'Iph_proxy', 
        'log_Rsh_proxy', 'log_Rs_proxy'
    ]
    
    return best_features


def get_curve_columns(n_points=181):
    """
    Generate column names for current and voltage measurements.
    
    Parameters:
    -----------
    n_points : int
        Number of measurement points along the I-V curve
    
    Returns:
    --------
    tuple
        Lists of current and voltage column names
    """
    
    colums_current = [f'current_{i}' for i in range(n_points)]
    colums_voltage = [f'voltage_{i}' for i in range(n_points)]
    
    return colums_current, colums_voltage


# =========================================================================
# EXAMPLE USAGE
# =========================================================================

if __name__ == "__main__":
    
    # Load your data (example)
    # df_end = pd.read_parquet('your_data.parquet')
    
    # Create features
    # df_normalized, cols_current, cols_voltage = create_features_from_curves(df_end)
    
    # Normalize targets
    # df_normalized = normalize_targets(df_normalized)
    
    # Get best features for modeling
    # features = select_best_features()
    # X = df_normalized[features].copy()
    
    print("Feature engineering module loaded successfully!")
