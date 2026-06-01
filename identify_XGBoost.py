"""
XGBoost Model for Solar Cell Parameter Identification
Identifies physical parameters (Iph, I0, Rs, Rsh, nNsVth) from IV curves
"""
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# Configuration constants
CONFIG = {
    'DIR_PARAMETROS': './base_train/',
    'DIR_CURVAS': './dados_cocoa/',
    'N_POINTS': 181,
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': 554,
    'CUDA_DEVICE': 'cuda',
    'SEED': 42,
}

BEST_PARAMS = {
    'Iph': {
        'subsample': 1.0, 'reg_lambda': 1, 'reg_alpha': 0, 'n_estimators': 3000,
        'min_child_weight': 1, 'max_depth': 5, 'learning_rate': 0.03, 'colsample_bytree': 0.8
    },
    'I0': {
        'subsample': 0.6, 'reg_lambda': 5, 'reg_alpha': 1, 'n_estimators': 1500,
        'min_child_weight': 3, 'max_depth': 12, 'learning_rate': 0.01, 'colsample_bytree': 0.6
    },
    'Rs': {
        'subsample': 0.6, 'reg_lambda': 5, 'reg_alpha': 1, 'n_estimators': 1500,
        'min_child_weight': 3, 'max_depth': 12, 'learning_rate': 0.01, 'colsample_bytree': 0.6
    },
    'Rsh': {
        'subsample': 0.6, 'reg_lambda': 10, 'reg_alpha': 0.1, 'n_estimators': 400,
        'min_child_weight': 3, 'max_depth': 9, 'learning_rate': 0.03, 'colsample_bytree': 1.0
    },
    'nNsVth': {
        'subsample': 0.6, 'reg_lambda': 1, 'reg_alpha': 0, 'n_estimators': 3000,
        'min_child_weight': 3, 'max_depth': 7, 'learning_rate': 0.03, 'colsample_bytree': 0.8
    }
}

TARGET_VARIABLES = ['Iph', 'I0', 'Rs', 'Rsh', 'nNsVth']


class SolarCellModel:
    """XGBoost model for identifying solar cell parameters from IV curves."""
    
    def __init__(self, config: Dict = None):
        self.config = {**CONFIG, **(config or {})}
        self.models = {}
        self.scaler = None
        
    @staticmethod
    def _create_column_names(n_points: int, prefix: str) -> List[str]:
        """Create column names for curve data."""
        return [f'{prefix}_{k}' for k in range(n_points)]
    
    def load_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load parameter and curve data from disk."""
        dir_params = Path(self.config['DIR_PARAMETROS'])
        dir_curves = Path(self.config['DIR_CURVAS'])
        
        # Load parameter files
        param_files = sorted([f for f in dir_params.glob('*.parquet')])
        df_params = pd.concat(
            [pd.read_parquet(f) for f in param_files],
            ignore_index=True
        )
        
        # Extract cell types and load curve data
        types = [f.stem.split('_')[1] for f in param_files]
        curves = pd.concat(
            [pd.read_feather(dir_curves / f'Cocoa_{t}.feather') for t in types],
            ignore_index=True
        )
        
        return pd.concat([df_params, curves], axis=1), types
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from IV curve data."""
        n_points = self.config['N_POINTS']
        cols_current = self._create_column_names(n_points, 'current')
        cols_voltage = self._create_column_names(n_points, 'voltage')
        
        df = df.copy()
        
        # Basic electrical features
        df['Voc'] = df['voltage_180']
        df['Isc'] = df['current_0']
        
        # Maximum power point
        power = df[cols_current].values * df[cols_voltage].values
        mpp_idx = np.argmax(power, axis=1)
        df['Vmp'] = df[cols_voltage].values[np.arange(len(df)), mpp_idx]
        df['Imp'] = df[cols_current].values[np.arange(len(df)), mpp_idx]
        
        # Fill factor
        df['FF'] = (df['Vmp'] * df['Imp']) / (df['Voc'] * df['Isc'] + 1e-8)
        
        # VOC region slope
        v_voc = df[[f'voltage_{i}' for i in range(150, 163)]].values
        i_voc = df[[f'current_{i}' for i in range(150, 163)]].values
        slopes_voc = np.array([np.polyfit(v_voc[i], i_voc[i], 1)[0] for i in range(len(df))])
        
        df['slope_voc'] = slopes_voc
        df['inv_slope_voc'] = np.abs(1 / (slopes_voc + 1e-8))
        df['I_near_0'] = df[[f'current_{i}' for i in range(15)]].mean(axis=1)
        
        # ISC region slopes
        df['slope_isc'] = (df['current_5'] - df['current_0']) / (
            df['voltage_5'] - df['voltage_0'] + 1e-8
        )
        df['slope_isc_region'] = (df['current_25'] - df['current_0']) / (
            df['voltage_25'] - df['voltage_0'] + 1e-8
        )
        df['inv_slope_isc'] = np.abs(1 / (df['slope_isc_region'] + 1e-8))
        
        return df.fillna(0)
    
    def create_analytical_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create analytical proxies for parameters."""
        df = df.copy()
        
        # nNsVth proxy
        denominator = np.log(np.clip(
            df['Isc'] / (df['Isc'] - df['Imp'] + 1e-8), 1e-5, None
        ))
        df['nNsVth_proxy'] = np.clip(
            (df['Voc'] - df['Vmp']) / (denominator + 1e-8), 0.1, 50.0
        )
        
        # I0 proxy
        exponent = df['Voc'] / (df['nNsVth_proxy'] * 2.302585)
        df['log_I0_proxy'] = np.clip(
            np.log10(np.clip(df['Isc'], 1e-8, None)) - exponent, -20.0, -2.0
        )
        
        # Other proxies
        df['Iph_proxy'] = df['Isc'] * 1.001
        df['log_Rsh_proxy'] = np.log10(np.clip(df['inv_slope_isc'], 1.0, 100000.0))
        df['log_Rs_proxy'] = np.log10(np.clip(df['inv_slope_voc'], 0.0001, 100.0))
        
        # FF loss
        voc_norm = df['Voc'] / df['nNsVth_proxy']
        ff_ideal = (voc_norm - np.log(np.clip(voc_norm + 0.72, 1e-5, None))) / (voc_norm + 1.0)
        df['FF_loss'] = ff_ideal - df['FF']
        
        return df
    
    @staticmethod
    def select_features() -> List[str]:
        """Return final feature set."""
        return [
            'Voc', 'Isc', 'Vmp', 'Imp', 'FF', 'FF_loss',
            'slope_voc', 'inv_slope_voc', 'I_near_0',
            'slope_isc', 'slope_isc_region', 'inv_slope_isc',
            'nNsVth_proxy', 'log_I0_proxy', 'Iph_proxy',
            'log_Rsh_proxy', 'log_Rs_proxy'
        ]
    
    @staticmethod
    def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare target variables with log transformation."""
        df = df.copy()
        df['I0'] = np.log10(np.clip(df['I0'], 1e-15, None))
        df['Rs'] = np.log10(np.clip(df['Rs'], 1e-4, None))
        df['Rsh'] = np.log10(np.clip(df['Rsh'], 1.0, None))
        df['nNsVth'] = np.log10(np.clip(df['nNsVth'], 1e-8, None))
        df['Iph'] = np.log10(np.clip(df['Iph'], 1e-8, None))
        return df
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Train XGBoost models for each target variable."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['TEST_SIZE'],
            random_state=self.config['RANDOM_STATE']
        )
        
        for target in TARGET_VARIABLES:
            print(f"Training model for {target}...")
            
            model = XGBRegressor(
                random_state=self.config['SEED'],
                device=self.config['CUDA_DEVICE'],
                **BEST_PARAMS[target]
            )
            
            model.fit(
                X_train, y_train[target],
                eval_set=[(X_test, y_test[target])],
                verbose=False
            )
            
            self.models[target] = model
            model.save_model(f'model_{target}.json')
    
    def run(self) -> None:
        """Execute full pipeline."""
        print("Loading data...")
        df, types = self.load_data()
        
        print("Engineering features...")
        df = self.engineer_features(df)
        df = self.create_analytical_proxies(df)
        
        # Prepare features and targets
        features = self.select_features()
        X = df[features].copy()
        
        df = self.prepare_targets(df)
        y = df[TARGET_VARIABLES]
        
        print("Training models...")
        self.train(X, y)
        
        print("Pipeline completed successfully!")



if __name__ == "__main__":
    model = SolarCellModel()
    model.run()