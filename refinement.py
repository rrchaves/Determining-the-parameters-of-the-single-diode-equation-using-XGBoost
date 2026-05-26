from dataclasses import dataclass
from scipy.optimize import least_squares
from typing import Tuple
import numpy as np
from scipy.special import lambertw


@dataclass
class DiodeParameters:
    """Diode model parameters for single-diode model."""
    photo_current: float  # Iph - Light-generated current
    reverse_saturation_current: float  # I0 - Diode saturation current
    series_resistance: float  # Rs - Series resistance
    shunt_resistance: float  # Rsh - Shunt resistance
    thermal_voltage_product: float  # nNsVth - Product of thermal voltage, ideality factor, and cell count
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to numpy array for optimization."""
        return np.array([
            self.photo_current,
            self.reverse_saturation_current,
            self.series_resistance,
            self.shunt_resistance,
            self.thermal_voltage_product
        ])
    
    @classmethod
    def from_array(cls, params: np.ndarray) -> "DiodeParameters":
        """Create DiodeParameters from numpy array."""
        return cls(*params)
    
    @classmethod
    def from_xgboost(cls, xgb_predictions: np.ndarray) -> "DiodeParameters":
        """Create parameters from XGBoost predictions with safety clipping."""
        iph, i0, rs, rsh, nvth = xgb_predictions
        
        # Safety clipping to ensure physically valid parameters
        iph = np.clip(iph, 0.01, 100.0)
        i0 = np.clip(i0, 1e-15, 1e-2)
        rs = np.clip(rs, 0.0001, 100.0)
        rsh = np.clip(rsh, 1.0, 100000.0)
        nvth = np.clip(nvth, 0.1, 100.0)
        
        return cls(iph, i0, rs, rsh, nvth)


def explicit_diode_model_lambertw(
    voltage: np.ndarray, 
    params: DiodeParameters
) -> np.ndarray:
    """
    Calculate diode current using explicit Lambert W function solution.
    
    Uses macroscopic parameters (nNsVth), eliminating the need for separate 
    number of cells (Ns), ideality factor (n), or temperature (T).
    
    Args:
        voltage: Applied voltage array
        params: DiodeParameters object containing model parameters
    
    Returns:
        Current array corresponding to input voltage
    """
    V = np.asarray(voltage)
    Iph = params.photo_current
    I0 = params.reverse_saturation_current
    Rs = params.series_resistance
    Rsh = params.shunt_resistance
    nNsVth = params.thermal_voltage_product
    
    epsilon = 1e-12  # Prevents division by zero if Rs or Rsh are very small
    
    # 1. Exponential argument
    # Equation: (Rs*(Iph + I0) + V) / (nNsVth * (1 + Rs/Rsh))
    exp_argument = (Rs * (Iph + I0) + V) / (nNsVth * (1 + Rs / (Rsh + epsilon)) + epsilon)
    
    # Prevent mathematical overflow by clipping the exponential argument
    exp_argument = np.clip(exp_argument, -200, 200)
    
    # 2. Lambert W function argument
    # Equation: (I0 * Rs * Rsh) / (nNsVth * (Rs + Rsh)) * exp(exp_argument)
    multiplication_term = (I0 * Rs * Rsh) / (nNsVth * (Rs + Rsh) + epsilon)
    lambert_arg = multiplication_term * np.exp(exp_argument)
    
    # Prevent Lambert W function failure with extremely small or large numbers
    lambert_arg = np.clip(lambert_arg, 1e-15, 1e70)
    
    # 3. Apply Lambert W function (principal real branch)
    lambert_w = np.real(lambertw(lambert_arg))
    
    # 4. Final current equation
    # I = ((Iph + I0)*Rsh - V) / (Rs + Rsh) - (nNsVth / Rs) * W
    current = ((Iph + I0) * Rsh - V) / (Rs + Rsh + epsilon) - (nNsVth / (Rs + epsilon)) * lambert_w
    
    return current


def residuals_lambertw(
    params: np.ndarray, 
    voltage: np.ndarray, 
    measured_current: np.ndarray
) -> np.ndarray:
    """
    Calculate residuals between measured and predicted current.
    
    Args:
        params: Parameter array [Iph, I0, Rs, Rsh, nNsVth]
        voltage: Applied voltage array
        measured_current: Measured current array
    
    Returns:
        Weighted residuals for optimization
    """
    diode_params = DiodeParameters.from_array(params)
    
    # Generate current using Lambert W equation
    predicted_current = explicit_diode_model_lambertw(voltage, diode_params)
    
    # Mathematical protection: if solver attempts invalid parameters
    if np.any(np.isnan(predicted_current)):
        return np.full_like(measured_current, 1e6)  # Return large error
    
    # Apply higher weight to curve knee region (end of curve)
    weights = 1.0 + 5.0 * (voltage / voltage[-1]) ** 3
    
    # least_squares expects raw error (not squared); it squares internally
    return (measured_current - predicted_current) * np.sqrt(weights)


def refine_xgboost_predictions(
    voltage: np.ndarray, 
    measured_current: np.ndarray, 
    xgb_predictions: np.ndarray
) -> np.ndarray:
    """
    Refine XGBoost predictions using local optimization.
    
    Uses adaptive bounds to ensure physically valid parameters and smooth
    refinement of the initial XGBoost estimates.
    
    Args:
        voltage: Applied voltage array
        measured_current: Measured current array
        xgb_predictions: Initial parameter estimates from XGBoost [Iph, I0, Rs, Rsh, nNsVth]
    
    Returns:
        Optimized parameter array
    """
    # Create DiodeParameters with safety clipping on initial XGBoost prediction
    initial_params = DiodeParameters.from_xgboost(xgb_predictions)
    p0 = initial_params.to_array()
    
    # 2. Adaptive bounds (multipliers applied to XGBoost estimates)
    # Parameter order: [Iph, I0, Rs, Rsh, nNsVth]
    
    # Lower bounds (e.g., allow Iph to drop to 80% of prediction, I0 to 1/100th)
    lower_bounds = np.array([
        initial_params.photo_current * 0.8,  # Iph
        np.max([1e-18, initial_params.reverse_saturation_current / 100.0]),  # I0 (with absolute floor)
        initial_params.series_resistance * 0.1,  # Rs
        initial_params.shunt_resistance * 0.1,  # Rsh
        initial_params.thermal_voltage_product * 0.5  # nNsVth
    ])
    
    # Upper bounds (e.g., allow Iph to rise 20%, I0 to multiply by 100)
    upper_bounds = np.array([
        initial_params.photo_current * 1.2,  # Iph
        np.min([1e-2, initial_params.reverse_saturation_current * 100.0]),  # I0 (with absolute ceiling)
        initial_params.series_resistance * 5.0,  # Rs
        initial_params.shunt_resistance * 5.0,  # Rsh
        initial_params.thermal_voltage_product * 2.0  # nNsVth
    ])
    
    # 3. Emergency safeguard: least_squares fails if initial point equals bounds
    # Ensure initial guess is strictly within margin
    for i in range(5):
        if p0[i] <= lower_bounds[i]:
            p0[i] = lower_bounds[i] + 1e-9
        if p0[i] >= upper_bounds[i]:
            p0[i] = upper_bounds[i] - 1e-9
    
    # 4. Run local optimizer
    result = least_squares(
        residuals_lambertw,
        x0=p0,
        bounds=(lower_bounds, upper_bounds),
        args=(voltage, measured_current),
        method='trf',
        ftol=1e-8,
        max_nfev=200
    )
    
    return result.x