import os
# Turn off multithreading in math libraries
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import differential_evolution, root_scalar, minimize_scalar
from scipy.special import lambertw
import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed 

# Exact Explicit Model (Lambert W) in Normalized Space
def normalized_diode_lambertw(v, params):
    """
    Calculate the normalized current using Lambert W.

    v: Normalized voltage (0 to 1)
    params: [i_ph, i_0, r_s, r_sh, alpha] (All in normalized space)
    """
    i_ph, i_0, r_s, r_sh, alpha = params
    eps = 1e-12

    arg_exp = (r_s * (i_ph + i_0) + v) / (alpha * (1 + r_s / (r_sh + eps)) + eps)
    arg_exp = np.clip(arg_exp, -200, 200) 
    
    term_mult = (i_0 * r_s * r_sh) / (alpha * (r_s + r_sh) + eps)
    arg = term_mult * np.exp(arg_exp)
    
    arg = np.clip(arg, 1e-15, 1e70)
    
    # Lambert W function
    W = np.real(lambertw(arg))
    
    # Normalized current calculated
    i_calc = ((i_ph + i_0) * r_sh - v) / (r_s + r_sh + eps) - (alpha / (r_s + eps)) * W
    
    return i_calc

# Data Preparation (Double Normalization)
def build_iv_arrays(sample):
    colums_current = [f'current_{i}' for i in range(181)]
    colums_voltage = [f'voltage_{i}' for i in range(181)]
    I_all = sample[colums_current].values
    V_all = sample[colums_voltage].values
    return V_all, I_all

def double_normalize(V, I):
    """Normalizes V and I to the space [0, 1]"""
    Isc = I[0] + 1e-8
    Voc = V[-1] + 1e-8 # Gets the last voltage point
    
    v_norm = V / Voc
    i_norm = I / Isc
    
    return v_norm, i_norm, Voc, Isc


# Loss And Individual Fit
def fit_single_curve_lambert(v_norm, i_meas_norm):
    
    # Physical Blockage: Strict Limits Based On The Reality Of Semiconductors.
    bounds = [
        (0.98, 1.05),     # 0: i_ph 
        (-12.0, -3.0),    # 1: log10(i_0) (logarithmic space)
        (0.0001, 0.15),   # 2: r_s 
        (0.5, 4.0),       # 3: log10(r_sh) (logarithmic space)
        (0.02, 0.25)      # 4: alpha 
    ]

    def loss_wrapper(p):
        p_real = [p[0], 10**p[1], p[2], 10**p[3], p[4]]
        i_pred = normalized_diode_lambertw(v_norm, p_real)
        
        # Mathematical failure protection: If the Lambert W explodes or gives NaN, return a huge loss to steer away from this region.
        if np.any(np.isnan(i_pred)):
            return 1e6
        
        # 1. Current RMSE (General curve adjustment)
        rmse_i = np.sqrt(np.mean((i_meas_norm - i_pred) ** 2))
        
        # 2. RMSE of Power (Forces the "knee" to become perfect)
        p_meas = v_norm * i_meas_norm
        p_pred = v_norm * i_pred
        rmse_p = np.sqrt(np.mean((p_meas - p_pred) ** 2))
        
    
        return rmse_i + rmse_p

    result = differential_evolution(
        loss_wrapper, bounds,
        maxiter=150, popsize=15, tol=1e-7, polish=True, disp=False
    )

    melhores_params = [
        result.x[0],          
        10 ** result.x[1],    
        result.x[2],          
        10 ** result.x[3],    
        result.x[4]           
    ]

    return melhores_params

# Physical extraction and denormalization
def extract_macros_lambert(V_real, params_real):
    """ Extract macros using Lambert W directly at real scale """
    Iph, I0, Rs, Rsh, nNsVth = params_real
    
    def calc_I(V):
        
        eps = 1e-12
        arg_exp = (Rs * (Iph + I0) + V) / (nNsVth * (1 + Rs / (Rsh+eps)) + eps)
        arg_exp = np.clip(arg_exp, -200, 200)
        arg = ((I0 * Rs * Rsh) / (nNsVth * (Rs + Rsh) + eps)) * np.exp(arg_exp)
        W = np.real(lambertw(np.clip(arg, 1e-15, 1e70)))
        return ((Iph + I0) * Rsh - V) / (Rs + Rsh + eps) - (nNsVth / (Rs + eps)) * W

    Isc_model = calc_I(0.0)

    def voc_eq(V):
        return Iph - I0 * (np.exp(np.clip(V / nNsVth, -100, 100)) - 1) - V / Rsh

    try:
        res_voc = root_scalar(voc_eq, bracket=[0, V_real[-1]*1.5], method='brentq')
        Voc_model = res_voc.root
    except ValueError:
        Voc_model = np.nan 

    def neg_power(V):
        return -(V * calc_I(V))

    if not np.isnan(Voc_model):
        res_mpp = minimize_scalar(neg_power, bounds=(0, Voc_model), method='bounded')
        Vmp_model = res_mpp.x
        Imp_model = calc_I(Vmp_model)
        Pmp_model = Vmp_model * Imp_model
    else:
        Vmp_model, Imp_model, Pmp_model = np.nan, np.nan, np.nan

    return {
        'Isc_model (A)': Isc_model,
        'Voc_model (V)': Voc_model,
        'Vmp_model (V)': Vmp_model,
        'Imp_model (A)': Imp_model,
        'Pmp_model (W)': Pmp_model
    }

# Worker's function
def worker_process_curve(args):
    idx, V_real, I_real = args
    
    # 1. Double Normalization
    v_norm, i_norm, Voc, Isc = double_normalize(V_real, I_real)
    
    # 2. Find parameters in the space (0 to 1)
    i_ph, i_0, r_s, r_sh, alpha = fit_single_curve_lambert(v_norm, i_norm)

    # 3. Exact physical denormalization
    Iph_real = i_ph * Isc
    I0_real  = i_0 * Isc
    Rs_real  = r_s * (Voc / Isc)
    Rsh_real = r_sh * (Voc / Isc)
    nNsVth_real = alpha * Voc 
    
    params_reais = [Iph_real, I0_real, Rs_real, Rsh_real, nNsVth_real]

    # 4. Extracting Macros
    macros = extract_macros_lambert(V_real, params_reais)

    linha_resultado = {
        "Index_Original": idx, 
        "Iph": Iph_real,
        "I0": I0_real,
        "Rs": Rs_real,
        "Rsh": Rsh_real,
        "nNsVth": nNsVth_real,
    }
    linha_resultado.update(macros)
    
    return linha_resultado

# =========================================================
# CALLED MAIN
# =========================================================
if __name__ == "__main__":
   
    base_path = './dados/'

    if os.path.exists(base_path):
        arquivos = [arq for arq in os.listdir(base_path) if arq.endswith('.feather')]

        for arq_name in arquivos:
            

            if 'Tandem' in arq_name or 'Triple' in arq_name:
                continue

            df = pd.read_feather(f'{base_path}{arq_name}')
            n_amostras = 400 
            sample_full = df.sample(n=n_amostras, random_state=42)
            
            indices_originais = sample_full.index.values
            V_all, I_all = build_iv_arrays(sample_full)

            tarefas = [
                (indices_originais[i], V_all[i], I_all[i])
                for i in range(n_amostras)
            ]

            resultados = []

            with ProcessPoolExecutor() as executor:
                futuros = {executor.submit(worker_process_curve, tarefa): tarefa for tarefa in tarefas}
                
                for futuro in tqdm(as_completed(futuros), total=n_amostras, colour="cyan"):
                    resultado = futuro.result() 
                    resultados.append(resultado)

            df_out = pd.DataFrame(resultados)
            df_out.set_index("Index_Original", inplace=True)
            df_out.sort_index(inplace=True)
            
            output_path = f"{arq_name.split('.')[0]}_LAMBERT_EXATO.parquet"
            df_out.to_parquet(output_path)
            
            

    else:
        print(f"Folder '{base_path}' not found.")