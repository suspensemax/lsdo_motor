import numpy as np
import csdl_alpha as csdl

from lsdo_motor.core.permeability.fitting_functions import fit_dep_B, fit_dep_H

def magnet_mec_model(num_nodes, inputs, fit_coeff_dep_H, fit_coeff_dep_B):
    B_delta = csdl.ImplicitVariable(name='B_delta', value=1.e-3, shape=(num_nodes,))

    ''' --- STATOR TOOTH CALCULATIONS --- '''
    t1 = inputs['tooth_pitch'] # TOOTH PITCH
    b1 = inputs['tooth_width'] # TOOTH WIDTH
    h_t1 = inputs['slot_height'] # DEPTH OF SLOT (h_slot in MATLAB code)
    kfe = 0.95 # lamination coefficient
    
    B_t = B_delta * t1/b1/kfe # STATOR TOOTH FLUX DENSITY
    H_t = fit_dep_B(B_t, fit_coeff_dep_B[0], fit_coeff_dep_B[1], fit_coeff_dep_B[2]) # STATOR TOOTH MAGNETIC FIELD
    F_t = 2*H_t*h_t1 # MMF OF TOOTH

    ''' --- YOKE & ROTOR TOOTH CALCULATIONS --- '''
    alpha_i = inputs['alpha_i'] # ELECTRICAL ANGLE PER SLOT
    tau = inputs['pole_pitch'] # POLE PITCH
    l_ef = inputs['l_ef'] # MAGNET LENGTH ALONG SHAFT (TYPICALLY STACK LENGTH)
    hj1 = inputs['height_yoke_stator'] # YOKE HEIGHT IN STATOR
    ly = inputs['L_j1'] # STATOR YOKE LENGTH OF MEC (CALLED L_j1 IN MATLAB & SIZING MODEL)

    phi_air = alpha_i*tau*l_ef*B_delta # AIR GAP FLUX; CALLED phi_air IN MATLAB CODE

    B_y = phi_air / (2*hj1*l_ef) # YOKE FLUX DENSITY
    H_y = fit_dep_B(B_y, fit_coeff_dep_B[0], fit_coeff_dep_B[1], fit_coeff_dep_B[2]) # YOKE MAGNETIC FIELD
    F_y = 2*ly*H_y # YOKE MMF

    ''' --- AIR GAP MMF CALCULATIONS --- '''
    mu_0 = np.pi*4e-7
    sigma_air = inputs['air_gap_depth'] # AIR GAP DEPTH
    K_theta = inputs['K_theta'] # CARTER'S COEFF; CALLED K_theta IN MATLAB & SIZING MODEL

    F_delta = 1.6*B_delta*(0.0001 + K_theta*sigma_air)/mu_0
    
    ''' --- MMF SUM --- '''
    F_total = F_t + F_y + F_delta

    ''' --- MAGNETIC BRIDGE CALCULATIONS --- '''
    hm = inputs['hm']
    H_f = F_total / hm
    B_f = fit_dep_H(H_f, fit_coeff_dep_H[0])

    ''' --- LEAKAGE FLUX CALCULATIONS --- '''
    # NOTE: phi_air already calculated above
    A_f2 = inputs['A_f2'] # CROSS SECTIONAL AREA OF MAGNET BRIDGE
    lambda_s = 0.336e-6
    # NOTE: lambda_s is not typically a constant so need to check later

    phi_f = B_f * A_f2
    phi_s = F_total * lambda_s
    
    phi_mag = phi_air + phi_f + phi_s

    ''' --- MAGNET MMF CALCULATIONS --- '''
    bm = inputs['bm'] # ARC LENGTH OF MAGNET
    lambda_leak = l_ef*bm*mu_0/0.0005
    F_m = F_total + phi_mag/lambda_leak # MAGNET MMF

    # RESIDUAL FLUX OF MAGNET (COMPUTED IN SIZING MODEL)
    phi_r = inputs['phi_r'] 
    lambda_m = inputs['lambda_m']

    phi_air = phi_r - F_m*lambda_m

    residual = phi_air - phi_mag
    eps = 1e-6
    Br  = 1.2
    solver = csdl.nonlinear_solvers.BracketedSearch('magnet_mec_bs')
    solver.add_state(B_delta, residual, bracket=(eps, Br))
    solver.run()

    # --- MEC POST-PROCESSING ---
    mu_0 = np.pi*4e-7
    hm = 0.004 # MAGNET THICKNESS
    mu_r = 1.05
    Am_r = inputs['Am_r']

    K_sigma_air = (phi_air+phi_f+phi_s)/phi_air

    lambda_theta = phi_air/F_total # MAIN MAGNETIC CONDUCTION
    lambda_theta_standard = (2*lambda_theta*hm) / (mu_r*mu_0*Am_r) # STANDARD VALUE OF lambda_theta
    lambda_n = K_sigma_air*lambda_theta_standard

    bm_0 =  lambda_n/(lambda_n + 1) # OPERATING POINT OF MAGNET
    lambda_leak_standard = (K_sigma_air - 1)*lambda_theta_standard

    '''
    Outputs needed in other models:
    - phi_air
    - F_total
    - F_delta
    - K_sigma_air
    - lambda_n
    - lambda_leak_standard
    '''

    outputs = {
        'B_delta': B_delta,
        'phi_air': phi_air,
        'F_total': F_total,
        'F_delta': F_delta,
        'K_sigma_air': K_sigma_air,
        'lambda_n': lambda_n,
        'lambda_leak_standard': lambda_leak_standard
    }

    return outputs