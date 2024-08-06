import numpy as np
import csdl_alpha as csdl

from lsdo_motor.core.permeability.fitting_functions import fit_dep_H, fit_dep_B

def inductance_model(inputs, fit_coeff_dep_B, num_nodes, p, m, Z, I_w_r, omega_rated):

    f_i = omega_rated*p/60. 
    I_d_r = I_w_r * np.sin(0.6283)
    Kf = inputs['Kf']
    Kaq = 0.36/Kf
    phi_air = inputs['phi_air']

    # region inductance residual computation
    phi_aq = csdl.ImplicitVariable(name='phi_aq', value=1.e-3, shape=(num_nodes,))

    alpha_i = inputs['alpha_i']
    pole_pitch = inputs['pole_pitch']
    l_ef = inputs['l_ef']
    B_aq = phi_aq/(alpha_i*pole_pitch*l_ef)

    K_theta = inputs['K_theta']
    air_gap_depth = inputs['air_gap_depth']
    mu_0 = np.pi*4e-7
    F_sigma_q = 1.6*B_aq*(K_theta*air_gap_depth)/mu_0

    tooth_pitch = inputs['tooth_pitch']
    tooth_width = inputs['tooth_width']
    kfe = 0.95 # LAMINATION COEFFICIENT

    B_t_q = B_aq*tooth_pitch/tooth_width/kfe
    H_t1_q = fit_dep_B(B_t_q, fit_coeff_dep_B[0], fit_coeff_dep_B[1], fit_coeff_dep_B[2])
    h_slot = inputs['h_slot']
    F_t1_q = 2*H_t1_q*h_slot # DIFF OF MAGNETIC POTENTIAL ALONG TOOTH

    h_ys = inputs['h_ys']
    B_j1_q = phi_aq/(2*l_ef*h_ys)
    H_j1_q = fit_dep_B(B_j1_q, fit_coeff_dep_B[0], fit_coeff_dep_B[1], fit_coeff_dep_B[2])

    Kdp1 = inputs['Kdp1']
    turns_per_phase = inputs['turns_per_phase']
    L_j1 = inputs['L_j1']
    F_j1_q = 2*H_j1_q*L_j1
    F_total_q = F_sigma_q + F_t1_q + F_j1_q # TOTAL MAGNETIC STRENGTH ON Q-AXIS
    I_q_temp = p*F_total_q/(0.9*m*Kaq*Kdp1*turns_per_phase) #  CURRENT AT Q-AXIS
    
    inductance_residual = I_d_r**2 + I_q_temp**2 - I_w_r**2
    # endregion

    eps = 1.e-5
    solver = csdl.nonlinear_solvers.BracketedSearch('inductance_bs')
    solver.add_state(phi_aq, inductance_residual, bracket=(eps, phi_air))
    solver.run()

    # region inductance post-processing
    F_total = inputs['F_total']
    F_delta = inputs['F_delta']
    mu_0 = np.pi*4e-7

    K_st = F_total/F_delta
    Cx = (4*np.pi*f_i*mu_0*l_ef*(Kdp1*turns_per_phase)**2) / p

    h_k = 0.0008 # NOT SURE HWAT THIS IS
    h_os = 1.5 * h_k # NOT SURE WHAT THIS IS
    b_sb = inputs['b_sb']
    b_s1 = inputs['b_s1']

    lambda_U1 = (h_k/b_sb) + (2*h_os/(b_sb+b_s1))
    lambda_L1 = 0.45
    lambda_S1 = lambda_U1 + lambda_L1

    X_s1 = (2*p*m*lambda_S1*Cx)/(Z*Kdp1**2)
    s_total = 0.1

    X_d1 = (m*pole_pitch*s_total*Cx) / (air_gap_depth*K_theta*K_st*(np.pi*Kdp1)**2)

    l_B = l_ef + 2*0.01 # straight length of coil
    Tau_y = inputs['Tau_y']

    X_E1 = 0.47*Cx*(l_B - 0.64*Tau_y)/(l_ef*Kdp1**2)
    X_1 = X_s1+X_d1+X_E1

    Kad = 1./Kf
    
    F_ad = 0.35*m*Kad*Kdp1*turns_per_phase*I_d_r/p
    hm = 0.004 # MAGNET THICKNESS
    Hc = 847138 # MAGNET COERCIVITY
    K_sigma_air = inputs['K_sigma_air'] # COEFFICIENT OF LEAKAGE IN AIR GAP

    f_a = F_ad / (K_sigma_air*hm*Hc)

    lambda_n = inputs['lambda_n']
    lambda_leak_standard = inputs['lambda_leak_standard']
    Am_r = inputs['Am_r']
    Br = 1.1208 # MAGNET REMANENCE
    K_phi = inputs['K_phi']

    E_o = 4.44*f_i*Kdp1*turns_per_phase*phi_air*K_phi
    aa = 1.0 + (-1 * f_a) # 1 - f_a DOES NOT WORK
    bb = lambda_n * aa
    cc = lambda_n + 1
    bm_N = bb/cc
    phi_air_N_temp = bm_N + (-1) * (1.0 + (-1) * bm_N)*lambda_leak_standard
    phi_air_N = phi_air_N_temp * Am_r * Br
    E_d = 4.44*f_i*Kdp1*turns_per_phase*phi_air_N*K_phi # EM at d-axis
    Xad  = ((E_o-E_d)**2)**0.5/I_d_r/2**0.5
    Xd = Xad + X_1
    Ld = Xd / (2*np.pi*f_i) # d-axis inductance

    E_aq = phi_aq*E_o/phi_air # EMF @ Q-AXIS
    Xaq = E_aq/I_q_temp
    Xq = Xaq + X_1
    Lq = Xq/(2*np.pi*f_i)
    # endregion

    inductance_outputs = {
        'Ld': Ld,
        'Lq': Lq
    }
    
    return inductance_outputs