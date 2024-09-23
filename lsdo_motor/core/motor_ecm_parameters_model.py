import numpy as np
import csdl_alpha as csdl

def motor_ecm_parameters_model(parameters, sizing_inputs):

    # region original sizing method
    m = parameters['phases']
    p = parameters['pole_pairs']
    Z = parameters['num_slots']
    I_w = parameters['rated_current']

    L = sizing_inputs.L
    D = sizing_inputs.D # outer diameter
    D_i = sizing_inputs.D_i
    pole_pitch = sizing_inputs.pole_pitch
    tooth_pitch = sizing_inputs.tooth_pitch
    air_gap_depth = sizing_inputs.air_gap_depth
    # l_ef = sizing_inputs.l_ef
    rotor_diameter = sizing_inputs.rotor_diameter
    turns_per_phase = sizing_inputs.turns_per_phase
    Acu = sizing_inputs.Acu
    tooth_width = sizing_inputs.tooth_width
    h_ys = sizing_inputs.h_ys # HEIGHT OF YOKE IN STATOR
    b_sb = sizing_inputs.b_sb # WIDTH OF BOTTOM OF SLOT
    h_slot = sizing_inputs.h_slot # HEIGHT OF SLOT
    b_s1 = sizing_inputs.b_s1 # RADIALLY INNER WIDTH OF SLOT
    hm = sizing_inputs.magnet_thickness
    magnet_embed_depth = sizing_inputs.magnet_embed_depth

    l_ef = L

    q = Z/(2*m*p) # SLOTS PER POLE PER PHASE
    mu_0 = np.pi*4e-7 # air permeability

    # line_load = 30000 # CURRENT PER UNIT LENGTH; THIS MAY NEED TO BE UPDATED
    # # NOTE: THIS IS THE ELECTRIC LOADING THAT ZEYU FOUND FROM A LOOKUP TABLE IN A CHINESE TEXTBOOK;
    # line_load = A_bar

    
    # # --- WINDINGS ---
    # PF = 1 # power factor
    # eta_0 = 0.96 # assumed initial efficiency
    # I_kw = I_w * eta_0 * PF
    # conductors_per_phase = eta_0 * np.pi * D_i * line_load \
    #     / (m*I_kw)
    # conductors_per_slot = m*a*conductors_per_phase/Z
    # turns_per_phase = conductors_per_phase/2
    # # these lines of code will cause errors compared to MATLAB code bc Zeyu
    # # uses floor to round, and we cannot do that

    N_p = 2

    Tau_y = np.pi*(D_i+h_slot) / (2*p)
    L_j1 = np.pi*(D-h_ys) / (4*p) # STATOR YOKE LENGTH FOR MAGNETIC CIRCUIT CALCULATION

    # --- WINDING FACTOR ---
    Kp1 = csdl.sin(pole_pitch*90*np.pi/pole_pitch/180) # INTEGRAL WINDING

    alpha = 360*p/Z # ELECTRICAL ANGLE PER SLOT
    Kd1 = np.sin(q*alpha/2)/(q*np.sin(alpha/2))
    Kdp1 = Kd1*Kp1

    # --- MAGNET GEOMETRY ---
    theta_p = 360/2/p # ANGULAR SWEEP OF POLE IN DEGREES
    theta_m = 0.78*theta_p
    Dm = rotor_diameter - magnet_embed_depth
    bm = Dm*np.pi*theta_m/360

    T = 75 # assuming normal operating temp
    Br_20 = 1.2 # Br at 20 C
    alpha_Br = -0.12 # temperature coefficients
    Br = 1+(T-20)*alpha_Br/100*Br_20

    Hc_20 = 907000; # coercivity
    Hc = (1+(T-20)*alpha_Br/100)*Hc_20
    mu_r = Br/mu_0/Hc; # relative permeability

    Am_r = bm*l_ef # RADIAL CROSS SECTIONAL AREA OF MAGNET

    phi_r = 1.2*Am_r
    Fc = 2*Hc*hm

    lambda_m = phi_r/Fc
    alpha_p1 = bm/pole_pitch
    alpha_i = alpha_p1+4/((pole_pitch/air_gap_depth)+(6/(1-alpha_p1)))

    Kf = 4*csdl.sin(alpha_i*np.pi/2)/np.pi # COEFF OF MAGNETIC FLUX DENSITY ALONG AIR GAP
    K_phi = 8.5*csdl.sin(alpha_i*np.pi/2)/(np.pi**2*alpha_i) # COEFF OF FLUX ALONG AIR GAP
    K_theta1 = tooth_pitch*(4.4*air_gap_depth + 0.75*b_sb)/(tooth_pitch*(4.4*air_gap_depth + 0.75*b_sb)-b_sb**2)
    K_theta2 = 1 # no rotor slot

    K_theta = K_theta1*K_theta2
    l_f2 = hm
    A_f2 = l_f2*l_ef

    # --- RESISTANCE & MASS CALCULATION
    rho = 0.0217e-6 # RESISTIVITY ------ GET CLARIFICATION ON UNITS
    l_B = l_ef + 2*0.01 # straight length of coil
    l_coil = l_B + 2.0*pole_pitch # length of half-turn
    
    Rdc = 2 * rho * turns_per_phase * l_coil / (Acu * N_p) # DC RESISTANCE

    ecmp_vg = csdl.VariableGroup()
    ecmp_vg.Tau_y = Tau_y
    ecmp_vg.L_j1 = L_j1
    ecmp_vg.Kdp1 = Kdp1
    ecmp_vg.bm = bm
    ecmp_vg.Am_r = Am_r
    ecmp_vg.phi_r = phi_r
    ecmp_vg.lambda_m = lambda_m
    ecmp_vg.alpha_i = alpha_i
    ecmp_vg.Kf = Kf
    ecmp_vg.K_phi = K_phi
    ecmp_vg.K_theta = K_theta
    ecmp_vg.A_f2 = A_f2
    ecmp_vg.Rdc = Rdc
    
    return ecmp_vg