import numpy as np
import csdl_alpha as csdl

def torque_regression_function(mass, order=1):
    fitting_coeff_dict = {
        '0':[1],
        # '1':[26.0489, -112.1432],
        '1':[25., 25.], # close to the one above, but crosses the x axis when motor mass < 0
        '2':[0.4840, 3.3169, 60.8142],
        '3':[1, 1, 1 ,1],
        '4':[1, 1, 1, 1, 1],
    }

    fitting_coeff = fitting_coeff_dict.get(str(order))
    max_torque = 0.
    for i, val in enumerate(fitting_coeff):
        max_torque = max_torque + val*mass**(order-i)

    return max_torque

def motor_sizing_model(L, omega, parameters, sizing_mode='torque', torque_density=15., power_density=7., A_bar=30.e3, B_bar=0.85, L_D=0.5):
    '''
    power density units are given as kW/kg
    '''

    # region torque based sizing (from mass)
    D = L/(L_D)
    # torque_density = 10.
    # torque_density = 500/27
    if sizing_mode == 'torque':
        mass = np.pi/2.*A_bar*B_bar*D**2*L/torque_density
    elif sizing_mode == 'power':
        mass = np.pi/2.*A_bar*B_bar*D**2*L*omega/power_density/1000
    
    # mass.print_on_update('mass')

    peak_torque = torque_regression_function(mass, order=2)
    # peak_torque.print_on_update('peak_torque')

    # endregion

    # region original sizing method
    m = parameters['phases']
    p = parameters['pole_pairs']
    Z = parameters['num_slots']
    I_w = parameters['rated_current']

    q = Z/(2*m*p) # SLOTS PER POLE PER PHASE
    mu_0 = np.pi*4e-7 # air permeability

    rated_omega = parameters['rated_omega']
    a = 1. # parallel branches
    eta_0 = 0.96 # assumed initial efficiency
    PF = 1 # power factor
    
    f_i = rated_omega*p/60
    B_air_gap_max = 0.85 # MAX VALUE OF B IN AIR GAP
    alpha_B = 0.7 # RATIO OF B_avg/B_max IN AIR GAP [0.66, 0.71]

    line_load = 30000 # CURRENT PER UNIT LENGTH; THIS MAY NEED TO BE UPDATED
    # NOTE: THIS IS THE ELECTRIC LOADING THAT ZEYU FOUND FROM A LOOKUP TABLE IN A CHINESE TEXTBOOK;
    line_load = A_bar

    lambda_i = 1.25
    D_i = D / lambda_i # finding inner_radius of stator

    # --- POLE PITCH AND OTHER PITCHES ---
    pole_pitch = np.pi*D_i/(2*p)
    tooth_pitch = np.pi*D_i/Z

    # --- AIR GAP LENGTH ---
    air_gap_depth = 0.4*line_load*pole_pitch/(0.9e6*B_air_gap_max)
    # l_ef = L + 2*air_gap_depth # final effective length of motor
    l_ef = L*1
    rotor_diameter = D_i - 2*air_gap_depth # D2 in MATLAB code
    D_shaft = 0.3 * rotor_diameter # outer radius of shaft
    
    # --- WINDINGS ---
    I_kw = I_w * eta_0 * PF
    conductors_per_phase = eta_0 * np.pi * D_i * line_load \
        / (m*I_kw)
    conductors_per_slot = m*a*conductors_per_phase/Z
    turns_per_phase = conductors_per_phase/2
    # these lines of code will cause errors compared to MATLAB code bc Zeyu
    # uses floor to round, and we cannot do that

    N_p = 2
    J = 5. # target current density
    Acu = I_w/(a*J*N_p) * 10.**(-6.)
    d_coil = 2 * (Acu/np.pi)**0.5
    
    # --- SLOT GEOMETRY ---
    kfe = 0.95 # LAMINATION COEFFICIENT
    Bt = 1.7 # FLUX DENSITY IN STATOR TOOTH

    tooth_width = tooth_pitch * B_air_gap_max / (kfe * Bt) # STATOR TOOTH WIDTH

    B_ys = 1.35 # FLUX DENSITY IN STATOR YOKE
    h_ys = (pole_pitch*alpha_B*B_air_gap_max) / (2*kfe*B_ys) # HEIGHT OF YOKE IN STATOR

    theta_t = 360/Z # ANGULAR SWEEP OF STATOR SLOT IN DEGREES
    theta_sso = 0.5*theta_t
    theta_ssi = 0.3*theta_sso
    b_sb = theta_ssi*np.pi*D_i/360 # WIDTH OF BOTTOM OF SLOT
    h_slot = (D - D_i)/2 - h_ys # HEIGHT OF SLOT

    h_k = 0.0008 # NOT SURE HWAT THIS IS
    h_os = 1.5 * h_k # NOT SURE WHAT THIS IS

    b_s1 = (np.pi*(D_i+2*(h_os+h_k)))/36 - tooth_width # RADIALLY INNER WIDTH OF SLOT
    b_s2 = (np.pi*(D_i+2*h_slot))/36 - tooth_width # RADIALLY OUTER WIDTH OF SLOT

    Tau_y = np.pi*(D_i+h_slot) / (2*p)
    L_j1 = np.pi*(D-h_ys) / (4*p) # STATOR YOKE LENGTH FOR MAGNETIC CIRCUIT CALCULATION

    # --- WINDING FACTOR ---
    Kp1 = csdl.sin(pole_pitch*90*np.pi/pole_pitch/180) # INTEGRAL WINDING

    alpha = 360*p/Z # ELECTRICAL ANGLE PER SLOT
    Kd1 = np.sin(q*alpha/2)/(q*np.sin(alpha/2))
    Kdp1 = Kd1*Kp1

    # --- MAGNET GEOMETRY ---
    hm = 0.004 # MAGNET THICKNESS
    theta_p = 360/2/p # ANGULAR SWEEP OF POLE IN DEGREES
    theta_m = 0.78*theta_p
    magnet_embed_depth = 0.002
    Dm = rotor_diameter - magnet_embed_depth
    bm = Dm*np.pi*theta_m/360 

    T = 75 # assuming normal operating temp
    Br_20 = 1.2 # Br at 20 C
    alpha_Br = -0.12 # temperature coefficients
    Br = 1+(T-20)*alpha_Br/100*Br_20

    Hc_20 = 907000; # coercivity
    Hc = (1+(T-20)*alpha_Br/100)*Hc_20

    Am_r = bm*l_ef # RADIAL CROSS SECTIONAL AREA OF MAGNET
    rho_magnet = 7.6 # MAGNET DENSITY (g/cm^3)
    mass_magnet = 2*p*bm*hm*l_ef*rho_magnet*1e3 # MAGNET MASS

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
    # A_f2 = hm*l_ef

    # --- RESISTANCE & MASS CALCULATION
    rho = 0.0217e-6 # RESISTIVITY ------ GET CLARIFICATION ON UNITS
    l_B = l_ef + 2*0.01 # straight length of coil
    l_coil = l_B + 2.0*pole_pitch # length of half-turn
    
    Rdc = 2 * rho * turns_per_phase * l_coil / (Acu * N_p) # DC RESISTANCE

    msvg = csdl.VariableGroup()
    msvg.L = L
    msvg.D = D # outer diameter
    msvg.mass = mass
    msvg.peak_torque = peak_torque
    msvg.D_i = D_i
    msvg.D_shaft = D_shaft
    msvg.pole_pitch = pole_pitch
    msvg.tooth_pitch = tooth_pitch
    msvg.air_gap_depth = air_gap_depth
    # msvg.l_ef = l_ef
    msvg.rotor_diameter = rotor_diameter
    msvg.turns_per_phase = turns_per_phase
    msvg.tooth_width = tooth_width
    msvg.h_ys = h_ys
    msvg.b_sb = b_sb
    msvg.h_slot = h_slot
    msvg.b_s1 = b_s1
    msvg.magnet_thickness = hm
    msvg.magnet_embed_depth = magnet_embed_depth # magnet embedded depth
    msvg.Acu = Acu

    # print(L.value)
    # print(D.value) # outer diameter
    # print(mass.value)
    # print(peak_torque.value)
    # print(D_i.value)
    # print(D_shaft.value)
    # print(pole_pitch.value)
    # print(tooth_pitch.value)
    # print(air_gap_depth.value)
    # print(rotor_diameter.value)
    # print(turns_per_phase.value)
    # print(tooth_width.value)
    # print(h_ys.value)
    # print(b_sb.value)
    # print(h_slot.value)
    # print(b_s1.value)
    # print(hm)
    # print(magnet_embed_depth) # magnet embedded depth
    # print(Acu)
    # exit()
    

    # msvg.Tau_y = Tau_y
    # msvg.L_j1 = L_j1
    # msvg.Kdp1 = Kdp1
    # msvg.Am_r = Am_r
    # msvg.phi_r = phi_r
    # msvg.lambda_m = lambda_m
    # msvg.alpha_i = alpha_i
    # msvg.Kf = Kf
    # msvg.K_phi = K_phi
    # msvg.K_theta = K_theta
    # msvg.A_f2 = A_f2
    # msvg.Rdc = Rdc

    
    # endregion

    return msvg