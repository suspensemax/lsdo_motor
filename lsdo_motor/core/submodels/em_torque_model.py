import numpy as np 
import csdl_alpha as csdl 
from typing import Union

from lsdo_motor.core.submodels.flux_weakening_model import flux_weakening_model
from lsdo_motor.core.submodels.mtpa_model import mtpa_model

def em_torque_model(inputs, parameters, mode, mtpa=True, flux_weakening=False,):
    num_nodes = parameters['num_nodes']
    rated_current = parameters['rated_current']
    p = parameters['pole_pairs']
    V_lim = parameters['V_lim']
    m = parameters['phases']

    # region EM torque residual computation
    if mode == 'input_load':
        state = T_em = csdl.ImplicitVariable(name='T_em', value=100., shape=(num_nodes,))
        load_torque = inputs['load_torque']
    elif mode == 'efficiency_map':
        state = load_torque = csdl.ImplicitVariable(name='load_torque', value=100., shape=(num_nodes,))
        T_em = inputs['T_em']

    R = inputs['R']
    Ld = inputs['Ld']
    Lq = inputs['Lq']
    PsiF = inputs['PsiF']
    Iq_rated = inputs['Iq_rated']
    omega = inputs['omega']

    upper_T_lim = inputs['upper_T_lim']

    f_i = 5000*p/60 # rated omega from sizing model = 3000
    U_d = -R*rated_current*np.sin(0.6283) - 2*np.pi*f_i*Lq*Iq_rated
    U_q = R*rated_current*np.sin(0.6283) + 2*np.pi*f_i*(PsiF - Ld*Iq_rated)
    U_rated = (U_d**2 + U_q**2)**(1/2)

    if flux_weakening:
        flux_weakening_inputs = {}
        Iq_fw = flux_weakening_model()

    mtpa_inputs = {
        'T_em': T_em,
        'Ld': Ld,
        'Lq': Lq,
        'PsiF': PsiF
    }
    mtpa_parameters = {'p': p,  'num_nodes': num_nodes}
    Iq_MTPA = mtpa_model(mtpa_inputs, mtpa_parameters)

    Id_MTPA = (Ld - Lq)**(-1) * (T_em/(3/2*p*Iq_MTPA) - PsiF)
    U_d_MTPA = R*Id_MTPA - omega*Lq*Iq_MTPA
    U_q_MTPA = omega*Ld*Id_MTPA + R*Iq_MTPA + omega*PsiF
    U_MTPA = (U_d_MTPA**2 + U_q_MTPA**2)**0.5

    if flux_weakening:
        k = .5 
        I_q = (csdl.exp(k*(U_MTPA - V_lim))*Iq_fw + Iq_MTPA) / (csdl.exp(k*(U_MTPA - V_lim)) + 1.0)
    else:
        I_q = Iq_MTPA * 1.

    I_d = (T_em / (1.5*p*I_q) - PsiF) / (Ld-Lq) # CHECK SIZE OF COMPUTATIONS HERE
    current_amplitude = (I_q**2 + I_d**2)**0.5

    ''' ==== POWER LOSS CALCULATIONS ==== '''
    # load power
    # eq of the form P0 = speed * torque
    # P0 = load_torque * omega * 2*np.pi/60
    P0 = load_torque*omega/p
    # P0 = T_em * omega * 2*np.pi/60/p
    # P0 = T_em * omega/p
    output_power = P0
    frequency = omega*p/60

    # copper loss
    copper_loss = m*R*current_amplitude**2

    # eddy_loss
    a = 0.00055 # lamination thickness in m
    sigma_c = 2e6 # bulk conductivity (2000000 S/m)

    D_i = inputs['D_i']
    B_delta = inputs['B_delta']
    l_ef = inputs['l_ef']
    D1 = inputs['outer_stator_diameter'] # outer_stator_radius
    D2 = inputs['rotor_diameter'] # rotor radius
    D_shaft = inputs['shaft_diameter']
    bm = inputs['bm']
    hm = inputs['hm'] # MAGNET THICKNESS
    Acu = inputs['Acu']
    # B_delta_expanded = csdl.expand(B_delta, (num_nodes,))
    B_delta_expanded = B_delta
    
    K_e = (a*np.pi)**2 * sigma_c/60
    # V_s = csdl.expand((np.pi*l_ef*(D1-D_i)**2)/4-36*l_ef*Acu, (num_nodes,)); # volume of stator
    # V_s1 = csdl.expand((np.pi*l_ef*(D2-D_shaft)**2)/4, (num_nodes,))
    # V_t = csdl.expand(2*p*l_ef*bm*hm, (num_nodes,))
    V_s = (np.pi*l_ef*(D1-D_i)**2)/4-36*l_ef*Acu; # volume of stator
    V_s1 = (np.pi*l_ef*(D2-D_shaft)**2)/4
    V_t = 2*p*l_ef*bm*hm
    K_c = 0.822
    P_eddy = K_e*(V_s+V_s1)*(B_delta_expanded*frequency)**2; # eddy loss
    P_eddy_s = K_e*(V_t)*(B_delta_expanded*frequency)**2; # eddy loss

    # hysteresis loss
    K_h = 100
    n_h = 2
    P_h = K_h*(V_s+V_s1)*frequency*B_delta_expanded**n_h

    # stress loss
    P_stress = 0.01*P0

    # windage & friction loss
    k_r = 4 # roughness coefficient
    fr = 3e-3 # friction coefficient
    rho_air = 1.225 # air density kg/m^3
    # D2 = self.declare_variable('rotor_radius')
    
    # l_ef_expanded = csdl.expand(l_ef, (num_nodes,))
    # D2_expanded = csdl.expand(D2, (num_nodes,))
    l_ef_expanded = l_ef
    D2_expanded = D2
    P_wo = k_r*np.pi*fr*rho_air*(2*np.pi*frequency)**2*l_ef_expanded*D2_expanded**4
    Pm = 100

    # total losses
    P_em_loss = P_eddy + P_h + P_stress + P_wo + Pm + P_eddy_s
    P_loss = copper_loss + P_em_loss
    input_power_active = P0 + P_loss
    efficiency_active = P0/input_power_active
    
    torque_equality = load_torque - efficiency_active*T_em
    # torque_equality = T_em - load_torque - P_em_loss/omega
    residual = torque_equality

    solver = csdl.nonlinear_solvers.BracketedSearch('em_torque_bs')
    solver.add_state(state, residual, (0., upper_T_lim))
    solver.run()
    # endregion

    outputs = {}
    if mode == 'input_load':
        outputs['T_em'] = T_em
    elif mode == 'efficiency_map':
        outputs['load_torque'] = load_torque

    outputs = {
        'efficiency': efficiency_active,
        'input_power': input_power_active,
        'output_power': output_power,
        'T_em': T_em,
        'load_torque': load_torque
    }

    return outputs