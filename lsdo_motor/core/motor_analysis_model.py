import numpy as np 
import csdl_alpha as csdl

from lsdo_motor.core.submodels.magnet_mec_model import magnet_mec_model
from lsdo_motor.core.submodels.inductance_model import inductance_model
from lsdo_motor.core.submodels.torque_limit_model import torque_limit_model
from lsdo_motor.core.submodels.flux_weakening_bracket_model import flux_weakening_bracket_model
from lsdo_motor.core.submodels.em_torque_model import em_torque_model

from typing import Literal
mode_types = Literal['input_load', 'efficiency_map']

def motor_analysis_model(sizing_inputs, motor_inputs, parameters, fit_coeff_dep_H, fit_coeff_dep_B,  
                         mode: mode_types='input_load', mtpa=True, flux_weakening=False):

    # region inputs
    outer_stator_radius = sizing_inputs.outer_stator_radius
    D_i = sizing_inputs.D_i
    pole_pitch = sizing_inputs.pole_pitch
    tooth_pitch = sizing_inputs.tooth_pitch
    air_gap_depth = sizing_inputs.air_gap_depth
    l_ef = sizing_inputs.l_ef
    rotor_radius = sizing_inputs.rotor_radius
    turns_per_phase = sizing_inputs.turns_per_phase
    Acu = sizing_inputs.Acu
    tooth_width = sizing_inputs.tooth_width
    h_ys = sizing_inputs.h_ys
    b_sb = sizing_inputs.b_sb
    h_slot = sizing_inputs.h_slot
    b_s1 = sizing_inputs.b_s1
    Tau_y = sizing_inputs.Tau_y
    L_j1 = sizing_inputs.L_j1
    Kdp1 = sizing_inputs.Kdp1
    bm = sizing_inputs.bm
    Am_r = sizing_inputs.Am_r
    phi_r = sizing_inputs.phi_r
    lambda_m = sizing_inputs.lambda_m
    alpha_i = sizing_inputs.alpha_i
    Kf = sizing_inputs.Kf
    K_phi = sizing_inputs.K_phi
    K_theta = sizing_inputs.K_theta
    A_f2 = sizing_inputs.A_f2
    Rdc = sizing_inputs.Rdc

    num_nodes = parameters['num_nodes']
    V_lim = parameters['V_lim']
    p = parameters['pole_pairs']
    m = parameters['phases']
    Z = parameters['num_slots']
    I_w_r = parameters['rated_current']
    omega_rated = parameters['rated_omega']

    # endregion

    # region magnet mec model
    magnet_mec_inputs = {
        'tooth_pitch': tooth_pitch,
        'tooth_width': tooth_width,
        'slot_height': h_slot,
        'alpha_i': alpha_i,
        'pole_pitch': pole_pitch,
        'l_ef': l_ef,
        'height_yoke_stator': h_ys,
        'L_j1': L_j1,
        'air_gap_depth': air_gap_depth,
        'K_theta': K_theta,
        'A_f2': A_f2,
        'bm': bm,
        'phi_r': phi_r,
        'lambda_m': lambda_m,
        'Am_r': Am_r,
    }
    magnet_mec_outputs = magnet_mec_model(
        num_nodes=num_nodes,
        inputs=magnet_mec_inputs,
        fit_coeff_dep_H=fit_coeff_dep_H,
        fit_coeff_dep_B=fit_coeff_dep_B
    )
    # endregion

    # region inductance model
    inductance_inputs = {
        'Kf': Kf,
        'phi_air': magnet_mec_outputs['phi_air'],
        'alpha_i': alpha_i,
        'pole_pitch': pole_pitch,
        'l_ef': l_ef,
        'K_theta': K_theta,
        'air_gap_depth': air_gap_depth,
        'tooth_pitch': tooth_pitch,
        'tooth_width': tooth_width,
        'h_slot': h_slot,
        'h_ys': h_ys,
        'Kdp1': Kdp1,
        'turns_per_phase': turns_per_phase,
        'L_j1': L_j1,
        'F_total': magnet_mec_outputs['F_total'],
        'F_delta': magnet_mec_outputs['F_delta'],
        'b_sb': b_sb,
        'b_s1': b_s1,
        'Tau_y': Tau_y,
        'K_sigma_air': magnet_mec_outputs['K_sigma_air'],
        'lambda_n': magnet_mec_outputs['lambda_n'],
        'lambda_leak_standard': magnet_mec_outputs['lambda_leak_standard'],
        'Am_r': Am_r,
        'K_phi': K_phi,
    }
    inductance_outputs = inductance_model(
        inputs=inductance_inputs,
        fit_coeff_dep_B=fit_coeff_dep_B,
        num_nodes=num_nodes,
        p=p,
        m=m,
        Z=Z,
        I_w_r=I_w_r,
        omega_rated=omega_rated
    )
    # endregion

    phi_air = magnet_mec_outputs['phi_air']
    W_1 = turns_per_phase
    PsiF = W_1 * phi_air

    Ld = inductance_outputs['Ld']
    Lq = inductance_outputs['Lq']

    # R_expanded = csdl.expand(Rdc, (num_nodes,))
    # Ld_expanded = csdl.expand(Ld, (num_nodes,))
    # Lq_expanded = csdl.expand(Lq, (num_nodes,))
    # PsiF_expanded = csdl.expand(PsiF, (num_nodes,))

    R_expanded = Rdc
    Ld_expanded = Ld
    Lq_expanded = Lq
    PsiF_expanded = PsiF

    gear_ratio = parameters['gear_ratio']
    omega = motor_inputs.rotor_rpm*gear_ratio*2*np.pi/60*p
    load_torque = motor_inputs.rotor_torque/gear_ratio

    torque_limit_inputs = {
        'R_expanded': R_expanded,
        'Ld_expanded': Ld_expanded,
        'Lq_expanded': Lq_expanded,
        'PsiF_expanded': PsiF_expanded,
        'omega': omega
    }
    T_lim = torque_limit_model(
        inputs=torque_limit_inputs,
        num_nodes=num_nodes,
        V_lim=V_lim,
        p=p,
    )

    if flux_weakening:
        _ = flux_weakening_bracket_model()

    em_torque_inputs = {
        'R': R_expanded,
        'Ld': Ld_expanded,
        'Lq': Lq_expanded,
        'PsiF': PsiF_expanded,
        'omega': omega,
        'Iq_rated': parameters['rated_current'],
        'D_i': D_i,
        'B_delta': magnet_mec_outputs['B_delta'],
        'l_ef': l_ef,
        'outer_stator_radius': outer_stator_radius,
        'rotor_radius': rotor_radius,
        'bm': bm,
        'Acu': Acu,
        'upper_T_lim': T_lim,
    }
    em_torque_parameters = {}
    em_torque_mode = mode
    
    if em_torque_mode == 'input_load':
        em_torque_inputs['load_torque'] = load_torque
    elif em_torque_mode == 'efficiency_map':
        em_torque_inputs['T_em'] = T_em
    output_dict = em_torque_model(
        inputs=em_torque_inputs,
        parameters=parameters,
        mode=em_torque_mode,
        mtpa=mtpa, 
        flux_weakening=flux_weakening
    )

    return output_dict