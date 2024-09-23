import numpy as np
import csdl_alpha as csdl
from dataclasses import dataclass
from csdl_alpha.utils.typing import VariableLike

from lsdo_motor.core.motor_sizing_model import motor_sizing_model
from lsdo_motor.core.motor_ecm_parameters_model import motor_ecm_parameters_model
from lsdo_motor.core.motor_analysis_model import motor_analysis_model

# motor inputs
@dataclass
class MotorInputVariableGroup(csdl.VariableGroup):
    rotor_rpm: VariableLike
    rotor_torque: VariableLike

# custom sizing inputs
@dataclass
class CustomSizingVariableGroup(csdl.VariableGroup): 
    # NOTE: UPDATE LIST TO ONLY ACCOUNT FOR HIGH-LEVEL GEOMETRY
    L: VariableLike # motor length (m)
    D: VariableLike # outer diameter (m)
    mass: VariableLike # mass (kg)
    peak_torque: VariableLike # max torque (Nm)
    D_i: VariableLike # # inner stator diameter (m)
    D_shaft: VariableLike # shaft diameter (m)
    pole_pitch: VariableLike # arc length of one magnet pole on the rotor surface (m)
    tooth_pitch: VariableLike # arc length of one tooth on the rotor surface (m)
    air_gap_depth: VariableLike # air gap size (m)
    rotor_diameter: VariableLike # rotor diameter
    turns_per_phase: VariableLike # turns per stator slot
    tooth_width: VariableLike # stator tooth arc length
    h_ys: VariableLike # stator yoke thickness (m) -> hj1 in diagram
    b_sb: VariableLike # width of slot opening between two teeth (m)
    h_slot: VariableLike # height of slot (m)
    b_s1: VariableLike # max thickness of slot (m)
    magnet_thickness: VariableLike # magnet thickness (m)
    magnet_embed_depth: VariableLike # magnet embedded depth 
    Acu: VariableLike # cross-sectional area of coils in stator windings (m^2)

# motor outputs
@dataclass
class MotorOutputVariableGroup(csdl.VariableGroup):
    efficiency: csdl.Variable
    input_power: csdl.Variable
    output_power: csdl.Variable
    T_em: csdl.Variable
    load_torque: csdl.Variable
    mass: csdl.Variable
    L: csdl.Variable
    peak_torque: csdl.Variable

from typing import Literal
mode_types = Literal['input_load', 'efficiency_map']

class MotorModel():
    def __init__(self, pole_pairs=6, phases=3, num_slots=36, V_lim=400., rated_current=120., gear_ratio=4,
                 A_bar=30.e3, B_bar=0.85, mode: mode_types='input_load', mtpa=True, flux_weakening=False):
        self.pole_pairs = pole_pairs
        self.phases = phases
        self.num_slots = num_slots
        self.V_lim = V_lim
        self.rated_current = rated_current
        self.gear_ratio = gear_ratio

        self.A_bar = A_bar
        self.B_bar = B_bar

        self.mode = mode 
        self.mtpa = mtpa
        self.flux_weakening = flux_weakening

        self.nl_sizing = False

    def toggle_nonlinear_sizing(self, max_iter=10, tolerance=1.e-6):
        self.nl_sizing =  True
        self.nl_solver_parameters = {
            'max_iter': max_iter,
            'tol': tolerance
        }

    def evaluate(
            self,
            motor_inputs: MotorInputVariableGroup,
            custom_sizing_var_group=False, 
            L_0: VariableLike=0.05,
            L_D=0.5,
            torque_delta=0.1,
            permeability_data=None,
            sizing_mode='torque',
            power_density=7., # kw/kg
            torque_density=15.,
        ):

        rotor_torque = motor_inputs.rotor_torque
        rotor_rpm = motor_inputs.rotor_rpm
        rotor_power = rotor_torque*rotor_rpm*2.*np.pi/60.

        num_nodes = rotor_torque.shape[0]

        if permeability_data is None:
            # compute permeability coefficients separately here
            def_coeff_H = np.array([1.92052530e-03,  1.03633835e+01, -2.98809161e+00])
            def_coeff_B = np.array([1.12832651, 1., 1., 1.]) 
        else:
            raise NotImplementedError('Custom permeability models not implemented yet.')
        
        parameters = {
            'num_nodes': num_nodes,
            'pole_pairs': self.pole_pairs,
            'phases': self.phases,
            'num_slots': self.num_slots,
            'V_lim': self.V_lim,
            'rated_current': self.rated_current,
            'gear_ratio': self.gear_ratio,
            'rated_omega': 5000.
        }

        # if self.nl_sizing:
        #     L = csdl.ImplicitVariable(value=L_0)
        # else:
        #     L = csdl.Variable(value=L_0)
        if type(L_0) is csdl.Variable:
            if L_0.shape != (num_nodes,):
                L_0 = csdl.reshape(L_0, shape=(num_nodes,))
            L = L_0
        else:
            L = csdl.Variable(value=L_0, shape=(num_nodes,))

        motor_omega = rotor_rpm*2*np.pi/60.*self.gear_ratio

        if custom_sizing_var_group:
            L = custom_sizing_var_group.L
            motor_sizing_variable_group = custom_sizing_var_group

        else: # using default sizing method
            motor_sizing_variable_group = motor_sizing_model(
                L=L,
                parameters=parameters,
                omega=motor_omega,
                sizing_mode=sizing_mode,
                torque_density=torque_density,
                power_density=power_density,
                A_bar=self.A_bar,
                B_bar=self.B_bar,
                L_D=L_D,
            )

        mass = motor_sizing_variable_group.mass
        peak_torque = motor_sizing_variable_group.peak_torque

        ecm_param_variable_group = motor_ecm_parameters_model(
            parameters=parameters,
            sizing_inputs=motor_sizing_variable_group,
        )

        output_dict = motor_analysis_model(
            sizing_inputs=motor_sizing_variable_group,
            ecm_inputs=ecm_param_variable_group,
            motor_inputs=motor_inputs,
            parameters=parameters,
            fit_coeff_dep_H=def_coeff_H,
            fit_coeff_dep_B=def_coeff_B,
            mode=self.mode,
            mtpa=self.mtpa,
            flux_weakening=self.flux_weakening
        )

        efficiency = output_dict['efficiency']
        input_power = output_dict['input_power']
        output_power = output_dict['output_power']
        T_em = output_dict['T_em']
        load_torque = output_dict['load_torque']

        if self.nl_sizing:
            max_iter = self.nl_solver_parameters['max_iter']
            tol = self.nl_solver_parameters['tol']
            residual = (peak_torque - T_em)/T_em - torque_delta
            solver = csdl.nonlinear_solvers.Newton('torque_delta_ns', max_iter=max_iter)
            solver.add_state(L, residual, tolerance=tol)
            # solver = csdl.nonlinear_solvers.BracketedSearch('torque_delta_bs', max_iter=10)
            # solver.add_state(L, residual, (.01, .5), tolerance=1.e-4)
            solver.run()

        motor_output_vg = MotorOutputVariableGroup(
            efficiency=efficiency,
            input_power=input_power,
            output_power=output_power,
            T_em=T_em,
            load_torque=load_torque,
            mass=mass,
            L=L,
            peak_torque=peak_torque
        )

        return motor_output_vg
    
'''
List of models:
- sizing model (DONE)

- inductance MEC model (DONE)
- magnet MEC model (DONE)
- torque limit model 
    - discrete check custom operation
    - finds upper limit for EM torque; case for quartic where discriminant = 0
    - NOTE: first version done, check again later
- flux weakening bracket model (SKIP FOR NOW)
    - sets up brackets for flux weakening bracketed search

- EM torque model
    - flux weakening model (SKIP FOR NOW)
    - mtpa model (DONE)
    - efficiency computation (DONE)
'''