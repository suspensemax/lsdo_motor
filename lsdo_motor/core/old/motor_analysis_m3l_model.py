import m3l
import numpy as np
from lsdo_motor.core.TC1_motor_analysis_model import TC1MotorAnalysisModel
from dataclasses import dataclass
from typing import List


@dataclass
class MotorAnalysisOutputs:
    """Data class for motor analysis outputs of low-fi model"""
    input_power : m3l.Variable
    efficiency : m3l.Variable
    torque_delta : m3l.Variable

def_coeff_H = np.array([1.92052530e-03,  1.03633835e+01, -2.98809161e+00])
def_coeff_B = np.array([1.12832651, 1., 1., 1.])

class MotorAnalysis(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        self.parameters.declare('pole_pairs', default=6)
        self.parameters.declare('phases', default=3)
        self.parameters.declare('num_slots', default=36)
        self.parameters.declare('rated_current', default=120.)
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('V_lim', default=600)
        self.parameters.declare('fit_coeff_dep_H', default=def_coeff_H) # FITTING COEFFICIENTS (X = H, B = f(H))
        self.parameters.declare('fit_coeff_dep_B', default=def_coeff_B) # FITTING COEFFICIENTS (X = B, H = g(B))
        self.parameters.declare('gear_ratio', default=4.)
        self.parameters.declare('flux_weakening', types=bool, default=False)

    def assign_attributes(self):
        self.name = self.parameters['name']

    def compute(self):
        model = TC1MotorAnalysisModel(
            pole_pairs=self.pole_pairs,
            phases=self.phases,
            num_slots=self.num_slots,
            num_nodes=self.num_nodes,
            rated_current=self.I_rated,
            V_lim=self.V_lim,
            fit_coeff_dep_H=self.fit_coeff_dep_H,
            fit_coeff_dep_B=self.fit_coeff_dep_B,
            gear_ratio=self.gear_ratio,
            flux_weakening=self.flux_weakening,
        )
        return model
    
    def evaluate(self, 
                 torque: m3l.Variable,
                 motor_parameters: m3l.Variable,
                 rpm : m3l.Variable,
                 motor_diameter : m3l.Variable) -> m3l.Variable:
        
        self.pole_pairs = self.parameters['pole_pairs']
        self.phases = self.parameters['phases']
        self.num_slots = self.parameters['num_slots']
        self.I_rated = self.parameters['rated_current']
        self.num_nodes = self.parameters['num_nodes']
        self.V_lim = self.parameters['V_lim']
        self.fit_coeff_dep_H = self.parameters['fit_coeff_dep_H']
        self.fit_coeff_dep_B = self.parameters['fit_coeff_dep_B']
        self.gear_ratio = self.parameters['gear_ratio']
        self.flux_weakening = self.parameters['flux_weakening']

        self.arguments={}
        self.arguments['load_torque_rotor'] = torque
        self.arguments['motor_parameters'] = motor_parameters
        self.arguments['rpm'] = rpm
        self.arguments['motor_diameter'] = motor_diameter

        input_power = m3l.Variable(
            name='input_power',
            shape=(self.num_nodes,),
            operation=self
        )
        efficiency = m3l.Variable(
            name='efficiency_active',
            shape=(self.num_nodes,),
            operation=self
        )

        torque_delta = m3l.Variable(
            name='torque_delta',
            shape=(self.num_nodes, ),
            operation=self,
        )

        outputs = MotorAnalysisOutputs(input_power=input_power, efficiency=efficiency, torque_delta=torque_delta)

        return outputs #input_power, efficiency
    

def evaluate_multiple_motor_analysis_models(
        rotor_outputs_list: list,
        motor_sizing_list : list,
        rotor_rpm_list : List[m3l.Variable],
        motor_diameter_list : List[m3l.Variable],
        name_prefix : str,
        m3l_model : m3l.Model=None,
        flux_weakening : bool=False,
        torque_delta : float=None,
) -> List[MotorAnalysisOutputs]: 
    """helper function for evaluating multiple motor analysis models"""

    if len(rotor_outputs_list) != len(motor_sizing_list):
        print(len(rotor_outputs_list))
        print(len(motor_sizing_list))
        raise Exception("Number of rotor ouputs not equal to number of motor parameters")
    
    num_models = len(rotor_outputs_list)
    motor_outputs_list = []

    for i in range(num_models):
        rotor_outputs = rotor_outputs_list[i]
        torque = rotor_outputs.Q
        rpm = rotor_rpm_list[i]
        d = motor_diameter_list[i]
        motor_parameters = motor_sizing_list[i].motor_parameters
    

        motor_model = MotorAnalysis(
            name=f'{name_prefix}_{i}',
            flux_weakening=flux_weakening,
        )
        motor_outputs = motor_model.evaluate(torque=torque, motor_parameters=motor_parameters, rpm=rpm, motor_diameter=d)
        motor_outputs_list.append(motor_outputs)

        if m3l_model is not None:
            m3l_model.register_output(motor_outputs)
            if torque_delta is not None:
                m3l_model.add_constraint(motor_outputs.torque_delta, lower=torque_delta)

    return motor_outputs_list