import m3l
import numpy as np
from lsdo_motor.core.TC1_motor_sizing_model import TC1MotorSizingModel
from dataclasses import dataclass
from typing import List


@dataclass
class MotorSizingOutputs:
    """Data class for low-fidelity motor sizing varaibles"""
    mass : m3l.Variable
    cg_vector : m3l.Variable
    inertia_tensor : m3l.Variable
    motor_parameters : m3l.Variable


class MotorSizing(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        self.parameters.declare('pole_pairs', default=6)
        self.parameters.declare('phases', default=3)
        self.parameters.declare('num_slots', default=36)
        self.parameters.declare('rated_current', default=120.)

    def assign_attributes(self):
        self.name = f"{self.parameters['name']}"

    def compute(self):
        model = TC1MotorSizingModel(
            pole_pairs=self.pole_pairs,
            phases=self.phases,
            num_slots=self.num_slots,
            rated_current=self.I_rated,
        )
        return model
    
    def evaluate(self, motor_diameter, motor_length, motor_origin) -> MotorSizingOutputs:
        self.pole_pairs = self.parameters['pole_pairs']
        self.phases = self.parameters['phases']
        self.num_slots = self.parameters['num_slots']
        self.I_rated = self.parameters['rated_current']

        self.arguments = {}
        self.arguments['motor_origin'] = motor_origin
        self.arguments['motor_diameter'] = motor_diameter
        self.arguments['motor_length'] = motor_length

        motor_mass = m3l.Variable(
            name='mass',
            shape=(1,),
            operation=self
        )

        motor_cg = m3l.Variable(
            name='cg_vector',
            shape=(3,),
            operation=self
        )

        motor_inertia = m3l.Variable(
            name='inertia_tensor',
            shape=(3, 3),
            operation=self
        )

        motor_parameters = m3l.Variable(
            name='motor_parameters',
            shape=(27,),
            operation=self
        ) # motor_parameters includes resistance and max torque

        outputs = MotorSizingOutputs(
            mass=motor_mass,
            cg_vector=motor_cg,
            inertia_tensor=motor_inertia,
            motor_parameters=motor_parameters,
        )


        return outputs # motor_mass, motor_cg, motor_inertia, motor_parameters
    
    # NOTE: # ADD PREFIX TO NAMES OF INPUTS AND OUTPUTS


def evaluate_multiple_motor_sizing_models(
    motor_diameter_list : List[m3l.Variable],
    motor_length_list : List[m3l.Variable],
    motor_origin_list : List[m3l.Variable],
    name_prefix : str,
    m3l_model : m3l.Model=None,
) -> List[MotorSizingOutputs]:
    """Helper function to evaluate multiple motor sizing models"""
    if len(motor_diameter_list) != len(motor_length_list):
        raise Exception("Number of motor diameters not equal to number of motor_lengths")
    else:
        num_models = len(motor_length_list)

    print(motor_origin_list)
    if len(motor_origin_list) != num_models:
        raise Exception("Number of specified motor origins not equal to number of motor diameters/motor lengths")
    
    motor_sizing_outputs_list = []
    for i in range(num_models):
        d = motor_diameter_list[i]
        l = motor_length_list[i]
        origin = motor_origin_list[i]
        motor_sizing_model = MotorSizing(
            name=f"{name_prefix}_{i}",
        )

        motor_sizing_outputs = motor_sizing_model.evaluate(motor_diameter=d, motor_length=l, motor_origin=origin)
        motor_sizing_outputs_list.append(motor_sizing_outputs)
        
        if m3l_model is not None:
            m3l_model.register_output(motor_sizing_outputs)

    return motor_sizing_outputs_list




