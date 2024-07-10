import numpy as np
import csdl
from python_csdl_backend import Simulator

from lsdo_motor import FullMotorModel # FULL MOTOR MODEL WITH SIZING AND ANALYSIS

class MultipointMotorModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('max_rotor_torque')
        self.parameters.declare('max_rotor_RPM')
        self.parameters.declare('nominal_rotor_torque')
        self.parameters.declare('nominal_rotor_RPM')
        self.parameters.declare('gear_ratio')
        self.parameters.declare('diameter_dv_lower_bound', default=0.06)
        self.parameters.declare('length_dv_lower_bound', default=0.06)
        self.parameters.declare('torque_delta_ratio_bound', default=0.)

    def define(self):
        max_rotor_torque = self.parameters['max_rotor_torque']
        max_rotor_RPM = self.parameters['max_rotor_RPM']
        nom_rotor_torque = self.parameters['nominal_rotor_torque']
        nom_rotor_RPM = self.parameters['nominal_rotor_RPM']
        gear_ratio = self.parameters['gear_ratio']

        D_lower_bound = self.parameters['diameter_dv_lower_bound']
        L_lower_bound = self.parameters['length_dv_lower_bound']
        torque_delta_ratio_bound = self.parameters['torque_delta_ratio_bound']

        D = self.create_input('motor_diameter')
        L = self.create_input('motor_length')

        rotor_torque = np.array([nom_rotor_torque, max_rotor_torque])
        rotor_RPM = np.array([nom_rotor_RPM, max_rotor_RPM])
        num_nodes = 2

        self.create_input('load_torque_rotor', val=rotor_torque, shape=(num_nodes,))
        self.create_input('rpm', val=rotor_RPM, shape=(num_nodes,))

        motor_model = self.add(
            FullMotorModel(
                num_nodes=num_nodes,
                V_lim=400,
                gear_ratio=gear_ratio
            ),
            name='full_motor_model',
        )
        motor_mass = self.declare_variable('motor_mass')
        efficiency_out = self.declare_variable('efficiency', shape=(num_nodes,))
        torque_delta_out = self.declare_variable('torque_delta', shape=(num_nodes,))

        nom_efficiency = self.register_output('nominal_efficiency', efficiency_out[0]*1.)
        max_torque_delta = self.register_output('max_torque_delta', torque_delta_out[1]*1.)

        self.add_design_variable('motor_diameter', lower=D_lower_bound)
        self.add_design_variable('motor_length', lower=L_lower_bound)

        self.add_constraint('max_torque_delta', lower=torque_delta_ratio_bound)
        self.add_objective('nominal_efficiency', scaler=-1.) # min -efficiency = max efficiency
        # self.add_objective('motor_mass')

'''
This file runs a sample motor optimization with the low-fidelity motor model
developed for TC1 of the NASA ULI project. This example uses SLSQP from the scipy 
library.

We optimize the motor under 2 conditions:
    - the motor is SIZED by the maximum torque operating condition
    - the efficiency is maximized at the nominal operating condition

==================== Inputs: ====================
- motor_diameter (m)
    - inner diameter of the motor stator
    - in our empirical sizing, inner stator diameter = outer diameter/1.25

- motor_length (m)
    - motor length

- rpm (RPM)
    - rotor RPM

- load_torque_rotor (Nm)
    - rotor torque

==================== Outputs of interest: ====================
- torque_delta ((Nm)/(Nm)) 
    - ratio defined as (max_torque - operating_torque)/max_torque

- efficiency (W/W)
    - motor efficiency, defined as (motor output power) / (motor input power)

- motor_mass (kg)
    - mass of motor from sizing model
'''

# region setting up inputs/parameters
D_i = 304.8e-3
L = 165.1e-3

# Data for max operating condition
max_rotor_torque = 1200
max_rotor_RPM = 1193.7

# Data for nominal operating condition
nominal_rotor_torque = 286
nominal_rotor_RPM = 4000

gear_ratio = 1.
# endregion

# region motor model
m = MultipointMotorModel(
    max_rotor_torque=max_rotor_torque,
    max_rotor_RPM=max_rotor_RPM,
    nominal_rotor_torque=nominal_rotor_torque,
    nominal_rotor_RPM=nominal_rotor_RPM,
    gear_ratio=gear_ratio,
    # diameter_dv_lower_bound=, # OPTIONAL INPUT
    # length_dv_lower_bound=, # OPTIONAL INPUT
    # torque_delta_ratio_bound=, # OPTIONAL INPUT
)
# endregion

# setting up Simulator and initial values
sim = Simulator(m)
sim['motor_diameter'] = D_i
sim['motor_length'] = L

# forward evaluation (not necessary)
sim.run()

# optimization
# from modopt.csdl_library import CSDLProblem
# from modopt.scipy_library import SLSQP
# prob = CSDLProblem(problem_name='motor_opt_test', simulator=sim)
# optimizer = SLSQP(prob, maxiter=500, ftol=1E-5)

# optimizer.solve()
# optimizer.print_results()

# printing relevant outputs
print('Optimized diameter (m): ', sim['motor_diameter'])
print('Optimized length (m): ', sim['motor_length'])
print('Optimized mass (kg): ', sim['motor_mass'])
print('Optimized efficiency at nominal operation (W/W): ', sim['nominal_efficiency'])
print('Max torque delta for optimized motor ((Nm)/(Nm)): ', sim['max_torque_delta'])
print('Nominal input power (kW): ', sim['input_power'][0] / 1000.)
print('Maximum input power (kW): ', sim['input_power'][1] / 1000.)

print('Maximum allowed torque: ', sim['T_upper_lim_curve'][0])

# add print statements to clearly identify outputs (torque delta, power, etc.)