import numpy as np
import csdl
from python_csdl_backend import Simulator

from lsdo_motor import FullMotorModel # FULL MOTOR MODEL WITH SIZING AND ANALYSIS

'''
This file runs a sample motor optimization with the low-fidelity motor model
developed for TC1 of the NASA ULI project. This example uses SLSQP from the scipy 
library.

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

- input_power (W)
    - input power, defined as (torque * rotational speed) + power losses

- output_power (W)
    - motor output power, defined as torque * rotational speed

- efficiency (W/W)
    - motor efficiency, defined as (motor output power) / (motor input power)

- motor_mass (kg)
    - mass of motor from sizing model
'''

# region setting up inputs/parameters
D_i = 0.11
L = 0.08
load_torque_rotor = np.array([172.317])
omega_rotor = np.array([3375.10])
num_nodes=len(load_torque_rotor)
gear_ratio = 4.
# endregion

# region motor model
m = FullMotorModel(
    num_nodes=num_nodes,
    V_lim=400,
    gear_ratio=gear_ratio
)
# endregion

# region setting inputs/constraints/design variables/objective
m.create_input('motor_diameter')
m.create_input('motor_length')

m.add_constraint('efficiency', lower=0.90)
m.add_constraint('torque_delta', lower=.3) 

m.add_design_variable('motor_diameter', lower = 0.1)
m.add_design_variable('motor_length', lower = 0.06)

m.add_objective('motor_mass')
# endregion


# setting up Simulator and initial values
sim = Simulator(m)
sim['motor_diameter'] = D_i
sim['motor_length'] = L
sim['rpm'] = omega_rotor
sim['load_torque_rotor'] = load_torque_rotor

# forward evaluation (not necessary)
sim.run()
print(sim['motor_mass']) # syntax for printing outputs

# optimization
from modopt.csdl_library import CSDLProblem
from modopt.scipy_library import SLSQP
prob = CSDLProblem(problem_name='motor_opt_test', simulator=sim)
optimizer = SLSQP(prob, maxiter=500, ftol=1E-5)

optimizer.solve()
optimizer.print_results()
