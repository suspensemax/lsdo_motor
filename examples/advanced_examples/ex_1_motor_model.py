import numpy as np
import csdl_alpha as csdl

from lsdo_motor import MotorModel, MotorInputVariableGroup

# region input rotor torque and rpm set
# ==== demo case ====
num_nodes = 1
rotor_torque = np.ones((num_nodes,))*172.317
rotor_rpm = np.ones((num_nodes,))*3375.10
# rotor_rpm = np.ones((num_nodes,))*6500

# ==== testing changing numbers across num_nodes ====
# rotor_torque = np.array([1200, 286])
# rotor_rpm = np.array([1193.7, 4000])

# ==== a 'high' torque case from a past lsdo_rotor test case ====
rotor_torque = np.array([1200])
rotor_rpm = np.array([1193.7])
# endregion

# instantiating MotorModel class; no need to change any inputs except the gear ratio
motor_model = MotorModel()

recorder = csdl.Recorder(inline=True)
recorder.start()
rotor_torque = csdl.Variable(value=rotor_torque)
rotor_rpm = csdl.Variable(value=rotor_rpm)

# motor input variable group
# NOTE: the two inputs are rotor torque and rotor rpm
input_vg = MotorInputVariableGroup(
    rotor_torque=rotor_torque,
    rotor_rpm=rotor_rpm
)
# method to call non-linear sizing; comment out to deactivate
motor_model.toggle_nonlinear_sizing(
    max_iter=10,
    tolerance=1.e-6
)
# ==== evaluating motor model w/ output variable group ====
output_vg = motor_model.evaluate(
    motor_inputs=input_vg,
    # L_0=csdl.Variable(value=0.1), # input for L as a csdl Variable
    L_0=.131, # input for L as a float
    L_D=1./3.,
    torque_delta=0.1,
    sizing_mode='torque', # can change between torque and power
    torque_density=15.,
    # sizing_mode='power',
    # power_density=7. #kW/kg
)

mass = output_vg.mass
efficiency = output_vg.efficiency
output_power = output_vg.output_power
input_power = output_vg.input_power
peak_torque = output_vg.peak_torque
T_em = output_vg.T_em
L = output_vg.L
# csdl.derivative_utils.verify_derivatives(ofs=efficiency, wrts=rotor_rpm, step_size=1.e-6)

recorder.stop()

'''
====
'''
# ==== running with the jax simulator ====
# jax_sim =  csdl.experimental.JaxSimulator(
#     recorder=recorder,
#     additional_inputs=[rotor_torque, rotor_rpm],
#     additional_outputs=[mass, efficiency, output_power, input_power]
# )
# jax_sim.run()

# mass = jax_sim[mass]
# efficiency = jax_sim[efficiency]
# output_power = jax_sim[output_power]
# input_power = jax_sim[input_power]