import numpy as np
import csdl_alpha as csdl

from lsdo_motor import MotorModel, MotorInputVariableGroup, CustomSizingVariableGroup

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

# custom motor sizing variable group
custom_sizing_vg = CustomSizingVariableGroup(
    L=0.131, # motor length (m)
    D=0.393, # stator outer diameter (m)
    mass=54.0287842, # motor mass (kg)
    peak_torque=1652.87128314, # max torque (Nm)
    D_i=0.3144, # inner stator diameter (m)
    D_shaft=0.09354532, # shaft diameter (m)
    pole_pitch=0.08230973, # arc length of one magnet pole on the rotor surface (m)
    tooth_pitch=0.02743658, # arc length of one tooth on the rotor surface (m)
    air_gap_depth=0.00129113, # air gap size (m)
    rotor_diameter=0.31181773, # rotor diameter
    turns_per_phase=41.15486376, # turns per stator slot
    tooth_width=0.0144403, # stator tooth arc length
    h_ys=0.01909329, # stator yoke thickness (m) -> hj1 in diagram
    b_sb=0.00411549, # width of slot opening between two teeth (m)
    h_slot=0.02020671, # height of slot (m)
    b_s1=0.01334534, # max thickness of slot (m)
    magnet_thickness=0.004, # magnet thickness (m)
    magnet_embed_depth=0.002, # magnet embedded depth 
    Acu=1.2e-05 # cross-sectional area of coils in stator windings (m^2)
)

# instantiating MotorModel class; no need to change any inputs except the gear ratio
motor_model = MotorModel(
    pole_pairs=6,
    phases=3,
    num_slots=36,
    V_lim=400,
    gear_ratio=4
)

# ==== evaluating motor model w/ output variable group ====
output_vg = motor_model.evaluate(
    motor_inputs=input_vg,
    custom_sizing_var_group=custom_sizing_vg,
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

print(mass)
print(efficiency.value)
print(output_power.value)
print(input_power.value)
print(peak_torque)
print(T_em.value)
print(L)

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