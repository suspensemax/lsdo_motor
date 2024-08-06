import numpy as np
import caddee.api as cd
import lsdo_geo as lg
import m3l
from python_csdl_backend import Simulator
import array_mapper as am

from lsdo_rotor.core.BEM_caddee.BEM_caddee import BEM, BEMMesh

from lsdo_acoustics import GEOMETRY_PATH, IMPORTS_PATH

from lsdo_motor import MotorSizing, MotorAnalysis

'''
This test script connects the BEM to the motor model. We consider a normal hover case.
'''

caddee = cd.CADDEE()
caddee.system_representation = system_rep = cd.SystemRepresentation()
caddee.system_parameterization = system_param = cd.SystemParameterization(system_representation=system_rep)

file_name = GEOMETRY_PATH / 'single_rotor.stp'
spatial_rep = system_rep.spatial_representation
spatial_rep.import_file(file_name=file_name, file_path=str(IMPORTS_PATH))
spatial_rep.refit_geometry(file_name=file_name, file_path=str(IMPORTS_PATH))

'''
======================================== GEOMETRY ========================================
'''
# region geometry
# ==================== ROTOR DISK ====================
prefix = 'rotor'
rotor_disk_primitive_names = list(
    spatial_rep.get_primitives(search_names=['Rotor_disk']).keys()
)
rotor_disk = cd.Rotor(
    name=f'{prefix}_disk',
    spatial_representation=spatial_rep,
    primitive_names=rotor_disk_primitive_names
)
system_rep.add_component(rotor_disk)

# PROJECTIONS
p11 = rotor_disk.project(np.array([0.500, 0.000, 0.000]), direction=np.array([0.,0.,1.]), plot=False)
p12 = rotor_disk.project(np.array([-0.500, 0.000, 0.000]), direction=np.array([0.,0.,1.]), plot=False)
p21 = rotor_disk.project(np.array([0.000, 0.500 , 0.000]), direction=np.array([0.,0.,1.]), plot=False)
p22 = rotor_disk.project(np.array([0.000, -0.500, 0.000]), direction=np.array([0.,0.,1.]), plot=False)
rotor_in_plane_x = am.subtract(p11,p12)
rotor_in_plane_y = am.subtract(p21,p22)
rotor_origin = rotor_disk.project(np.array([0.000, 0.000, 0.000]), direction=np.array([0.,0.,1.]))
system_rep.add_output(f'{prefix}_disk_in_plane_1', quantity=rotor_in_plane_x)
system_rep.add_output(f'{prefix}_disk_in_plane_2', quantity=rotor_in_plane_y)
system_rep.add_output(f'{prefix}_disk_origin', quantity=rotor_origin)

# ==================== ROTOR BLADES ====================
rotor_blade_primitive_names = list(
    spatial_rep.get_primitives(search_names=['Rotor_blades, 0']).keys()
)
rotor_blade = cd.Rotor(
    name=f'{prefix}_blade',
    spatial_representation=spatial_rep,
    primitive_names=rotor_blade_primitive_names
)
system_rep.add_component(rotor_blade)

# PROJECTIONS
le_tip = np.array([-0.032, 0.5, 0.007])
le_root = np.array([-0.014, 0.1, 0.015])
te_tip = np.array([0.032, 0.5, -0.007])
te_root = np.array([0.014, 0.1, -0.015])

offset_x = 0.05
offset_y = 0.05
offset_z = 0.05
num_radial = 60
blade_le = np.linspace(
    le_root + np.array([-offset_x, -offset_y, offset_z]),
    le_tip + np.array([-offset_x, 2*offset_y, offset_z]),
    num_radial
) # array around LE with offsets
blade_te = np.linspace(
    te_root - np.array([-offset_x, offset_y, offset_z]),
    te_tip - np.array([-offset_x, -2*offset_y, offset_z]),
    num_radial
) # array around TE with offsets

p_le = rotor_blade.project(blade_le, direction=np.array([0., 0., -1.]), grid_search_n=50, plot=False) # projection onto LE of BLADE
p_te = rotor_blade.project(blade_te, direction=np.array([0., 0., 1.]), grid_search_n=50, plot=False) # projection onto TE of BLADE

rotor_blade_chord = am.subtract(p_le, p_te) # produces array map of vectors radially, from TE to LE

rotor_disk_le_proj = rotor_disk.project(p_le.evaluate(), direction=np.array([0., 0., -1.]), grid_search_n=50, plot=False) # LE projected onto disk
rotor_disk_te_proj = rotor_disk.project(p_te.evaluate(), direction=np.array([0., 0., 1.]), grid_search_n=50, plot=False) # TE projected onto disk

rotor_v_dist_le = am.subtract(p_le, rotor_disk_le_proj) # vertical distance from LE to disk
rotor_v_dist_te = am.subtract(p_te, rotor_disk_te_proj) # vertical distance from TE to disk
rotor_v_dist_tot = am.subtract(rotor_v_dist_le, rotor_v_dist_te) 

system_rep.add_output(f'{prefix}_blade_chord_length', rotor_blade_chord)
system_rep.add_output(f'{prefix}_blade_twist', rotor_v_dist_tot)
# endregion
'''
======================================== SYSTEM MODEL ========================================
'''
caddee.system_model = system_model = cd.SystemModel()
design_scenario = cd.DesignScenario(name='motor_hover_test')



# ==================== HOVER ====================
hover_model = m3l.Model()
hover_condition = cd.HoverCondition(name='hover')
hover_condition.atmosphere_model = cd.SimpleAtmosphereModel()

hover_condition.set_module_input(name='altitude', val=500)
hover_condition.set_module_input(name='observer_location', val=np.array([0, 0, 500]))
hover_condition.set_module_input(name='hover_time', val=120.)
ac_states = hover_condition.evaluate_ac_states()
hover_model.register_output(ac_states)

# MOTOR MASS SOLVER, NOT SURE WHERE THIS NEEDS TO GO

motor_sizing = MotorSizing(
    rotor_component=rotor_disk
)
motor_sizing.set_module_input('motor_diameter', 0.08, dv_flag=True, lower=0.12, upper=0.3, scaler=10.)
motor_sizing.set_module_input('motor_length', 0.05, dv_flag=True, lower=0.06, upper=0.1, scaler=100.)

motor_mass, motor_cg, motor_inertia, motor_parameters = motor_sizing.evaluate() 
hover_model.register_output(motor_mass) # this would then feed into some mass properties solver
hover_model.register_output(motor_parameters)

# region BEM
rotor_bem_mesh = BEMMesh(
    meshes=dict(
        rotor_in_plane_1=rotor_in_plane_x,
        rotor_in_plane_2=rotor_in_plane_y,
        rotor_origin=rotor_origin
    ),
    airfoil='NACA_4412',
    num_blades=4,
    chord_b_spline_rep=True,
    twist_b_spline_rep=True,
    num_radial=num_radial,
    num_tangential=30,
    mesh_units='ft',
    use_airfoil_ml=False,
)

bem_model = BEM(component=rotor_disk, mesh=rotor_bem_mesh, disk_prefix='rotor_disk', blade_prefix='rotor_blade')
bem_model.set_module_input('rpm', val=1350.)
F, M, dT, dQ, dD, CT, Q = bem_model.evaluate(ac_states=ac_states)
hover_model.register_output(dT)
hover_model.register_output(dD)
hover_model.register_output(CT)
hover_model.register_output(Q)
# endregion

motor_model = MotorAnalysis(
    component=rotor_disk
)

P_in, efficiency = motor_model.evaluate(
    torque=Q,
    motor_parameters=motor_parameters,
    design_condition=hover_condition
)
hover_model.register_output(P_in) # TOTAL INPUT POWER FOR THE MOTOR

hover_condition.add_m3l_model('hover_model', hover_model)
design_scenario.add_design_condition(hover_condition)

system_model.add_design_scenario(design_scenario=design_scenario)
caddee_csdl_model = caddee.assemble_csdl()

caddee_csdl_model.connect(
    'system_model.motor_hover_test.hover.hover.rotor_disk_motor_sizing_model.motor_diameter',
    'system_model.motor_hover_test.hover.hover.hover_rotor_disk_motor_analysis_model.motor_diameter'
)

caddee_csdl_model.connect(
    'system_model.motor_hover_test.hover.hover.rotor_disk_bem_model.rpm',
    'system_model.motor_hover_test.hover.hover.hover_rotor_disk_motor_analysis_model.rpm'
)

sim = Simulator(caddee_csdl_model, analytics=True)
sim.run()