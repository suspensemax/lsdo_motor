import numpy as np
import csdl
from python_csdl_backend import Simulator

from lsdo_motor import TC1MotorAnalysisModel, TC1MotorSizingModel

def_coeff_H = np.array([1.92052530e-03,  1.03633835e+01, -2.98809161e+00])
def_coeff_B = np.array([1.12832651, 1., 1., 1.])

class FullMotorModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('num_active_nodes', default=None)
        self.parameters.declare('pole_pairs', default=6)
        self.parameters.declare('phases', default=3)
        self.parameters.declare('num_slots', default=36)
        self.parameters.declare('V_lim', default=400)
        self.parameters.declare('rated_current', default=120.)
        self.parameters.declare('fit_coeff_dep_H', default=def_coeff_H) # FITTING COEFFICIENTS (X = H, B = f(H))
        self.parameters.declare('fit_coeff_dep_B', default=def_coeff_B) # FITTING COEFFICIENTS (X = B, H = g(B))

    def define(self):
        num_nodes = self.parameters['num_nodes']
        num_active_nodes = self.parameters['num_active_nodes']
        pole_pairs = self.parameters['pole_pairs']
        phases = self.parameters['phases']
        num_slots = self.parameters['num_slots']
        V_lim = self.parameters['V_lim']
        I_rated = self.parameters['rated_current']
        fit_coeff_dep_H = self.parameters['fit_coeff_dep_H']
        fit_coeff_dep_B = self.parameters['fit_coeff_dep_B']

        

        self.add(
            TC1MotorSizingModel(
                pole_pairs=pole_pairs,
                phases=phases,
                num_slots=num_slots,
                rated_current=I_rated,
                component_name='dummy'
            ),
            'sizing_model'
        )

        self.add(
            TC1MotorAnalysisModel(
                pole_pairs=pole_pairs,
                phases=phases,
                num_slots=num_slots,
                V_lim=V_lim,
                rated_current=I_rated,
                fit_coeff_dep_B=fit_coeff_dep_B,
                fit_coeff_dep_H=fit_coeff_dep_H,
                num_nodes=num_nodes,
                num_active_nodes=num_active_nodes,
                use_caddee=False,
                gear_ratio=1.
            ),
            'analysis_model'
        )


# D_i = 0.182
# L = 0.086

D_i = 0.11
L = 0.08

load_torque_rotor = np.array([172.317])
omega_rotor = np.array([3375.10])

num_nodes=len(load_torque_rotor)

m = FullMotorModel(
    num_nodes=num_nodes,
    V_lim=400
)

m.create_input('motor_diameter')
m.create_input('motor_length')

m.add_constraint('efficiency_active', lower=0.90)
m.add_constraint('torque_delta', lower=.3)

m.add_design_variable('motor_diameter', lower = 0.1)
m.add_design_variable('motor_length', lower = 0.06)

m.add_objective('motor_mass')

sim = Simulator(m)
sim['motor_diameter'] = D_i
sim['motor_length'] = L

sim['rpm'] = omega_rotor
sim['load_torque_rotor'] = load_torque_rotor

sim.run()
print(sim['motor_mass'])

from modopt.csdl_library import CSDLProblem
from modopt.snopt_library import SNOPT
prob = CSDLProblem(problem_name='motor_opt_test', simulator=sim)

optimizer = SNOPT(
    prob, 
    Major_iterations=50, 
    Major_optimality=1e-5, 
    Major_feasibility=1e-5,
    append2file=True,
    Iteration_limit=500000,
    Print_frequency=1
)

# optimizer = SLSQP(prob, maxiter=500, ftol=1E-5)

import time
t_start = time.time()
optimizer.solve()
optimizer.print_results()
t_end = time.time()
opt_time = t_end - t_start