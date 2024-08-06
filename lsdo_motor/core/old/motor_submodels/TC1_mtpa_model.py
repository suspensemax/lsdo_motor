import numpy as np
import matplotlib.pyplot as plt

from csdl import Model, ScipyKrylov, NewtonSolver
import csdl
from python_csdl_backend import Simulator

class MTPAImplicitModel(Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        num_nodes = self.parameters['num_nodes']
        I_base_expanded = self.declare_variable('I_base_expanded', shape=(num_nodes,))
        MTPA_upper_bracket = self.declare_variable('MTPA_upper_bracket', shape=(num_nodes,))
        T_em_star = self.declare_variable('T_em_star', shape=(num_nodes,))

        model = Model()
        T_em_star = model.declare_variable('T_em_star', shape=(num_nodes,))

        Iq_MTPA_star = model.declare_variable('Iq_MTPA_star', shape=(num_nodes,))
        model.register_output(
            'MTPA_residual',
            Iq_MTPA_star**4 + T_em_star*Iq_MTPA_star - T_em_star**2
        )
        mtpa_lower_bracket = self.declare_variable('mtpa_lower_bracket', val=0., shape=(num_nodes,))
        mtpa_implicit_op = self.create_implicit_operation(model)
        mtpa_implicit_op.declare_state(
            'Iq_MTPA_star',
            residual='MTPA_residual',
            bracket=(mtpa_lower_bracket, MTPA_upper_bracket)
        )

        T_em_star = self.declare_variable('T_em_star', shape=(num_nodes,))
        Iq_MTPA_star = mtpa_implicit_op(T_em_star)

    
class MTPAModel(Model):
    def initialize(self):
        self.parameters.declare('pole_pairs') # 6
        self.parameters.declare('num_nodes')

    def define(self):
        p = self.parameters['pole_pairs']
        num_nodes = self.parameters['num_nodes']

        T_em = self.declare_variable('T_em', shape=(num_nodes,))
        L_d_expanded = self.declare_variable('L_d_expanded', shape=(num_nodes,))
        L_q_expanded = self.declare_variable('L_q_expanded', shape=(num_nodes,))
        PsiF_expanded = self.declare_variable('PsiF_expanded', shape=(num_nodes,))

        I_base_expanded = self.register_output(
            'I_base_expanded',
            -1*PsiF_expanded/(L_d_expanded - L_q_expanded)
        )

        T_em_base = 1.5*p*PsiF_expanded*I_base_expanded

        self.register_output('MTPA_upper_bracket', 50*I_base_expanded)

        T_em_star = self.register_output(
            'T_em_star',
            T_em/T_em_base
        ) # NON-DIM TORQUE


        self.add(
            MTPAImplicitModel(
                num_nodes=num_nodes
            ),
            'MTPAImplicitModel'
        )
        Iq_MTPA_star = self.declare_variable('Iq_MTPA_star', shape=(num_nodes,))

        # --- ADD POST-PROCESSING OF THE OUTPUT Iq_MTPA_star
        Iq_MTPA = self.register_output(
            'Iq_MTPA',
            Iq_MTPA_star * I_base_expanded
        ) # NEED TO DIMENSIONALIZE Iq_MTPA_star COMING FROM MTPA IMPLICIT SOLVER

if __name__ == '__main__':
    num_nodes = 2
    p = 6

    m = MTPAModel(
        num_nodes=num_nodes, pole_pairs=p
    )

    # rep = GraphRepresentation(m)
    sim = Simulator(m)

    sim['T_em'] = 5000
    sim['L_d_expanded'] = 0.0011
    sim['L_q_expanded'] = 0.0022
    sim['PsiF_expanded'] = 0.5494

    sim.run()
    print(sim['Iq_MTPA_star'])
    print(sim['Iq_MTPA'])
