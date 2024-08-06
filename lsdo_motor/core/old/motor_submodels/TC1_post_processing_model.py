import numpy as np
import matplotlib.pyplot as plt

from csdl import Model
import csdl
from python_csdl_backend import Simulator

class PostProcessingModel(Model):
    def initialize(self):
        self.parameters.declare('pole_pairs') # 6
        self.parameters.declare('phases') # 3
        self.parameters.declare('V_lim')
        self.parameters.declare('num_nodes')

    def define(self):

        m = self.parameters['phases']
        p = self.parameters['pole_pairs']
        V_lim = self.parameters['V_lim']
        num_nodes = self.parameters['num_nodes']

        ''' FLUX WEAKENING (I_d) & MTPA SMOOTHING (I_q) '''
        # convert I_d of flux weakening into I_q using torque equation
        Id_fw = self.declare_variable('Id_fw', shape=(num_nodes,))
        L_q = self.declare_variable('L_q')
        L_d = self.declare_variable('L_d')
        phi_air = self.declare_variable('phi_air')
        T_em = self.declare_variable('T_em', shape=(num_nodes,))

        L_d_expanded = csdl.expand(L_d, (num_nodes,))
        L_q_expanded = csdl.expand(L_q, (num_nodes,))
        phi_m_expanded = csdl.expand(phi_air, (num_nodes,))

        Iq_fw = T_em / (1.5*p * (phi_m_expanded + (L_d_expanded-L_q_expanded)*Id_fw)) # CHECK SIZE OF COMPUTATIONS HERE

        # smoothing
        Iq_MTPA = self.declare_variable('Iq_MTPA', shape=(num_nodes,)) # CHECK NAMING SCHEME FOR VARIABLE
        k = 1 # ASK SHUOFENG WHAT THIS IS
        I_q = (np.exp(k*(op_voltage - V_lim))*Iq_fw + np.exp(-1/(k*(op_voltage - V_lim)))*Iq_MTPA) / \
            (np.exp(k*(op_voltage - V_lim)) + np.exp(-1/(k*(op_voltage - V_lim))))

        # calculating I_d
        I_d = (T_em / (1.5*p*I_q) - phi_m_expanded) / (L_d_expanded-L_q_expanded) # CHECK SIZE OF COMPUTATIONS HERE

        ''' POWER LOSS CALCULATIONS '''
        torque = self.declare_variable('torque', shape=(num_nodes,))
        omega = self.declare_variable('omega', shape=(num_nodes,))
        # load power
        # eq of the form P0 = speed * torque
        P0 = torque * omega
        frequency = omega*p/60

        # copper loss
        Rdc = self.declare_variable('Rdc')
        Rdc_expanded = csdl.expand(Rdc, (num_nodes,))
        P_copper = m*Rdc_expanded*(I_q**2 + I_d**2)**0.5 # NEED TO CHECK IF THIS IS ELEMENT-WISE SQUARING

        # eddy_loss
        a = 0.00055 # lamination thickness in m
        sigma_c = 2e6 # bulk conductivity (2000000 S/m)
        l_ef = self.declare_variable('l_ef')
        D1 = self.declare_variable('outer_stator_radius')
        D_i = self.declare_variable('inner_stator_radius')
        Acu = self.declare_variable('Acu')
        B_delta = self.declare_variable('B_delta')
        B_delta_expanded = csdl.expand(B_delta, (num_nodes,))
        
        K_e = (a*np.pi)**2 * sigma_c/6
        V_s = csdl.expand((np.pi*l_ef*(D1-D_i)**2)-36*l_ef*Acu, (num_nodes,)); # volume of stator
        K_c = 0.822;
        P_eddy = K_e*V_s*(B_delta_expanded*frequency)**2; # eddy loss

        # hysteresis loss
        K_h = 100
        n_h = 2
        P_h = K_h*V_s*frequency*B_delta_expanded**n_h

        # stress loss
        P_stress = 0.01*P0

        # windage & friction loss
        k_r = 4 # roughness coefficient
        fr = 3e-3 # friction coefficient
        rho_air = 1.225 # air density kg/m^3
        D2 = self.declare_variable('rotor_radius')
        l_ef_expanded = csdl.expand(l_ef, (num_nodes,))
        D2_expanded = csdl.expand(D2, (num_nodes,))
        P_wo = k_r*np.pi*fr*rho_air*(2*np.pi*frequency)**2*l_ef_expanded*D2_expanded**4

        # total losses
        P_loss = P_copper + P_eddy + P_h + P_stress + P_wo
        efficiency = P0/(P0+P_loss)

        efficiency = self.register_output('efficiency', efficiency)
        self.register_output('input_power', P0 + P_loss)
        
''' ---- NOTES ----
- MTPA-Flux Weakening smoothing happens here

- we are calculating the power losses
    - copper, eddy, hysteresis, windage & friction
- use same methods from the high-fidelity modeling to calculate torque loss
- final output is the efficiency for the implicit motor analysis model

'''
