import numpy as np
import matplotlib.pyplot as plt

from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import csdl
from python_csdl_backend import Simulator

class InductanceQImplicitModel(Model):
    def initialize(self):
        self.parameters.declare('pole_pairs') # 6
        self.parameters.declare('phases') # 3
        self.parameters.declare('num_slots') # 36
        self.parameters.declare('rated_current')
        self.parameters.declare('rated_d_current')
        self.parameters.declare('fit_coeff_dep_H') # FITTING COEFFICIENTS (X = H, B = f(H))
        self.parameters.declare('fit_coeff_dep_B') # FITTING COEFFICIENTS (X = B, H = g(B))

    def fitting_dep_H(self, H):
        f = self.fit_coeff_dep_H[0] * csdl.tanh(H/300 - 0.25) + 0.4
        return f

    def fitting_dep_B(self, B):
        a = self.fit_coeff_dep_B[0]
        b = self.fit_coeff_dep_B[1]
        c = self.fit_coeff_dep_B[2]
        f = (a*csdl.exp(b*B + c) + 200) * B**1.4
        return f

    def define(self):
        m = self.parameters['phases']
        p = self.parameters['pole_pairs']
        Z = self.parameters['num_slots']
        I_w = self.parameters['rated_current']
        I_d_temp = self.parameters['rated_d_current']
        self.fit_coeff_dep_H = self.parameters['fit_coeff_dep_H']
        self.fit_coeff_dep_B = self.parameters['fit_coeff_dep_B']
        
        phi_aq = self.declare_variable('phi_aq') # STATE

        alpha_i = self.declare_variable('alpha_i')
        pole_pitch = self.declare_variable('pole_pitch')
        l_ef = self.declare_variable('l_ef')
        B_aq = phi_aq/(alpha_i*pole_pitch*l_ef)

        K_theta = self.declare_variable('K_theta')
        air_gap_depth = self.declare_variable('air_gap_depth')
        mu_0 = np.pi*4e-7
        F_sigma_q = 1.6*B_aq*(K_theta*air_gap_depth)/mu_0

        tooth_pitch = self.declare_variable('tooth_pitch')
        tooth_width = self.declare_variable('tooth_width')
        kfe = 0.95 # LAMINATION COEFFICIENT

        B_t_q = B_aq*tooth_pitch/tooth_width/kfe
        H_t1_q = self.fitting_dep_B(B_t_q)
        h_slot = self.declare_variable('slot_height')
        F_t1_q = 2*H_t1_q*h_slot # DIFF OF MAGNETIC POTENTIAL ALONG TOOTH

        h_ys = self.declare_variable('height_yoke_stator')
        B_j1_q = phi_aq/(2*l_ef*h_ys)
        H_j1_q = self.fitting_dep_B(B_j1_q)

        Kaq = self.declare_variable('Kaq')
        Kdp1 = self.declare_variable('Kdp1')
        turns_per_phase = self.declare_variable('turns_per_phase')
        L_j1 = self.declare_variable('L_j1')
        F_j1_q = 2*H_j1_q*L_j1
        F_total_q = F_sigma_q + F_t1_q + F_j1_q # TOTAL MAGNETIC STRENGTH ON Q-AXIS
        I_q_temp = self.register_output(
            'I_q_temp',
            p*F_total_q/(0.9*m*Kaq*Kdp1*turns_per_phase)
         ) #  CURRENT AT Q-AXIS
        
        
        inductance_residual = self.register_output(
            'inductance_residual',
            I_d_temp**2 + I_q_temp**2 - I_w**2
        )

class InductanceModel(Model):
    def initialize(self):
        self.parameters.declare('pole_pairs') # 6
        self.parameters.declare('phases') # 3
        self.parameters.declare('num_slots') # 36
        self.parameters.declare('rated_current')
        self.parameters.declare('fit_coeff_dep_H') # FITTING COEFFICIENTS (X = H, B = f(H))
        self.parameters.declare('fit_coeff_dep_B') # FITTING COEFFICIENTS (X = B, H = g(B))

    def define(self):
        m = self.parameters['phases']
        p = self.parameters['pole_pairs']
        Z = self.parameters['num_slots']
        I_w = self.parameters['rated_current']
        self.fit_coeff_dep_H = self.parameters['fit_coeff_dep_H']
        self.fit_coeff_dep_B = self.parameters['fit_coeff_dep_B']

        rated_omega = 3000
        f_i = rated_omega*p/60
        I_d_temp = I_w * np.sin(0.6283)

        Kf = self.declare_variable('Kf')
        Kaq = self.register_output('Kaq', 0.36/Kf)
        phi_air = self.declare_variable('phi_air')

        '''--------- q-axis inductance (implicit operation) ---------'''
        q_inductance_model = InductanceQImplicitModel(
            pole_pairs = p,
            phases=m,
            num_slots=Z,
            rated_current = I_w,
            rated_d_current = I_d_temp,
            fit_coeff_dep_H = self.fit_coeff_dep_H,
            fit_coeff_dep_B = self.fit_coeff_dep_B,
        )

        eps = self.create_input('eps', 1e-5)
        Inductance_MEC = self.create_implicit_operation(q_inductance_model)
        Inductance_MEC.declare_state('phi_aq', residual='inductance_residual', bracket=(eps, phi_air))

        alpha_i = self.declare_variable('alpha_i')
        pole_pitch = self.declare_variable('pole_pitch')
        l_ef = self.declare_variable('l_ef')
        K_theta = self.declare_variable('K_theta')
        air_gap_depth = self.declare_variable('air_gap_depth')
        tooth_pitch = self.declare_variable('tooth_pitch')
        tooth_width = self.declare_variable('tooth_width')
        h_slot = self.declare_variable('slot_height')
        h_ys = self.declare_variable('height_yoke_stator')
        # Kaq = self.declare_variable('Kaq')
        Kdp1 = self.declare_variable('Kdp1')
        W_1 = self.declare_variable('turns_per_phase')
        L_j1 = self.declare_variable('L_j1')
        
        phi_aq, I_q_temp = Inductance_MEC(
            alpha_i, pole_pitch, l_ef,  K_theta, air_gap_depth,
            tooth_pitch, tooth_width, h_slot, h_ys, Kaq, Kdp1, 
            W_1, L_j1,
            expose=['I_q_temp']
        )

        '''--------- d-axis inductance ---------'''
        F_total = self.declare_variable('F_total')
        F_delta = self.declare_variable('F_delta')
        mu_0 = np.pi*4e-7
        l_ef = self.declare_variable('l_ef')
        Kdp1 = self.declare_variable('Kdp1')
        W_1 = self.declare_variable('turns_per_phase')

        K_st = F_total/F_delta
        Cx = (4*np.pi*f_i*mu_0*l_ef*(Kdp1*W_1)**2) / p

        h_k = 0.0008 # NOT SURE HWAT THIS IS
        h_os = 1.5 * h_k # NOT SURE WHAT THIS IS
        b_sb = self.declare_variable('slot_bottom_width')
        b_s1 = self.declare_variable('slot_width_inner')

        lambda_U1 = (h_k/b_sb) + (2*h_os/(b_sb+b_s1))
        lambda_L1 = 0.45
        lambda_S1 = lambda_U1 + lambda_L1

        X_s1 = (2*p*m*lambda_S1*Cx)/(Z*Kdp1**2)

        s_total = 0.1
        pole_pitch = self.declare_variable('pole_pitch')
        air_gap_depth = self.declare_variable('air_gap_depth')
        K_theta = self.declare_variable('K_theta')

        X_d1 = (m*pole_pitch*s_total*Cx) / (air_gap_depth*K_theta*K_st*(np.pi*Kdp1)**2)

        l_B = l_ef + 2*0.01 # straight length of coil
        Tau_y = self.declare_variable('Tau_y')

        X_E1 = 0.47*Cx*(l_B - 0.64*Tau_y)/(l_ef*Kdp1**2)
        X_1 = X_s1+X_d1+X_E1

        Kf = self.declare_variable('Kf')
        Kad = self.register_output('Kad', 1/Kf)

        
        F_ad = 0.35*m*Kad*Kdp1*W_1*I_d_temp/p
        hm = 0.004 # MAGNET THICKNESS
        Hc = 847138 # MAGNET COERCIVITY
        K_sigma_air = self.declare_variable('K_sigma_air') # COEFFICIENT OF LEAKAGE IN AIR GAP

        f_a = F_ad / (K_sigma_air*hm*Hc)
        self.register_output('f_a', f_a) # DELETE

        lambda_n = self.declare_variable('lambda_n')
        lambda_leak_standard = self.declare_variable('lambda_leak_standard')
        Am_r = self.declare_variable('Am_r')
        Br = 1.1208 # MAGNET REMANENCE
        K_phi = self.declare_variable('K_phi')

        E_o = 4.44*f_i*Kdp1*W_1*phi_air*K_phi
        aa = 1.0 + (-1 * f_a) # 1 - f_a DOES NOT WORK
        bb = lambda_n * aa
        cc = lambda_n + 1
        bm_N = bb/cc
        phi_air_N_temp = bm_N + (-1) * (1.0 + (-1) * bm_N)*lambda_leak_standard
        phi_air_N = phi_air_N_temp * Am_r * Br
        E_d = 4.44*f_i*Kdp1*W_1*phi_air_N*K_phi # EM at d-axis
        Xad  = ((E_o-E_d)**2)**0.5/I_d_temp/2**0.5
        Xd = Xad + X_1
        Ld = self.register_output(
            'L_d',
            Xd / (2*np.pi*f_i)
        ) # d-axis inductance

        E_aq = phi_aq*E_o/phi_air # EMF @ Q-AXIS
        Xaq = E_aq/I_q_temp
        Xq = Xaq + X_1
        Lq = self.register_output(
            'L_q',
            Xq/(2*np.pi*f_i)
        )
