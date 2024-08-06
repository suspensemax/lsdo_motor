import numpy as np
import matplotlib.pyplot as plt

from csdl import Model, ScipyKrylov, NewtonSolver
import csdl
from python_csdl_backend import Simulator

class MagnetMECImplicitModel(Model):
    def initialize(self):
        self.parameters.declare('fit_coeff_dep_H') # FITTING COEFFICIENTS (X = H, B = f(H))
        self.parameters.declare('fit_coeff_dep_B') # FITTING COEFFICIENTS (X = B, H = g(B))

    def fitting_dep_H_old(self, H):
        f = []
        order = 10
        for i in range(order + 1):
            f.append(self.fit_coeff_dep_H[i] * H**(order-i))
        return csdl.sum(*f)
    ''' FOR LOOP MIGHT BE A PROBLEM BECAUSE OBJECT H IS BEING APPENDED TO A NORMAL PYTHON LIST '''

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
        self.fit_coeff_dep_H = self.parameters['fit_coeff_dep_H']
        self.fit_coeff_dep_B = self.parameters['fit_coeff_dep_B']

        B_delta = self.declare_variable('B_delta') # STATE

        ''' --- STATOR TOOTH CALCULATIONS --- '''
        t1 = self.declare_variable('tooth_pitch') # TOOTH PITCH
        b1 = self.declare_variable('tooth_width') # TOOTH WIDTH
        h_t1 = self.declare_variable('slot_height') # DEPTH OF SLOT (h_slot in MATLAB code)
        kfe = 0.95 # lamination coefficient
        
        B_t = B_delta * t1/b1/kfe # STATOR TOOTH FLUX DENSITY
        H_t = self.fitting_dep_B(B_t) # STATOR TOOTH MAGNETIC FIELD
        F_t = 2*H_t*h_t1 # MMF OF TOOTH

        ''' --- YOKE & ROTOR TOOTH CALCULATIONS --- '''
        alpha_i = self.declare_variable('alpha_i') # ELECTRICAL ANGLE PER SLOT
        tau = self.declare_variable('pole_pitch') # POLE PITCH
        l_ef = self.declare_variable('l_ef') # MAGNET LENGTH ALONG SHAFT (TYPICALLY STACK LENGTH)
        hj1 = self.declare_variable('height_yoke_stator') # YOKE HEIGHT IN STATOR
        ly = self.declare_variable('L_j1') # STATOR YOKE LENGTH OF MEC (CALLED L_j1 IN MATLAB & SIZING MODEL)

        phi_air = self.register_output(
            'phi_air',
            alpha_i*tau*l_ef*B_delta # AIR GAP FLUX; CALLED phi_air IN MATLAB CODE
        )
        B_y = phi_air / (2*hj1*l_ef) # YOKE FLUX DENSITY
        H_y = self.fitting_dep_B(B_y) # YOKE MAGNETIC FIELD
        self.register_output('H_y', H_y)
        F_y = 2*ly*H_y # YOKE MMF

        ''' --- AIR GAP MMF CALCULATIONS --- '''
        mu_0 = np.pi*4e-7
        sigma_air = self.declare_variable('air_gap_depth') # AIR GAP DEPTH
        K_theta = self.declare_variable('K_theta') # CARTER'S COEFF; CALLED K_theta IN MATLAB & SIZING MODEL

        F_delta = self.register_output(
            'F_delta',
            1.6*B_delta*(0.0001 + K_theta*sigma_air)/mu_0
        )
        
        ''' --- MMF SUM --- '''
        F_total = self.register_output(
            'F_total',
            F_t + F_y + F_delta
        )

        ''' --- MAGNETIC BRIDGE CALCULATIONS --- '''
        hm = 0.004 # MAGNET THICKNESS
        H_f = F_total / hm
        B_f = self.fitting_dep_H(H_f)

        ''' --- LEAKAGE FLUX CALCULATIONS --- '''
        # NOTE: phi_air already calculated above
        A_f2 = self.declare_variable('A_f2') # CROSS SECTIONAL AREA OF MAGNET BRIDGE
        lambda_s = 0.336e-6
        # NOTE: lambda_s is not typically a constant so need to check later

        phi_f = self.register_output(
            'phi_f',
            B_f * A_f2
        )

        phi_s = self.register_output(
            'phi_s',
            F_total * lambda_s
        )
        
        phi_mag = phi_air + phi_f + phi_s
        phi_mag = self.register_output(
            'phi_mag',
            phi_mag
        )

        ''' --- MAGNET MMF CALCULATIONS --- '''
        bm = self.declare_variable('bm') # ARC LENGTH OF MAGNET
        lambda_leak = l_ef*bm*mu_0/0.0005
        F_m = F_total + phi_mag/lambda_leak # MAGNET MMF

        # RESIDUAL FLUX OF MAGNET (COMPUTED IN SIZING MODEL)
        phi_r = self.declare_variable('phi_r') 
        lambda_m = self.declare_variable('lambda_m')

        phi_air = phi_r - F_m*lambda_m

        residual = self.register_output(
            'residual',
            phi_air - phi_mag
        )


class MagnetMECModel(Model):
    def initialize(self):
        self.parameters.declare('fit_coeff_dep_H') # FITTING COEFFICIENTS (X = H, B = f(H))
        self.parameters.declare('fit_coeff_dep_B') # FITTING COEFFICIENTS (X = B, H = g(B))

    def define(self):
        self.fit_coeff_dep_H = self.parameters['fit_coeff_dep_H']
        self.fit_coeff_dep_B = self.parameters['fit_coeff_dep_B']

        # IMPLICIT MODEL FOR MAGNET
        MECmodel = MagnetMECImplicitModel(
            fit_coeff_dep_H=self.fit_coeff_dep_H,
            fit_coeff_dep_B=self.fit_coeff_dep_B,
        )
        
        ''' --- SETTING UP NONLINEAR SOLVER --- '''
        eps = 1e-6
        Br  = 1.2
        lower_bound = eps
        upper_bound = Br
        solve_MEC = self.create_implicit_operation(MECmodel)
        solve_MEC.declare_state('B_delta', residual='residual', bracket=(lower_bound, upper_bound))

        ''' --- DECLARING VARIABLES AGAIN --- '''
        t1      = self.declare_variable('tooth_pitch') # TOOTH PITCH
        b1      = self.declare_variable('tooth_width') # TOOTH WIDTH
        h_t1      = self.declare_variable('slot_height') # HEIGHT OF SLOT
        alpha_i   = self.declare_variable('alpha_i') # ELECTRICAL ANGLE PER SLOT
        tau     = self.declare_variable('pole_pitch') # POLE PITCH
        l_ef    = self.declare_variable('l_ef') # MAGNET LENGTH ALONG SHAFT (TYPICALLY STACK LENGTH)
        hj1      = self.declare_variable('height_yoke_stator') # YOKE HEIGHT IN STATOR
        ly      = self.declare_variable('L_j1') # STATOR YOKE LENGTH OF MEC (CALLED L_j1 IN MATLAB & SIZING MODEL)
        sigma_air   = self.declare_variable('air_gap_depth') # AIR GAP DEPTH
        K_theta = self.declare_variable('K_theta') # CARTER'S COEFF; CALLED K_theta IN MATLAB & SIZING MODEL
        A_f2     = self.declare_variable('A_f2') # CROSS SECTIONAL AREA OF MAGNET BRIDGE
        bm      = self.declare_variable('bm') # ARC LENGTH OF MAGNET
        phi_r   = self.declare_variable('phi_r') 
        lambda_m = self.declare_variable('lambda_m')

        B_delta, phi_air, H_y, F_delta, F_total,  phi_f, phi_s, phi_mag = solve_MEC(
            t1, b1, h_t1, alpha_i, tau, l_ef, hj1, ly, sigma_air, K_theta, 
            A_f2, bm, phi_r, lambda_m, 
            expose=['phi_air', 'H_y', 'F_delta', 'F_total',  'phi_f', 'phi_s', 'phi_mag']
        )

        # --- MEC POST-PROCESSING ---
        mu_0 = np.pi*4e-7
        hm = 0.004 # MAGNET THICKNESS
        mu_r = 1.05
        Am_r = self.declare_variable('Am_r')

        K_sigma_air = self.register_output(
            'K_sigma_air',
            (phi_air+phi_f+phi_s)/phi_air
        )

        lambda_theta = phi_air/F_total # MAIN MAGNETIC CONDUCTION
        lambda_theta_standard = (2*lambda_theta*hm) / (mu_r*mu_0*Am_r) # STANDARD VALUE OF lambda_theta
        lambda_n = self.register_output(
            'lambda_n',
            K_sigma_air*lambda_theta_standard
        )

        bm_0 =  lambda_n/(lambda_n + 1) # OPERATING POINT OF MAGNET
        lambda_leak_standard = self.register_output(
            'lambda_leak_standard', 
            (K_sigma_air - 1)*lambda_theta_standard
        )


if __name__ == '__main__':
    from TC1_motor_model.permeability.mu_fitting import permeability_fitting
    file_name = 'Magnetic alloy, silicon core iron C.tab'
    order=10

    mu_fitting = permeability_fitting(
        file_name=file_name,
        test=True,
    )

    fit_coeff_dep_H = mu_fitting[0]
    fit_coeff_dep_B = mu_fitting[1]

    m = MagnetMECModel(
        fit_coeff_dep_H=fit_coeff_dep_H,
        fit_coeff_dep_B=fit_coeff_dep_B
    )

    sim = Simulator(m)
    print(sim['B_delta'])
    sim.run()
    print(sim['B_delta'])
    