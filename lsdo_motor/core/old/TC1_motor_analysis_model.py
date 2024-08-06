import numpy as np
from csdl import Model, ScipyKrylov, NewtonSolver
from python_csdl_backend import Simulator
import csdl

from lsdo_motor.core.old.motor_submodels.TC1_magnet_mec_model import MagnetMECModel
from lsdo_motor.core.old.motor_submodels.TC1_inductance_mec_model import InductanceModel
from lsdo_motor.core.old.motor_submodels.TC1_torque_limit_model import TorqueLimitModel
from lsdo_motor.core.old.motor_submodels.TC1_implicit_em_torque_model import EMTorqueModel
from lsdo_motor.core.old.motor_submodels.TC1_flux_weakening_model import FluxWeakeningModel, FluxWeakeningBracketModel
from lsdo_motor.core.old.motor_submodels.TC1_mtpa_model import MTPAModel
from lsdo_motor.core.old.motor_submodels.TC1_post_processing_model import PostProcessingModel


'''
THIS MODEL CONTAINS THE ENTIRE MOTOR ANALYSIS MODEL.
THE GOAL OF THIS MODEL IS TO IMPLICITLY SOLVE FOR POWER AND EFFICIENCY
GIVEN A RESIDUAL CONSIDERING TORQUE AND EFFICIENCY.

FEOC = FOR EACH OPERATING CONDITION
INPUTS:
    - motor geometry (from sizing model)
    - RPM and load torque (FEOC)
    - voltage limit
    - voltage (FEOC); fed back in from the battery model

OUTPUTS:
    - power and efficiency (FEOC); fed into other models
    - EM torque (FEOC)
'''

class ParseActiveOperatingConditions(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('num_active_nodes') # COMING FROM CADDEE

    def define(self):

        self.num_nodes = self.parameters['num_nodes']
        self.num_active_nodes = self.parameters['num_active_nodes']

        self.add_input('rpm', shape=(self.num_nodes,))
        self.add_input('load_torque_rotor', shape=(self.num_nodes,))

        self.add_output('omega_rotor_active', shape=(self.num_active_nodes,))
        self.add_output('load_torque_rotor_active', shape=(self.num_active_nodes,))
        self.add_output('selection_indices', shape=(self.num_nodes,self.num_active_nodes))

        self.declare_derivatives('omega_rotor_active', 'rpm')
        self.declare_derivatives('load_torque_rotor_active', 'load_torque_rotor')

    def compute(self, inputs, outputs):

        omega_rotor = inputs['rpm']
        load_torque_rotor = inputs['load_torque_rotor']

        omega_rotor_active = []
        load_torque_rotor_active = []
        self.selection_indices = np.zeros((self.num_nodes, self.num_active_nodes))

        active_counter = 0
        for i in range(self.num_nodes):
            if omega_rotor[i] != 0 and load_torque_rotor[i] != 0:
                self.selection_indices[i, active_counter] = 1
                omega_rotor_active.append(omega_rotor[i])
                load_torque_rotor_active.append(load_torque_rotor[i])
                active_counter += 1
        
        outputs['omega_rotor_active'] = np.array(omega_rotor_active)
        outputs['load_torque_rotor_active'] = np.array(load_torque_rotor_active)
        outputs['selection_indices'] = self.selection_indices
        
    def compute_derivatives(self, inputs, derivatives):
        derivatives['omega_rotor_active', 'rpm'] = np.transpose(self.selection_indices)
        derivatives['load_torque_rotor_active', 'load_torque_rotor'] = np.transpose(self.selection_indices)

class MotorParametersModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('motor_variable_names')

    def define(self):
        motor_variable_names = self.parameters['motor_variable_names']

        motor_parameters = self.declare_variable('motor_parameters', shape=(27,)) # array of motor sizing outputs
        for i in range(motor_parameters.shape[0]-2):
            self.register_output(motor_variable_names[i], motor_parameters[i])

class TC1MotorAnalysisModel(csdl.Model):
    def initialize(self, model_test=False):
        self.parameters.declare('pole_pairs') # 6
        self.parameters.declare('phases') # 3
        self.parameters.declare('num_slots') # 36
        self.parameters.declare('V_lim')
        self.parameters.declare('rated_current')
        self.parameters.declare('fit_coeff_dep_H') # FITTING COEFFICIENTS (X = H, B = f(H))
        self.parameters.declare('fit_coeff_dep_B') # FITTING COEFFICIENTS (X = B, H = g(B))
        self.parameters.declare('num_nodes')
        self.parameters.declare('num_active_nodes', default=None)
        self.parameters.declare('flux_weakening', default=False)
        self.parameters.declare('use_caddee', default=True)

        self.parameters.declare('gear_ratio', default=4.)


        self.motor_variable_names = [
            'outer_stator_radius', 'pole_pitch', 'tooth_pitch', 'air_gap_depth', 'l_ef',
            'rotor_radius', 'turns_per_phase', 'Acu',  'tooth_width', 'height_yoke_stator',
            'slot_bottom_width', 'slot_height', 'slot_width_inner', 'Tau_y', 'L_j1', 'Kdp1',
            'bm', 'Am_r', 'phi_r', 'lambda_m', 'alpha_i', 'Kf', 'K_phi', 'K_theta', 'A_f2',
            'Rdc', 'T_em_max' # LAST TWO ARE SPECIAL VARIABLES
        ]

    def define(self):
        # INPUT PARAMETERS
        m = self.parameters['phases']
        p = self.parameters['pole_pairs']
        Z = self.parameters['num_slots']
        V_lim = self.parameters['V_lim']
        rated_current = self.parameters['rated_current']
        fit_coeff_dep_H = self.parameters['fit_coeff_dep_H']
        fit_coeff_dep_B = self.parameters['fit_coeff_dep_B']
        num_nodes = self.parameters['num_nodes']
        num_active_nodes = self.parameters['num_active_nodes']
        flux_weakening=self.parameters['flux_weakening']
        use_caddee = self.parameters['use_caddee']

        # DECLARE VARIABLES FROM SIZING & UPSTREAM MODELS
        D_i = self.declare_variable('motor_diameter') # Diameter (DV or input)
        # T_em_max = self.declare_variable('T_em_max') # Max torque (structural)
        # Rdc = self.declare_variable('Rdc') # Resistance
        # motor_parameters = self.declare_variable('motor_parameters', shape=(27,)) # array of motor sizing outputs
        # for i in range(motor_parameters.shape[0]-2):
        #     self.register_output(self.motor_variable_names[i], motor_parameters[i])
        motor_parameters = self.declare_variable('motor_parameters', shape=(27,)) # array of motor sizing outputs

        self.add(MotorParametersModel(motor_variable_names=self.motor_variable_names), 'motor_parameters_model')

        if use_caddee:
            Rdc = self.register_output('Rdc', motor_parameters[-2])
            T_em_max = self.register_output('T_em_max', motor_parameters[-1])
        else:
            Rdc = motor_parameters[-2] # NOTE: CHECK IF THIS IS CAUSING ISSUES BC IT'S NOT DECLARED AS A VARIABLE
            T_em_max = motor_parameters[-1] # NOTE: CHECK IF THIS IS CAUSING ISSUES BC IT'S NOT DECLARED AS A VARIABLE


        # ========================= MAGNET MEC =========================
        self.add(
            MagnetMECModel(
                fit_coeff_dep_H=fit_coeff_dep_H,
                fit_coeff_dep_B=fit_coeff_dep_B,
            ),
            'magnet_MEC_model',
        )

        # ========================= INDUCTANCE MEC =========================
        self.add(
            InductanceModel(
                pole_pairs=p,
                phases=m,
                num_slots=Z,
                rated_current=rated_current,
                fit_coeff_dep_H=fit_coeff_dep_H,
                fit_coeff_dep_B=fit_coeff_dep_B,
            ),
            'inductance_MEC_model',
        )
        L_d = self.declare_variable('L_d')
        L_q = self.declare_variable('L_q')
        phi_air = self.declare_variable('phi_air')
        W_1 = self.declare_variable('turns_per_phase')
        PsiF = self.register_output('PsiF', W_1 * phi_air * D_i / D_i) # added D_i so that connection is easier

        # ========================= TORQUE AND RPM INPUT ADJUSTMENT =========================
        omega_rotor = self.declare_variable('rpm', shape=(num_nodes,))
        load_torque_rotor = self.declare_variable('load_torque_rotor', shape=(num_nodes,))

        if num_active_nodes is None:
            num_active_nodes_orig = num_active_nodes
            num_active_nodes = num_nodes
        else:
            omega_rotor_active, load_torque_rotor_active, selection_indices = csdl.custom(
                omega_rotor, load_torque_rotor,
                op=ParseActiveOperatingConditions(num_nodes=num_nodes, num_active_nodes=num_active_nodes)
            )

            # omega_rotor_active = self.register_output('omega_rotor_active', omega_rotor_active)
            # load_torque_rotor_active = self.register_output('load_torque_rotor_active', load_torque_rotor_active)

            omega_rotor = self.register_output('omega_rotor_active', omega_rotor_active)
            load_torque_rotor = self.register_output('load_torque_rotor_active', load_torque_rotor_active)

        R_expanded = self.register_output('R_expanded', csdl.expand(Rdc, (num_active_nodes,)))
        L_d_expanded = self.register_output('L_d_expanded', csdl.expand(L_d, (num_active_nodes,)))
        L_q_expanded = self.register_output('L_q_expanded', csdl.expand(L_q, (num_active_nodes,)))
        PsiF_expanded = self.register_output('PsiF_expanded', csdl.expand(PsiF, (num_active_nodes,)))

        # ========================= GEARBOX =========================
        gear_ratio = self.parameters['gear_ratio']
        omega = self.register_output('omega', omega_rotor * gear_ratio * 2*np.pi*p/60)
        load_torque = self.register_output('load_torque', load_torque_rotor/gear_ratio)

        # ========================= FINDING UPPER TORQUE LIMIT (DISCRIMINANT = 0 CASE) =========================
        self.add(
                TorqueLimitModel(
                    pole_pairs=p,
                    V_lim=V_lim,
                    num_nodes=num_active_nodes
                ),
                'torque_limit_model'
        )
        T_lim = self.declare_variable('T_lim', shape=(num_active_nodes,))
        T_lower_lim = self.declare_variable('T_lower_lim', val=0., shape=(num_active_nodes,))
        
        D = (3*p*(L_d_expanded-L_q_expanded))
        if flux_weakening:

            # ========================= FLUX WEAKENING BRACKETING IMPLICIT MODEL (TO FIND BRACKET LIMIT) =========================
            a_bracket = self.register_output(
                'a_bracket',
                D**2 * ((omega*L_q_expanded)**2 + R_expanded**2)
            )
            c_bracket = self.register_output(
                'c_bracket',
                (3*p*PsiF_expanded)**2 * (R_expanded**2 + (omega*L_q_expanded)**2) + \
                12*p*omega*R_expanded*T_lim*(L_d_expanded-L_q_expanded)**2 - (V_lim*D)**2
            )
            d_bracket = self.register_output(
                'd_bracket',
                -12*p*PsiF_expanded*T_lim*(R_expanded**2 + omega**2*L_d_expanded*L_q_expanded)
            )
            e_bracket = self.register_output(
                'e_bracket',
                4*T_lim**2*(R_expanded**2 + (omega*L_d_expanded)**2)
            )
            
            self.add(
                FluxWeakeningBracketModel(
                    pole_pairs=p,
                    num_nodes=num_active_nodes
                ),
                'flux_weakening_bracket_method'
            )

            Iq_fw_bracket = self.declare_variable('Iq_fw_bracket', shape=(num_active_nodes, ))
            Id_fw_bracket = self.declare_variable('Id_fw_bracket', shape=(num_active_nodes, ))

        self.add(
            EMTorqueModel(
                pole_pairs=p,
                V_lim=V_lim,
                num_nodes=num_active_nodes,
                rated_current=rated_current,
                phases=m,
                motor_variable_names=self.motor_variable_names,
                flux_weakening=flux_weakening
            ),
            'implicit_em_torque_model'
        )
        if num_active_nodes_orig is None:
            input_power = self.declare_variable('input_power_active', shape=(num_active_nodes,))
            # self.register_output('input_power', input_power * 1.)
            self.register_output('input_power', input_power * 1.)
        else:
            input_power_active = self.declare_variable('input_power_active', shape=(num_active_nodes,))
            input_power = csdl.matvec(selection_indices, input_power_active)
            self.register_output('input_power', input_power)

        '''
        NOTE TO SELF:
        Make this active calculation available for all relevant outputs:
            - efficiency, input power, output power, load torque, EM torque (potentially others)
        '''
        
        
        # CALCULATING UPPER LIMIT TORQUE CURVE  
        if flux_weakening:
            T_upper_lim_curve = self.register_output(
                'T_upper_lim_curve',
                # -csdl.log(csdl.exp(-T_lim) + csdl.exp(-csdl.expand(T_em_max, (num_active_nodes,))))
                csdl.min(csdl.reshape(T_lim, new_shape=(num_active_nodes, )),
                            csdl.expand(T_em_max, (num_active_nodes, )))
            )
    
            max_torque_constraint = self.register_output(name='torque_delta',
                                                            var=(T_upper_lim_curve-load_torque)/T_upper_lim_curve)
        else:
            T_upper_lim_curve = self.register_output(
                'T_upper_lim_curve',
                csdl.expand(T_em_max, (num_active_nodes, ))
            )
            max_torque_constraint = self.register_output(name='torque_delta',
                                                            var=(T_upper_lim_curve-load_torque)/T_upper_lim_curve)
        


