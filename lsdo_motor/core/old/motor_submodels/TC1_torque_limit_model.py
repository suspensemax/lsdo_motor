import numpy as np
from csdl import Model, NewtonSolver, ScipyKrylov
import csdl
from python_csdl_backend import Simulator

class MaxTorqueImplicitModel(Model):
    def define(self):
        A = self.declare_variable('A_quartic')
        B = self.declare_variable('B_quartic')
        C = self.declare_variable('C_quartic')
        D = self.declare_variable('D_quartic')
        E = self.declare_variable('E_quartic')

        T_lim = self.declare_variable('T_lim', val=10000) # STATE

        T_lim_residual = self.register_output(
            'T_lim_residual',
            A/C*T_lim**4 + B/C*T_lim**3 + T_lim**2 + D/C*T_lim + E/C
        )

class MaxTorqueModel(Model):
    def initialize(self):
        self.parameters.declare('num_nodes')

    def define(self):
        num_nodes = self.parameters['num_nodes']

        A = self.declare_variable('A_quartic', shape=(num_nodes,))
        B = self.declare_variable('B_quartic', shape=(num_nodes,))
        C = self.declare_variable('C_quartic', shape=(num_nodes,))
        D = self.declare_variable('D_quartic', shape=(num_nodes,))
        E = self.declare_variable('E_quartic', shape=(num_nodes,))

        lower_bracket = self.declare_variable('lower_bracket', shape=(num_nodes,))
        upper_bracket = self.declare_variable('upper_bracket', shape=(num_nodes,))
        self.register_output('dummy_output', upper_bracket + lower_bracket) # HERE TO CREATE PROPER ORDER OF OPERATIONS

        # max_torque_implicit_model = MaxTorqueImplicitModel()
        max_torque_implicit_model = csdl.Model()
        A = max_torque_implicit_model.declare_variable('A_quartic', shape=(num_nodes,))
        B = max_torque_implicit_model.declare_variable('B_quartic', shape=(num_nodes,))
        C = max_torque_implicit_model.declare_variable('C_quartic', shape=(num_nodes,))
        D = max_torque_implicit_model.declare_variable('D_quartic', shape=(num_nodes,))
        E = max_torque_implicit_model.declare_variable('E_quartic', shape=(num_nodes,))
        T_lim = max_torque_implicit_model.declare_variable('T_lim', shape=(num_nodes,)) # STATE

        T_lim_residual = max_torque_implicit_model.register_output(
            'T_lim_residual',
            (A/E*T_lim**4 + B/E*T_lim**3 + C/E*T_lim**2 + D/E*T_lim + 1) / 1e3
        )

        max_torque_implicit_op = self.create_implicit_operation(max_torque_implicit_model)
        max_torque_implicit_op.declare_state(
            'T_lim',
            residual='T_lim_residual',
            bracket=(lower_bracket, upper_bracket)
        )
        max_torque_implicit_op.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=1000,
            iprint=True
        )
        max_torque_implicit_op.linear_solver = ScipyKrylov()

        A = self.declare_variable('A_quartic', shape=(num_nodes,))
        B = self.declare_variable('B_quartic', shape=(num_nodes,))
        C = self.declare_variable('C_quartic', shape=(num_nodes,))
        D = self.declare_variable('D_quartic', shape=(num_nodes,))
        E = self.declare_variable('E_quartic', shape=(num_nodes,))

        T_lim = max_torque_implicit_op(
            A, B, C, D, E
        )

class TorqueLimitModel(Model):
    def initialize(self):
        self.parameters.declare('pole_pairs')
        self.parameters.declare('V_lim')
        self.parameters.declare('num_nodes')

    def define(self):
        p = self.parameters['pole_pairs']
        V_lim = self.parameters['V_lim']
        num_nodes = self.parameters['num_nodes']
        R = self.declare_variable('Rdc')
        Ld = self.declare_variable('L_d')
        Lq = self.declare_variable('L_q')
        omega = self.declare_variable('omega', shape=(num_nodes,))
        PsiF = self.declare_variable('PsiF')

        R_expanded = self.declare_variable('R_expanded', shape=(num_nodes,))
        L_d_expanded = self.declare_variable('L_d_expanded', shape=(num_nodes,))
        L_q_expanded = self.declare_variable('L_q_expanded', shape=(num_nodes,))
        PsiF_expanded = self.declare_variable('PsiF_expanded', shape=(num_nodes,))

        # COMPILING COEFFICIENTS FROM ORIGINAL I_q FLUX WEAKENING EQUATION
        den = 3*p*(L_d_expanded-L_q_expanded)
        a = den**2*((omega*L_q_expanded)**2 + R_expanded**2)
        # c_1 and c_2 below make up coefficients for c = A*T + B
        c_1 = 12*p*omega*R_expanded*(L_d_expanded-L_q_expanded)**2 # labeled A in notes
        c_2 = (3*p*PsiF_expanded)**2*(R_expanded**2 + (omega*L_q_expanded)**2) - (V_lim*den)**2 # labeled B in notes
        d = -12*p*PsiF_expanded*(omega**2*L_d_expanded*L_q_expanded + R_expanded**2) # coefficient without torque
        e = 4*((omega*L_d_expanded)**2 + R_expanded**2) # coefficient without torque

        # COMBINED COEFFICIENTS FOR QUARTIC TORQUE EQUATION (DISCRIMINANT = 0 CASE)
        A = self.register_output('A_quartic', 256*a**2*e**3 - 128*a*e**2*c_1**2 + 16*e*c_1**4)
        B = self.register_output('B_quartic', -256*a*e**2*c_1*c_2 + 144*a*d**2*e*c_1 + 64*e*c_1**3*c_2 - 4*d**2*c_1**3)
        C = self.register_output('C_quartic', -128*a*e**2*c_2**2 + 144*a*d**2*e*c_2 - 27*a*d**4 + 96*c_1**2*c_2**2*e - 12*d**2*c_1**2*c_2)
        D = self.register_output('D_quartic', 64*e*c_1*c_2**3 - 12*d**2*c_1*c_2**2)
        E = self.register_output('E_quartic', 16*e*c_2**4 - 4*d**2*c_2**3)

        lower_bracket, upper_bracket = csdl.custom(A, B, C, D, E, op = DiscreteCheck(num_nodes = num_nodes))

        self.register_output('lower_bracket', lower_bracket)
        self.register_output('upper_bracket', upper_bracket)

        self.add(MaxTorqueModel(num_nodes=num_nodes), 'max_torque_model')


class DiscreteCheck(csdl.CustomExplicitOperation):

    def initialize(self):
        self.parameters.declare('num_nodes')

    def define(self):

        self.num_nodes = self.parameters['num_nodes']
        self.add_input('A_quartic', shape = (self.num_nodes, ))
        self.add_input('B_quartic', shape = (self.num_nodes, ))        
        self.add_input('C_quartic', shape = (self.num_nodes, ))       
        self.add_input('D_quartic', shape = (self.num_nodes, ))
        self.add_input('E_quartic', shape = (self.num_nodes, ))

        self.add_output('lower_bracket', shape = (self.num_nodes, ))
        self.add_output('upper_bracket', shape = (self.num_nodes, ))

    def evaluate_residual(self, x, A, B, C, D, E):
        res = A*x**4 + B*x**3 + C*x**2 + D*x + E
        return res

    def compute(self, inputs, outputs):

        A = inputs['A_quartic']
        B = inputs['B_quartic']
        C = inputs['C_quartic']
        D = inputs['D_quartic']
        E = inputs['E_quartic']

        a = 4*A
        b = 3*B
        c = 2*C
        d = D

        t_shift = b/(3*a)
        p = (3*a*c - b**2) / (3*a**2)
        q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)
                
        for i in range(self.num_nodes):
            p_iter = p[i]
            q_iter = q[i]
            cond = 4*p_iter**3 + 27*q_iter**2 # THIS DETERMINES THE EXISTENCE OF VARIOUS ROOTS

            # IF CLAUSE TO COMPUTE LOWER BRACKET
            if cond > 0:
                # cardano_sol = (-q_iter/2 + (q_iter**2/4 + p_iter**3/27)**(1/2))**(1/3) + \
                #     (-q_iter/2 - (q_iter**2/4 + p_iter**3/27)**(1/2))**(1/3)

                cubic_arg_1 = -q_iter/2 + (q_iter**2/4 + p_iter**3/27)**(1/2)
                cubic_arg_2 = -q_iter/2 - (q_iter**2/4 + p_iter**3/27)**(1/2)

                cardano_sol = np.cbrt(cubic_arg_1) + np.cbrt(cubic_arg_2)

                lower_bracket_val = cardano_sol - t_shift[i]

            elif cond == 0:
                t1 = 3*q_iter/p_iter
                t2 = t3 = -3*q_iter/p_iter

                lower_bracket_val = np.max(np.array([t1-t_shift[i], t2-t_shift[i], t3-t_shift[i]]))

            elif cond < 0:
                # print('cond less than 0')
                a_cubic = 3*B[i]/(4*A[i])
                b_cubic = 2*C[i]/(4*A[i])
                c_cubic = D[i]/(4*A[i])

                P1 = (a_cubic**2 - 3*b_cubic)/9 # Q IN NOTES
                P2 = (2*a_cubic**3 - 9*a_cubic*b_cubic + 27*c_cubic)/54 # R IN NOTES
                theta = np.arccos(P2/(P1**3)**(1/2))

                root1 = -1 * (2*(P1)**0.5*np.cos(theta/3)) - a_cubic/3
                root2 = -1 * (2*(P1)**0.5*np.cos((theta+2*np.pi)/3)) - a_cubic/3
                root3 = -1 * (2*(P1)**0.5*np.cos((theta-2*np.pi)/3)) - a_cubic/3

                lower_bracket_val = np.max(np.array([root1, root2, root3]))

            outputs['lower_bracket'][i] = lower_bracket_val 

            # ITERATIVE METHOD TO FIND UPPER BRACKET
            A_iter = A[i]
            B_iter = B[i]
            C_iter = C[i]
            D_iter = D[i]
            E_iter = E[i]
            start = lower_bracket_val
            res_start = self.evaluate_residual(start, A_iter, B_iter, C_iter, D_iter, E_iter)
            res_start_sign = np.sign(res_start) # GIVES SIGN OF STARTING RESIDUAL
            torque_step = 100
            j = 0
            while True:
                j += 1
                start = start + torque_step
                res_loop = self.evaluate_residual(start, A_iter, B_iter, C_iter, D_iter, E_iter)
                res_loop_sign = np.sign(res_loop)
                if res_start_sign != res_loop_sign and res_loop_sign != 0:
                    break

                if j > 1000:
                    KeyError('Method did not converge')
            
            outputs['upper_bracket'][i] = start

if __name__ == '__main__':
    p = 6
    
    V_lim = 1000
    Rdc = 0.0313
    L_d = 0.0011
    L_q = 0.0022
    omega = 1100
    PsiF = 0.5494 # 0.0153 OR 0.5494

    # V_lim = 3000
    # Rdc = 0.091
    # L_d = 0.0043
    # L_q = 0.0126
    # omega = 2000
    # PsiF = 1.2952

    m = TorqueLimitModel(
        pole_pairs=p,
        V_lim=V_lim,
        num_nodes=1
    )

    # rep = GraphRepresentation(m)

    sim = Simulator(m)

    sim['Rdc'] = Rdc
    sim['L_d'] = L_d
    sim['L_q'] = L_q
    sim['omega'] = omega
    sim['PsiF'] = PsiF

    sim.run()
    # print('before visualize_implementation:')
    # print('torque: (found implicitly)', sim['T_lim'])
    # sim.visualize_implementation()
    # print('after visualize_implementation:')
    print('torque: (found implicitly)', sim['T_lim'])
    print(sim['max_cubic_root'])
    print(sim['upper_quartic_bracket'])
    print(sim['A_quartic'])
    print(sim['B_quartic'])
    print(sim['C_quartic'])
    print(sim['D_quartic'])
    print(sim['E_quartic'])
    print('------')
    print(sim['test_out'])
    print(sim['cubic_roots'])
