import csdl_alpha as csdl
import numpy as np

def torque_limit_model(inputs, num_nodes, V_lim, p):
    # p = self.parameters['pole_pairs']
    # V_lim = self.parameters['V_lim']
    # num_nodes = self.parameters['num_nodes']

    # R = inputs['Rdc']
    # Ld = inputs['L_d']
    # Lq = inputs['L_q']
    # omega = inputs['omega'] # shape=(num_nodes,))
    # PsiF = PsiF

    R_expanded = inputs['R_expanded']
    L_d_expanded = inputs['Ld_expanded'] # shape=(num_nodes,))
    L_q_expanded = inputs['Lq_expanded'] # shape=(num_nodes,))
    PsiF_expanded = inputs['PsiF_expanded'] # shape=(num_nodes,))
    omega = inputs['omega'] # shape=(num_nodes,))

    # COMPILING COEFFICIENTS FROM ORIGINAL I_q FLUX WEAKENING EQUATION
    den = 3*p*(L_d_expanded-L_q_expanded)
    a = den**2*((omega*L_q_expanded)**2 + R_expanded**2)
    # c_1 and c_2 below make up coefficients for c = A*T + B
    c_1 = 12*p*omega*R_expanded*(L_d_expanded-L_q_expanded)**2 # labeled A in notes
    c_2 = (3*p*PsiF_expanded)**2*(R_expanded**2 + (omega*L_q_expanded)**2) - (V_lim*den)**2 # labeled B in notes
    d = -12*p*PsiF_expanded*(omega**2*L_d_expanded*L_q_expanded + R_expanded**2) # coefficient without torque
    e = 4*((omega*L_d_expanded)**2 + R_expanded**2) # coefficient without torque

    # COMBINED COEFFICIENTS FOR QUARTIC TORQUE EQUATION (DISCRIMINANT = 0 CASE)
    A = 256*a**2*e**3 - 128*a*e**2*c_1**2 + 16*e*c_1**4
    B = -256*a*e**2*c_1*c_2 + 144*a*d**2*e*c_1 + 64*e*c_1**3*c_2 - 4*d**2*c_1**3
    C = -128*a*e**2*c_2**2 + 144*a*d**2*e*c_2 - 27*a*d**4 + 96*c_1**2*c_2**2*e - 12*d**2*c_1**2*c_2
    D = 64*e*c_1*c_2**3 - 12*d**2*c_1*c_2**2
    E = 16*e*c_2**4 - 4*d**2*c_2**3

    # finding upper and lower bracket
    discrete_check = DiscreteCheck(num_nodes=num_nodes)
    outputs = discrete_check.evaluate(A,B,C,D,E)
    lower_bracket = outputs.lower_bracket
    upper_bracket = outputs.upper_bracket

    # lower_bracket, upper_bracket = csdl.custom(A, B, C, D, E, op = DiscreteCheck(num_nodes = num_nodes))


    # recorder = csdl.Recorder(inline=True)
    # recorder.start()

    # inputs = csdl.VariableGroup()

    # inputs.x = csdl.Variable(value=0.0, name='x')
    # inputs.y = csdl.Variable(value=0.0, name='y')
    # inputs.z = csdl.Variable(value=0.0, name='z')

    # paraboloid = Paraboloid(a=2, b=4, c=12, return_g=True)
    # outputs = paraboloid.evaluate(inputs)

    # f = outputs.f
    # g = outputs.g

    # region implicit model to find limit torque
    T_lim = csdl.ImplicitVariable(name='T_lim', shape=(num_nodes,), value=1.)
    limit_torque_residual = (A/E*T_lim**4 + B/E*T_lim**3 + C/E*T_lim**2 + D/E*T_lim + 1) / 1e3

    solver = csdl.nonlinear_solvers.BracketedSearch('torque_lim_bs')
    solver.add_state(T_lim, limit_torque_residual, bracket=(lower_bracket, upper_bracket))
    solver.run()
    # endregion


    return T_lim

class DiscreteCheck(csdl.CustomExplicitOperation):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes

    def evaluate_residual(self, x, A, B, C, D, E):
        res = A*x**4 + B*x**3 + C*x**2 + D*x + E
        return res

    def evaluate(self, A, B, C, D, E):
        self.declare_input('A', A)
        self.declare_input('B', B)
        self.declare_input('C', C)
        self.declare_input('D', D)
        self.declare_input('E', E)
        
        lower_bracket = self.create_output('lower_bracket', (self.num_nodes,))
        upper_bracket = self.create_output('upper_bracket', (self.num_nodes,))

        input_names_list = ['A', 'B', 'C', 'D', 'E']
        output_names_list = ['lower_bracket', 'upper_bracket']

        for i_name in input_names_list:
            for o_name in output_names_list:
                self.declare_derivative_parameters(
                    o_name,
                    i_name,
                    dependent=False
                )

        outputs = csdl.VariableGroup()
        outputs.lower_bracket = lower_bracket
        outputs.upper_bracket = upper_bracket

        return outputs
    
    def compute(self, input_vals, output_vals):
        A = input_vals['A']
        B = input_vals['B']
        C = input_vals['C']
        D = input_vals['D']
        E = input_vals['E']

        a = 4*A
        b = 3*B
        c = 2*C
        d = D

        lower_bracket_array = np.zeros((self.num_nodes,))
        upper_bracket_array = np.zeros((self.num_nodes,))

        t_shift = b/(3*a)
        p = (3*a*c - b**2) / (3*a**2)
        q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)
                
        for i in range(self.num_nodes):
            p_iter = p[i]
            q_iter = q[i]
            cond = 4*p_iter**3 + 27*q_iter**2 # THIS DETERMINES THE EXISTENCE OF VARIOUS ROOTS

            # IF CLAUSE TO COMPUTE LOWER BRACKET
            if cond > 0:

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

            # output_vals['lower_bracket'][i] = lower_bracket_val 
            lower_bracket_array[i] = lower_bracket_val
            # output_vals['lower_bracket'] = output_vals['lower_bracket'].set(
            #     csdl.slice[i],
            #     value=lower_bracket_val
            # )

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
            
            upper_bracket_array[i] = start
            # output_vals['upper_bracket'][i] = start
            # output_vals['upper_bracket'] = output_vals['upper_bracket'].set(
            #     csdl.slice[i],
            #     value=start
            # )
        
        output_vals['lower_bracket'] = lower_bracket_array
        output_vals['upper_bracket'] = upper_bracket_array

    
    def compute_derivatives(self, input_vals, output_vals, derivatives):
        A = input_vals['A']
        B = input_vals['B']
        C = input_vals['C']
        D = input_vals['D']
        E = input_vals['E']

        input_names_list = ['A', 'B', 'C', 'D', 'E']
        output_names_list = ['lower_bracket', 'upper_bracket']

        for i_name in input_names_list:
            for o_name in output_names_list:
                derivatives[o_name, i_name] = csdl.Variable(value=0.)
