import numpy as np 
import csdl_alpha as csdl

def mtpa_model(inputs, parameters):
    T_em = inputs['T_em']
    Ld = inputs['Ld']
    Lq = inputs['Lq']
    PsiF = inputs['PsiF']

    p = parameters['p']
    num_nodes = parameters['num_nodes']

    I_base = -1.*PsiF/(Ld-Lq)
    T_em_base = 1.5*p*PsiF*I_base
    T_em_star = T_em/T_em_base
    mtpa_upper_bracket = 50.*I_base

    # region implicit MTPA model
    Iq_mtpa_star = csdl.ImplicitVariable(name='Iq_mtpa', shape=(num_nodes,), value=1.)
    residual = Iq_mtpa_star**4 + T_em_star*Iq_mtpa_star - T_em_star**2

    solver = csdl.nonlinear_solvers.BracketedSearch('mtpa_bs')
    solver.add_state(Iq_mtpa_star, residual, bracket=(0., mtpa_upper_bracket))
    solver.run()
    # endregion

    Iq_mtpa = Iq_mtpa_star*I_base

    return Iq_mtpa