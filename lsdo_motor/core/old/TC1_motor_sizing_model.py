import numpy as np 
import matplotlib.pyplot as plt
from python_csdl_backend import Simulator
# from python_csdl_backend import Simulator
from csdl import Model, GraphRepresentation
import csdl



class TorqueMassModel(Model):
    '''
    INPUTS TO THIS MODEL:
        - constant max torque (before base speed)
        - R, Ld, Lq, p,
        - omega for each operating condition

    OUTPUTS OF THIS MODEL:
        - base speed
        - max torque at each value of omega
    '''
    def initialize(self):
        self.parameters.declare('fitting_order')
        self.fitting_coeff = {
            '0':[1],
            # '1':[26.0489, -112.1432],
            '1':[25., 25.], # close to the one above, but crosses the x axis when motor mass < 0
            '2':[0.4840, 3.3169, 60.8142],
            '3':[1, 1, 1 ,1],
            '4':[1, 1, 1, 1, 1],
        }

    def fit_torque_to_mass(self, motor_mass):
        fitting_coeff = self.fitting_coeff.get(str(self.order))

        torque_fitting_array = self.create_output(
            'torque_fitting_array',
            shape=(self.order + 1,)
        )
        for i, val in enumerate(fitting_coeff):
            torque_fitting_array[i] = val * motor_mass**(self.order-i)
        return csdl.sum(torque_fitting_array)

    def define(self):
        self.order = self.parameters['fitting_order']        
        motor_mass = self.declare_variable('motor_mass')

        T_em_max = self.register_output(
            'T_em_max',
            self.fit_torque_to_mass(motor_mass)
        )

class TC1MotorSizingModel(csdl.Model):
    '''
    INPUTS TO THIS MODEL:
        - length and diameter of motor (as DVs or inputs)

    OUTPUTS OF THIS MODEL:
        - motor geometry
        - motor resistance
        - motor max torque (structural)

    '''
    def initialize(self):
        # MOTOR DISCRETE PARAMETERS
        self.parameters.declare('pole_pairs') # 6
        self.parameters.declare('phases') # 3
        self.parameters.declare('num_slots') # 36
        self.parameters.declare('rated_current')

    def define(self):
        # --- DEFINING INPUTS FROM INITIALIZE & BASIC PARAMETERS --- 
        m = self.parameters['phases']
        p = self.parameters['pole_pairs']
        Z = self.parameters['num_slots']
        I_w = self.parameters['rated_current']
        a = 1. # PARALLEL BRANCHES
        q = Z/(2*m*p) # SLOTS PER POLE PER PHASE
        mu_0 = np.pi*4e-7

        # --- RATED PARAMETERS AS INPUTS FROM OPTIMIZER ---
        D_i = self.declare_variable('motor_diameter', shape=(1,)) # inner radius of stator
        L = self.declare_variable('motor_length', shape=(1,)) # effective length of motor
        rated_omega = 5000
        eta_0 = 0.96 # ASSUMED INITIAL EFFICIENCY; MATLAB CODE STARTS WITH 0.88
        PF = 1 # POWER FACTOR

        f_i = rated_omega*p/60
        B_air_gap_max = 0.85 # MAX VALUE OF B IN AIR GAP
        alpha_B = 0.7 # RATIO OF B_avg/B_max IN AIR GAP [0.66, 0.71]
        kwm = 1.11 # COEFFICIENT OF B-AIR GAP CURVE (NOT USED)
        kdp1 = 0.925 # COEFFICIENT OF STATOR WINDING FUNDAMENTAL COMPONENT (NOT USED)

        line_load = 30000 # CURRENT PER UNIT LENGTH; THIS MAY NEED TO BE UPDATED
        # NOTE: THIS IS THE ELECTRIC LOADING THAT ZEYU FOUND FROM A LOOKUP TABLE IN A CHINESE TEXTBOOK;

        lambda_i = 1.25
        outer_stator_radius = D_i * lambda_i
        
        # --- POLE PITCH AND OTHER PITCHES ---
        pole_pitch = np.pi*D_i/(2*p)
        tooth_pitch = np.pi*D_i/Z
        
        # --- AIR GAP LENGTH ---
        air_gap_depth = 0.4*line_load*pole_pitch/(0.9e6*B_air_gap_max)
        l_ef = L + 2*air_gap_depth # final effective length of motor
        rotor_radius = D_i - 2*air_gap_depth # D2 in MATLAB code
        D_shaft = 0.3 * rotor_radius # outer radius of shaft
        
        # --- WINDINGS ---
        I_kw = I_w * eta_0 * PF
        conductors_per_phase = eta_0 * np.pi * D_i * line_load \
            / (m*I_kw)
        conductors_per_slot = m*a*conductors_per_phase/Z
        turns_per_phase = conductors_per_phase/2
        # these lines of code will cause errors compared to MATLAB code bc Zeyu
        # uses floor to round, and we cannot do that

        N_p = self.create_input('N_p', 2.)

        J = 5. # target current density
        Acu = I_w/(a*J*N_p) * 10.**(-6.)
        d_coil = 2 * (Acu/np.pi)**0.5

        # --- SLOT GEOMETRY ---
        kfe = 0.95 # LAMINATION COEFFICIENT
        Bt = 1.7 # FLUX DENSITY IN STATOR TOOTH

        tooth_width = tooth_pitch * B_air_gap_max / (kfe * Bt) # STATOR TOOTH WIDTH

        B_ys = 1.35 # FLUX DENSITY IN STATOR YOKE
        h_ys = (pole_pitch*alpha_B*B_air_gap_max) / (2*kfe*B_ys) # HEIGHT OF YOKE IN STATOR

        theta_t = 360/Z # ANGULAR SWEEP OF STATOR SLOT IN DEGREES
        theta_sso = 0.5*theta_t
        theta_ssi = 0.3*theta_sso
        b_sb = theta_ssi*np.pi*D_i/360 # WIDTH OF BOTTOM OF SLOT
        h_slot = (outer_stator_radius - D_i)/2 - h_ys # HEIGHT OF SLOT

        h_k = 0.0008 # NOT SURE HWAT THIS IS
        h_os = 1.5 * h_k # NOT SURE WHAT THIS IS

        b_s1 = (np.pi*(D_i+2*(h_os+h_k)))/36 - tooth_width # RADIALLY INNER WIDTH OF SLOT
        b_s2 = (np.pi*(D_i+2*h_slot))/36 - tooth_width # RADIALLY OUTER WIDTH OF SLOT

        Tau_y = np.pi*(D_i+h_slot) / (2*p)
        L_j1 = np.pi*(outer_stator_radius-h_ys) / (4*p) # STATOR YOKE LENGTH FOR MAGNETIC CIRCUIT CALCULATION
        A_slot = (b_s1+b_s2)*(h_slot-h_k-h_os)/2

        # --- WINDING FACTOR ---
        Kp1 = csdl.sin(pole_pitch*90*np.pi/pole_pitch/180) # INTEGRAL WINDING

        alpha = 360*p/Z # ELECTRICAL ANGLE PER SLOT
        Kd1 = np.sin(q*alpha/2)/(q*np.sin(alpha/2))
        Kdp1 = Kd1*Kp1

        # --- MAGNET GEOMETRY ---
        hm = 0.004 # MAGNET THICKNESS
        theta_p = 360/2/p # ANGULAR SWEEP OF POLE IN DEGREES
        theta_m = 0.78*theta_p
        Dm = rotor_radius - 0.002
        bm = Dm*np.pi*theta_m/360

        T = 75 # assuming normal operating temp
        Br_20 = 1.2 # Br at 20 C
        alpha_Br = -0.12 # temperature coefficients
        IL = 0 # revercible loss
        Br = 1+(T-20)*alpha_Br/100*(1-IL/100)*Br_20

        Hc_20 = 907000; # coercivity
        Hc = (1+(T-20)*alpha_Br/100)*(1- IL/100)*Hc_20
        mu_r = Br/mu_0/Hc; # relative permeability

        # Br = 1.2 # MAGNET REMANENCE
        # Hc = 907000. # MAGNET COERCIVITY

        mu_r = Br/(mu_0*Hc) # RELATIVE MAGNET PERMEABILITY

        Am_r = bm*l_ef # RADIAL CROSS SECTIONAL AREA OF MAGNET
        rho_magnet = 7.6 # MAGNET DENSITY (g/cm^3)
        mass_magnet = 2*p*bm*hm*l_ef*rho_magnet*1e3 # MAGNET MASS

        phi_r = 1.2*Am_r
        Fc = 2*Hc*hm

        lambda_m = phi_r/Fc
        alpha_p1 = bm/pole_pitch
        alpha_i = alpha_p1+4/((pole_pitch/air_gap_depth)+(6/(1-alpha_p1)))

        Kf = 4*csdl.sin(alpha_i*np.pi/2)/np.pi # COEFF OF MAGNETIC FLUX DENSITY ALONG AIR GAP
        K_phi = 8.5*csdl.sin(alpha_i*np.pi/2)/(np.pi**2*alpha_i) # COEFF OF FLUX ALONG AIR GAP
        K_theta1 = tooth_pitch*(4.4*air_gap_depth + 0.75*b_sb)/(tooth_pitch*(4.4*air_gap_depth + 0.75*b_sb)-b_sb**2)
        K_theta2 = 1 # no rotor slot

        K_theta = K_theta1*K_theta2
        l_f2 = hm
        A_f2 = l_f2*l_ef

        # --- RESISTANCE & MASS CALCULATION
        rho = 0.0217e-6 # RESISTIVITY ------ GET CLARIFICATION ON UNITS
        l_B = l_ef + 2*0.01 # straight length of coil
        l_coil = l_B + 2.0*pole_pitch # length of half-turn
        
        Rdc = self.register_output(
            'Rdc',
            2 * rho * turns_per_phase * l_coil / \
            (a * Acu * N_p) 
        ) # DC RESISTANCE

        if False:
            delta = (rho/(np.pi*mu_0*f_i)) ** 0.5

            Rac = self.register_output(
                'Rac',
                Rdc / ((2*delta/d_coil) - (delta/d_coil)**2)
            ) # IGNORING FOR NOW, MAY USE LATER

        C = 1.05
        rho_cu = 8.9 # mass density of copper in g/cm^3
        mass_cu = C*l_coil*conductors_per_slot*Z*Acu*rho_cu*1e3 # mass of copper

        rho_fe = 7.8 # mass density of iron in g/cm^3
        mass_deficit_slot = A_slot*l_ef*rho_fe*Z*1e3
        mass_deficit_mag = 2*p*bm*hm*l_ef*rho_fe*1e3

        motor_mass = self.register_output(
            'motor_mass', 
            (mass_cu-mass_deficit_slot) + (mass_magnet-mass_deficit_mag) + \
            np.pi*l_ef*rho_fe*((outer_stator_radius/2)**2 - (D_shaft/2)**2)*1e3
        )

        self.add(
            TorqueMassModel(fitting_order=1),
            'max_torque_model'
        ) # GIVES US T_em_max AS A FUNCTION OF MASS
        max_torque = self.declare_variable('T_em_max')

        # POPULATE motor_parameters WITH THE RELEVANT MOTOR VARIABLES
        # THE ONLY ONES THAT WILL BE IGNORED ARE motor_mass AND resistance (Rdc)
        motor_parameters = self.create_output(
            'motor_parameters',
            shape=(27,),
            val=0.
        )
        
        motor_parameters[0]  = outer_stator_radius
        motor_parameters[1]  = pole_pitch
        motor_parameters[2]  = tooth_pitch
        motor_parameters[3]  = air_gap_depth
        motor_parameters[4]  = l_ef
        motor_parameters[5]  = rotor_radius
        motor_parameters[6]  = turns_per_phase
        motor_parameters[7]  = Acu
        motor_parameters[8]  = tooth_width
        motor_parameters[9]  = h_ys
        motor_parameters[10] = b_sb
        motor_parameters[11] = h_slot
        motor_parameters[12] = b_s1
        motor_parameters[13] = Tau_y
        motor_parameters[14] = L_j1
        motor_parameters[15] = Kdp1
        motor_parameters[16] = bm
        motor_parameters[17] = Am_r
        motor_parameters[18] = phi_r
        motor_parameters[19] = lambda_m
        motor_parameters[20] = alpha_i
        motor_parameters[21] = Kf
        motor_parameters[22] = K_phi
        motor_parameters[23] = K_theta
        motor_parameters[24] = A_f2
        motor_parameters[25] = Rdc
        motor_parameters[26] = max_torque

        # MOTOR CG AND INERTIA COMPUTATION
        units = 'ft' # CHANGE LATER TO MAKE MORE GENERAL
        if units == 'ft':
            origin = self.declare_variable(f'motor_origin', shape=(3,)) * 0.3048
        else:
            origin = self.declare_variable(f'motor_origin', shape=(3,))

        x, y, z = origin[0], origin[1], origin[2]

        motor_cg = self.register_output(f'motor_cg', origin * 1.)

        ixx = motor_mass * (y**2 + z**2)
        ixy = -motor_mass * x*y
        ixz = -motor_mass * x*z 
        iyx = ixy*1.
        iyy = motor_mass * (x**2 + z**2) 
        iyz = -motor_mass * y*z 
        izx = ixz * 1.
        izy = iyz * 1.
        izz = motor_mass * (x**2 + y**2)

        inertia_tensor = self.create_output(f'motor_inertia', shape=(3,3), val=0.)
        inertia_tensor[0, 0] = csdl.reshape(ixx, (1, 1))
        inertia_tensor[0, 1] = csdl.reshape(ixy, (1, 1))
        inertia_tensor[0, 2] = csdl.reshape(ixz, (1, 1))
        inertia_tensor[1, 0] = csdl.reshape(iyx, (1, 1))
        inertia_tensor[1, 1] = csdl.reshape(iyy, (1, 1))
        inertia_tensor[1, 2] = csdl.reshape(iyz, (1, 1))
        inertia_tensor[2, 0] = csdl.reshape(izx, (1, 1))
        inertia_tensor[2, 1] = csdl.reshape(izy, (1, 1))
        inertia_tensor[2, 2] = csdl.reshape(izz, (1, 1))

        self.register_output('mass', motor_mass * 1.)
        self.register_output('cg_vector', motor_cg * 1.)
        self.register_output('inertia_tensor', inertia_tensor * 1.)



if __name__ == '__main__':
    m = TC1MotorSizingModel(
        pole_pairs = 6,
        phases=3,
        num_slots=36,
        rated_current=123
    )

    D_i = 0.182
    L = 0.086

    # D_i = 0.3723
    # L = 0.2755

    rep = GraphRepresentation(m)
    sim = Simulator(rep)
    sim['motor_diameter'] = D_i
    sim['motor_length'] = L

    sim.run()
    print(sim['Rdc'])
    print(sim['motor_parameters'])
    print([
            'outer_stator_radius', 'pole_pitch', 'tooth_pitch', 'air_gap_depth', 'l_ef',
            'rotor_radius', 'turns_per_phase', 'Acu',  'tooth_width', 'height_yoke_stator',
            'slot_bottom_width', 'slot_height', 'slot_width_inner', 'Tau_y', 'L_j1', 'Kdp1',
            'bm', 'Am_r', 'phi_r', 'lambda_m', 'alpha_i', 'Kf', 'K_phi', 'K_theta', 'A_f2',
        ])
