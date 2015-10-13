# This file contains useful class definitions

import numpy as np
from functions import *
from collections import namedtuple



class ICE:
    """A class to represent an internal combustion engine (ICE)"""
    def __init__(self, a0=8.4973e-8, nr_TdICE1_cons=2, TdICE1_max=1.0e+03*np.array([1.823029420753276,2.197838461538461,2.197838461538461]), Ek_TdICE1_max=1.0e+07*np.array([0.931383297138416,1.001985072653245,1.606950281826688]), gear_ratios=np.array([[0.785256410256410],[1.000000000000000],[1.281250000000000],[1.631632653061224],[2.118421052631579],[2.697744360902256],[3.414158305462653],[4.347826086956522],[5.570652173913043],[7.094055013309671],[9.210526315789473],[11.729323308270677]]), widle=68.067840827778852, wmax=30*2*np.pi, J=3.8, marginal_engine_efficiency=0.5, average_engine_efficiency_at_reduced_gear = 0.3): 
        self.a0 = a0
        self.nr_TdICE1_cons = nr_TdICE1_cons
        self.TdICE1_max = TdICE1_max
        self.Ek_TdICE1_max = Ek_TdICE1_max
        self.gear_ratios = gear_ratios
        self.widle = widle
        self.wmax = wmax
        self.J = J
        self.marginal_engine_efficiency = marginal_engine_efficiency
        self.average_engine_efficiency_at_reduced_gear = average_engine_efficiency_at_reduced_gear



class EM: 
    """A class to represent an electric machine (EM)"""
    def __init__(self, r_em=3.3654, wmax=200*2*np.pi, J=0.054):
        self.ice          = ICE()          # only used to define gears
        self.r_em         = r_em  
        self.gear_ratios  = self.r_em * self.ice.gear_ratios
        self.wmax         = wmax           # max em speed [rad/s] (em -- electric machine)
        self.J            = J              # moment of inertia of electric machine [kg*m^2]
    


class Truck: 
    """A class to represent instances of heavy-duty trucks in the combined platooning and energy-management problem in the SHC project""" 
    def __init__(self, isHEV=False, m=40235.0, P_aux_m=3000.0, rw = 0.4910, v_min=8.3333, eta=0.9819, derate_max_pwr_demand=0.85, max_pwr_demand=300000.0, r_t=2.6429, A=9.7, Cd=0.53, cr=0.0047, trq_ret_max=4.0e3, Pd_max=300.0e3, under_speed_tolerance=8.0/3.6, over_speed_tolerance=8/3.6, acc_min=-2, total_wheel_and_driveline_inertia=12.189224251278306): 
        self.isHEV                             = isHEV             # Bool that determines if the truck is a hybrid electric vehicle or a conventional truck.
        self.m                                 = m                 # vehicle mass [kg]
        self.m_eff                             = self.m + 63.0     # effective mass at cruise gear (rotary parts considered) [kg]
        self.P_aux_m                           = P_aux_m           # mechanically coupled auxiliary power that is drawn from engine [W]
        self.rw                                = rw                # wheel radius [m]
        self.ice                               = ICE()  
        #self.ice.a0                            = 8.4973e-8          # coefficient in affine model for friction torque Tfric = TdICEfric0 + a0 * Ek
        self.ice.TdICEfric0                    = 118.8495           # coefficient in affine model for friction torque [Nm]
        self.v_min                             = v_min              # minimum speed allowed for cruise controller [m/s]
        self.eta                               = eta
        self.derate_max_pwr_demand             = derate_max_pwr_demand
        self.max_pwr_demand                    = max_pwr_demand     # max power for engine
        self.r_t                               = r_t                # transmission ratio
        self.A                                 = A                  # cross-sectional area
        self.Cd                                = Cd                 # air drag coefficient
        self.CdA                               = self.A*self.Cd     # air drag coefficient times cross-sectional area of vehicle
        self.cr                                = cr                 # rolling resistance coefficient
        self.trq_ret_max                       = trq_ret_max        # max retarder torque [Nm]
        self.Pd_max                            = Pd_max             # maximum total traction power at gearbox output [W]
        self.r_ice                             = self.ice.gear_ratios[0]  # Transmission ratio for cruise gear
        self.em                                = EM()
        
        self.under_speed_tolerance             = under_speed_tolerance          # [m/s]
        self.over_speed_tolerance              = over_speed_tolerance           # [m/s]
        self.acc_min                           = acc_min                 # [m/s^2]  (max deceleration)
        # self.ice.widle                         = 68.067840827778852 # idling speed [rad/s]
        # self.ice.wmax                          = 30*2*pi            # max ice speed [rad/s]
        
        self.total_wheel_and_driveline_inertia = total_wheel_and_driveline_inertia



class Environment: 
    """A class to setup a standard environment for the combined platooning and energy-management problem etc acceleration, etc.)"""
    def __init__(self): 
        self.g = 9.81  # [m/s**2],  gravitational acceleration
        self.airdens = 1.2  # [kg/m**3],  density of air



class Platoon: 
    """A class to setup a platoon for the combined platooning and energy-management problem"""
    def __init__(self, number_of_trucks=3, v_avg=70.0/3.6): 
        self.trucks = [] # a list of trucks
        for i in range(number_of_trucks):
            self.trucks.append(Truck())  
        self.v_avg = v_avg

    def get_number_of_trucks(self): 
        """Return the number of trucks for the platoon"""
        return len(self.trucks)



class DriveCycle: 
    """A class to represent a drive cycle for the combined platooning and energy-management problem"""
    def __init__(self, length=5e3, sd=40.0, max_alt_shorthill=20.0, shorthill_len=500.0, max_alt_longhill=80.0, longhill_len=2000.0): 
        self.length = length               # length of drive cycle [m]
        self.sd = sd                       # sample distance
        self.max_alt_shorthill = max_alt_shorthill   # maximum altitude change for a short hill
        self.shorthill_len = shorthill_len      # length of a short hill
        self.max_alt_longhill = max_alt_longhill
        self.longhill_len = longhill_len
        self.distance = linspace(0, self.length, self.length/self.sd)
        self.create_random_drive_cycle()

    def create_random_drive_cycle(self): 
        """Create and return a drive cycle"""

        nr_shorthills = round(self.length/self.shorthill_len)
        alt_distance_vector1 = linspace(0, self.length, nr_shorthills)
        alt_vector1 = self.max_alt_shorthill * np.random.rand(nr_shorthills)
        altitude1 = interp1(alt_distance_vector1, alt_vector1, self.distance, 'pchip')
        
        nr_longhills = round(self.length/self.longhill_len)
        alt_distance_vector2 = linspace(0, self.length, nr_longhills)
        alt_vector2 = self.max_alt_longhill*np.random.rand(nr_longhills)
        altitude2 = interp1(alt_distance_vector2, alt_vector2, self.distance, 'pchip')
        
        # superimpose and remove affine trend
        altitude = altitude1+altitude2
        k_alt = (altitude[-1]-altitude[0]) / self.length
        altitude = altitude - k_alt*self.distance
        altitude = altitude - altitude[0]
        # calculate slope under assumption that slope is small (i.e., tan(a/b) \approx a/b)
        slope = np.append(diff(altitude)/self.sd, np.array([0]))
        
        self.pos   = self.distance
        self.slope = slope
        self.altitude = altitude

