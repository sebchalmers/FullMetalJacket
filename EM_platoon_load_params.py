# This script is a helper script for EM_platoon.py and contains
# numerical values for different properties and variables that are
# used as inputs for the optimization problem
#
# This script can but is not intended to be run on its own. It is
# executed within EM_platoon.py


#from casadi import *
#from casadi.tools import *  # For struct_symSX for example
import numpy as NP
from scipy import interpolate
from matplotlib.pyplot import plot, draw, show, ion
from classes import *

#
# Initiate instances of useful objects and define values for parameters in the problem
#
nt = 2                 # number of trucks

dc  = DriveCycle()     # dc -- drive cycle
sd  = dc.sd            # sd -- sample distance
trk = range(nt)
for i in range(nt):
    trk[i] = Truck()

env = Environment()

nk = length(dc.pos)    # number of samples in the drive cycle


# Decide scaling parameters for the optimization problem, to constrain
# the decision variables in the same order of magnitude; around
# 1. This is in general useful to avoid numerical problems in
# optimization solvers.
sobj = 1.0e6    # scaling parameter for cost function
sTd  = 1.0e3    # scaling parameter for torque
sEk  = 1.0e6    # scaling parameter for kinetic energy


# Calculate parameters in affine approximation of time--kinetic-energy relationship
speeds = NP.linspace(40/3.6, 90/3.6, num=20) # [m/s]
speed_buffer_size = 16/3.6                   # [m/s] (+-8 km/h generic speed buffer size for calc_aff_pars)
a0_vec, a1_vec = calc_aff_pars(speeds, speed_buffer_size, sd)


## Generate a feasible speed trajectory for each vehicle, taking the
## vehicle in front into consideration.

# Declare memory for variables
v_set      = NP.zeros([nk, nt])
v_set_filt = NP.zeros([nk+1, nt])
t_set_filt = NP.zeros([nk, nt])
v_ub       = NP.zeros([nk, nt])
v_lb       = NP.zeros([nk, nt])
Td         = NP.zeros([nk, nt])
t_allowed_lag_from_v_set_filt = NP.zeros(nt)  # [s]
t_gap_safety = NP.zeros(nt)  # [s]
t_init = NP.zeros(nt)

# First vehicle
t_allowed_lag_from_v_set_filt[0] = 5  # [s]
t_init[0] = 0                                      # initial time
v_set[0:nk, 0] = 80 / 3.6 * NP.ones(dc.pos.shape)  # reference/set speed
v_init = v_set[0, 0]  # initial speed
v_set_filt[0:nk,0], v_ub[0:nk,0], v_lb[0:nk,0], Td[0:nk,0] = prefilter_vset(trk[0], env, v_init, dc.pos, dc.slope, v_set[0:nk,0]) # note: only 6 arguments when no vehicle in front
v_set_filt[nk,0] = v_set_filt[nk-1,0]


# Remaining vehicles
for k in range(nt):
    t_allowed_lag_from_v_set_filt[k] = 10
    t_gap_safety[k] = 0.5  # safe distance to truck in front [s]
    t_init[k] = t_init[k-1] + 1.2*t_gap_safety[k]
    v_set[0:nk, k] = 80 / 3.6 * NP.ones(dc.pos.shape)
    v_init = v_set[0, k]  # + speed_buffer_size * (rand - 0.5);
    v_set_filt[0:nk, k], v_ub[0:nk, k], v_lb[0:nk, k], Td[0:nk, k] = prefilter_vset(trk[k],
                                                                                env,
                                                                                v_init, 
                                                                                dc.pos, 
                                                                                dc.slope, 
                                                                                v_set[0:nk, k], 
                                                                                v_set_filt[0:nk, k-1],  # 7th argument is v_set for vehicle in front
                                                                                t_init[k] - t_init[k-1],
                                                                                t_gap_safety[k])

# Calculate corresponding time vectors for v_set_filt
for k in range(nt):
    t_set_filt[0:nk,k] = NP.cumsum(sd / v_set_filt[0:nk,k]) - sd/v_set_filt[0,k] + t_init[k]


Tdeps = 10   # Hmm, what is this for?


P_aux_m = NP.zeros(nt)
A = NP.zeros(nt)
Cd = NP.zeros(nt)
CdA = NP.zeros(nt)
m_eff = NP.zeros(nt)
r_ice = NP.zeros(nt)
rw = NP.zeros(nt)
r_t = NP.zeros(nt)
cr = NP.zeros(nt)
a0 = NP.zeros(nt)
TdICEfric0 = NP.zeros(nt)
Ek0 = NP.zeros(nt)
trq_ret_max = NP.zeros(nt)
Ek_set = NP.zeros([nk+1, nt])
Ek_min = NP.zeros([nk+1, nt])
Ek_max = NP.zeros([nk+1, nt])
a00 = NP.zeros([nk+1, nt])
a01 = NP.zeros([nk+1, nt])
Pd_max = NP.zeros(nt)
Td_max0 = NP.zeros([nk,nt])
k_Td = NP.zeros([nk,nt])
nr_TdICE1_cons = NP.zeros(nt)
TdICE1_max = NP.zeros([len(trk[0].ice.TdICE1_max),nt])
Ek_TdICE1_max = NP.zeros([len(trk[0].ice.TdICE1_max),nt])
# Shorthand notations for cleaner code
for k in range(nt):
    P_aux_m[k]          = trk[k].P_aux_m                                          # Mechanically coupled auxiliary power that is drawn from engine
    A[k]                = trk[k].A
    Cd[k]               = trk[k].Cd
    CdA[k]              = trk[k].CdA                                              # Air drag coefficient times cross-sectional area of vehicle
    m_eff[k]            = trk[k].m_eff                                            # Effective mass (considers inertia of rotating parts)
    r_ice[k]            = trk[k].r_ice                                            # Transmission ratio for
    rw[k]               = trk[k].rw                                               # Wheel radius
    r_t[k]              = trk[k].r_t                                              # Transmission ratio
    cr[k]               = trk[k].cr                                               # Rolling resistance coefficient
    a0[k]               = trk[k].ice.a0                                           # Parameter in affine approximation of friction torque
    TdICEfric0[k]       = trk[k].ice.TdICEfric0                                   # Parameter in affine approximation of friction torque
    Ek0[k]              = 0.5*m_eff[k]*v_set_filt[0,k]**2                                # Ek0 is the initial kinetic energy
    trq_ret_max[k]      = trk[k].trq_ret_max
    # Ek1            = 0.5*m_eff*v_set(end)**2 # Ek0 is the initial kinetic energy
    Ek_set[0:nk+1,k]    = 0.5 * m_eff[k] * v_set_filt[:,k]**2                           # Ek_set is the kinetic energy on v_set
    Ek_min[0:nk,k]      = 0.5 * m_eff[k] * v_lb[:,k]**2                            # Ek_min is the lower bound of the kinetic buffer
    Ek_min[nk,k]        = Ek_min[nk-1,k]
    Ek_max[0:nk,k]      = 0.5 * m_eff[k] * v_ub[:,k]**2                            # Ek_max is the upper bound of the kinetic buffer
    Ek_max[nk,k]        = Ek_max[nk-1,k]
    a00[:,k]            = interp1(speeds, a0_vec, NP.sqrt(2 * Ek_set[:,k] / m_eff[k])) # Coefficients for affine relation ts = a0 + a1 * (2*Ek_set/m_eff)
    a01[:,k]            = interp1(speeds, a1_vec, NP.sqrt(2 * Ek_set[:,k] / m_eff[k])) # Coefficients for affine relation ts = a0 + a1 * (2*Ek_set/m_eff)
    Pd_max[k]           = trk[k].max_pwr_demand                                   # Pd_max is maximum total traction power at gearbox output. Non-convex constrain when working in distance, so we linearize around Ek_set.
    
    #  t_approx = A0 + A1 * v^2.
    #
    #      ^
    #      |
    #      |          .
    #      |          ...
    #      |            ...
    #      |               .
    #  A0  +--              ..
    #      |  \----          ...
    #      |       \----       ..
    #      |            \---    ....
    #      |                \----  .....
    #      |                     \---- .....
    #      |                          \---- ....
    #      |                               \---.....
    #      |                                    \---........
    #      |                                         \---   ............ ......      s / v
    #      |                                             \----                 ..............
    #      |                                                  \----
    #      |                                                       \--
    #      |                                :                         A0 + A1 * v^2
    #      |                                :
    #      |                                :
    #      |                                :
    #      |                                :
    #      +--------------------------------:------------------------------------------------------>  v^2
    #
    #
    Td_max0[0:nk,k] = (NP.sqrt(2)*Pd_max[k]*rw[k]) / (2*r_t[k]*NP.sqrt(Ek_set[0:nk,k]/m_eff[k])) \
                  + (NP.sqrt(2)*Pd_max[k]*rw[k]*Ek_set[0:nk,k]) / (4*Ek_set[0:nk,k]*r_t[k]*NP.sqrt(Ek_set[0:nk,k]/m_eff[k])) \
                  + Tdeps
    k_Td[0:nk,k]    = -(NP.sqrt(2)*Pd_max[k]*rw[k]) / (4*Ek_set[0:nk,k] * r_t[k] * NP.sqrt(Ek_set[0:nk,k] / m_eff[k]))
    nr_TdICE1_cons[k] = trk[k].ice.nr_TdICE1_cons                               # Number of affine inequalities to use for piecewise affine approximation of max torque curve
    TdICE1_max[:,k]     = trk[k].ice.TdICE1_max                                   # Points in range of piecewise approximation of max torque curve
    Ek_TdICE1_max[:,k]  = trk[k].ice.Ek_TdICE1_max                                # Points in domain of piecewise approximation of max torque curve
    

#   # A comment on convex approximation of maximum torque map
#
#   ^ Td (torque)
#   |
#   |            /
#   |           /
#   |          /              (Ek3,Td3)
#   | --------+--------------+-----
#   |        / (Ek2,Td2)
#   |       /
#   |      /
#   |     /
#   |    /
#   |   + (Ek1, Td1)
#   |  /
#   | /              (feasible set below lines)
#   |/
#   |
#   +------------------------------------------------> Ek (kinetic energy)
#
#   The equation for a line, Td = k * Ek + m, can be expressed
#   given two pairs on the line, (Ek1,Td1), (Ek2,Td2), with
#
#   k = (Td2-Td1)/(Ek2-Ek1);   m = Td1 - k*Ek1
#
#   The following for loop expresses a feasible set for TdICE1
#   using such relations for a piecewise affine approximation
#   of the maximum torque curve on cruise gear.
#
k_EkTd = NP.zeros([nr_TdICE1_cons[0], nt])
m_EkTd = NP.zeros([nr_TdICE1_cons[0], nt])
for kt in range(nt):
    for kk in range(int(nr_TdICE1_cons[kt])):  # For the number of pieces in piecewise approximation of max torque
        k_EkTd[kk, kt] = (TdICE1_max[kk+1, kt] - TdICE1_max[kk, kt])/(Ek_TdICE1_max[kk+1, kt] - Ek_TdICE1_max[kk, kt])
        m_EkTd[kk, kt] = TdICE1_max[kk, kt] - k_EkTd[kk, kt] * Ek_TdICE1_max[kk, kt]


#In CVX:  Td<=(Td_max0(1:nk)+k_Td(1:nk).*Ek(1:nk));
#Linearization in Ek around Ek_set results in

# Coefficients for pareto-optimality (i.e., weights given to different and possibly conflicting desires)
c0 = NP.zeros(nt)
c1 = NP.zeros(nt)
c2 = NP.zeros(nt)
for k in range(nt):
    c0[k] = r_t[k]/rw[k]/trk[k].ice.marginal_engine_efficiency                     # Penalty associated with
    c1[k] = r_t[k]/rw[k]/trk[k].ice.average_engine_efficiency_at_reduced_gear      # Penalty associated with
    c2[k] = 1                                                             # Penalty factor on mechanical brake energy

Fr = NP.zeros([nk, nt])
Fg = NP.zeros([nk, nt])
Fr_plus_Fg_num = NP.zeros([nk, nt])
for k in range(nt):
    Fr[0:nk, k] = cr[k] * m_eff[k] * env.g * NP.cos(dc.slope[0:nk]) # Rolling resistance, vector version
    Fg[0:nk, k] = m_eff[k] * env.g * NP.sin(dc.slope[0:nk])          # Force due to gravity, vector version
    Fr_plus_Fg_num[0:nk, k] = Fr[0:nk, k] + Fg[0:nk, k]


slack_penalty_TdICE1 = 10.0e3   # Large value so that slack is only used when forced to
