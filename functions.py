# This file contains useful function definitions

import numpy as np
from scipy.interpolate import pchip
import matplotlib.pyplot as plt

## First some Matlab equivalents
def length(a): 
    return np.amax(a.shape)

def find(a): 
    return np.where(a)[0]

def max(a):
    return np.amax(a)

def min(a):
    return np.amin(a)

def linspace(start, stop, nsamples):
    return np.linspace(start, stop, num=nsamples)

def diff(a): 
    return np.diff(a)

def round(a): 
    return np.round(a)

def rand(a=-1, b=-1):
    if a==-1:
        return np.random.rand()
    elif b==-1:
        return np.random.rand(a, a)
    else:
        return np.random.rand(a, b)

def myplot(x,y):
    plt.figure(1)
    plt.clf()
    plt.plot(x,y)
    plt.grid()
    plt.show(block=False)



def interp1(x,y,xi,method='linear'):
    if method=='linear':
        return np.interp(np.squeeze(xi), np.squeeze(x), np.squeeze(y))
    elif method=='pchip':
        interp = pchip(np.squeeze(x),np.squeeze(y))
        return interp(np.squeeze(xi))


def prefilter_vset(truck, env, v_init, pos, slope, v_set, v_set_front=200.0/3.6, t_gap_init=0, t_gap_safety=-1):
    # function [v_set_filt,v_ub,v_lb,Td] = prefilter_vset(truck, env, v_init, pos, slope, v_set, v_set_front, t_gap_init, t_gap_safety)
    """Find feasible speed trajectory (no vehicle in front) as well as required torque trajectory that gives desired speed trajectory
    
    Author: magnus.nilsson@viktoria.se
    Date: Friday, February 20, 2015
    """
    
    N          = length(pos)
    sd         = np.zeros([N,])
    dpos       = diff(pos)
    sd[0:N-1]  = dpos
    #sd[N]      = sd[N-1]
    time_front = np.cumsum(sd/v_set_front)
    
    
    # v_set_filt is forced to never go below v_min.
    
    time = np.zeros(slope.shape)
    time[0] = t_gap_init
    
    # Shorthand notations
    rw       = truck.rw
    r_t      = truck.r_t
    CdA      = truck.CdA
    v_min    = truck.v_min
    r_ice    = truck.ice.gear_ratios
    r_em     = truck.em.gear_ratios
    r_ice_em = r_ice/r_em
    
    v_set[v_min > v_set] = v_min
    v_init = max(np.array([v_init, v_min]))
    

    v_set_filt = np.zeros(length(pos),)
    v_set_filt[0] = v_init
    
    
    Td = np.zeros(N-1,)  # The propulsion torque required to follow the set speed.
    
    
    v_lb = v_set - truck.under_speed_tolerance
    #Check if the speed is close enough to the set_speed, if not continue to accelerate/decelerate towards set speed
    for pos_i in range(N-1):
        t_s = (pos[pos_i+1] - pos[pos_i]) / v_set_filt[pos_i]
        time[pos_i+1] = time[pos_i] + t_s
        
        spd_out = r_t * v_set_filt[pos_i] / rw # speed at gearbox output
        spd_ice = r_ice * spd_out
        spd_em = r_em * spd_out
        feasible_gear_index = find((truck.ice.widle <= spd_ice) & (spd_ice <= truck.ice.wmax) & (spd_em <= truck.em.wmax))
        spd_ice_s = max(spd_ice[feasible_gear_index]) # inertia torque of the engine and motor are calculated based on the assumption that the engine is at high rpm (delivering max power)
        gear_index = np.argmax(spd_ice[feasible_gear_index])
        trq_out_max = truck.derate_max_pwr_demand * truck.max_pwr_demand / spd_out

        r_ice_s = r_ice[feasible_gear_index[gear_index]]
        r_em_s = r_em[feasible_gear_index[gear_index]]
        
        m_eff = truck.m + (truck.em.J * r_em_s**2 + truck.ice.J * r_ice_s**2 + truck.total_wheel_and_driveline_inertia) / (rw**2)
        acc = (trq_out_max / rw * r_t - (truck.cr*truck.m * env.g * np.cos(slope[pos_i]) + 0.5 * (v_set_filt[pos_i]**2) * env.airdens * CdA + truck.m * env.g * np.sin(slope[pos_i]))) / m_eff
        
        # Force vehicles to stay safe distance apart
        if time[pos_i+1] <= time_front[pos_i+1] + t_gap_safety:
            v_set_next = v_set_front[pos_i+1]
        else:
            v_set_next = v_set[pos_i+1]

        max1 = max(np.array([v_set_next, t_s * truck.acc_min + v_set_filt[pos_i]]))
        npa1 = np.array([v_set_filt[pos_i] + acc * t_s, max1])
        min1 = min(npa1)
        v_set_filt[pos_i+1] = max(np.array([v_min, min1]))
        
        if v_lb[pos_i] >= v_set_filt[pos_i] - truck.under_speed_tolerance*0.1:
            v_lb[pos_i] = v_set_filt[pos_i] - truck.under_speed_tolerance*0.1

    if v_lb[N-1] >= v_set_filt[N-1] - truck.under_speed_tolerance*0.1:
        v_lb[N-1] = v_set_filt[N-1] - truck.under_speed_tolerance*0.1



    v_ub = v_set_filt + truck.over_speed_tolerance


    Ek = 0.5* truck.m_eff * (v_set_filt**2);
    Fa = Ek * env.airdens * truck.CdA / truck.m_eff
    Fr = truck.cr * truck.m * env.g * np.cos(slope[0:N])
    Fg = truck.m * env.g * np.sin(slope[0:N])
    dEosd = diff(Ek) / sd[0:N-1]
    Ftot = Fa + Fr + Fg
    Td = (dEosd + Ftot[0:N-1]) / (r_t/rw)
    Td = np.append(Td, Td[-1])
    #Td = Td(:); # return a column vector
    
    return v_set_filt, v_ub, v_lb, Td


def calc_aff_pars(speed=np.array([40.0/3.6, 50.0/3.6, 60.0/3.6, 70.0/3.6, 80.0/3.6]), speed_buffer_size=10.0/3.6, sd=40.0):
    """Calculate parameters in affine approximation of time--distance relation
     
       t_approx = A0 + A1 * v^2.
     
           ^
           |
           |          .
           |          ...
           |            ...
           |               .
       A0  +--              ..
           |  \----          ...
           |       \----       ..
           |            \---    ....
           |                \----  .....
           |                     \---- .....
           |                          \---- ....
           |                               \---.....
           |                                    \---........
           |                                         \---   ............ ......      s / v
           |                                             \----                 ..............
           |                                                  \----
           |                                                       \--
           |                            :           :                 A0 + A1 * v^2
           |                            :           :
           |                            :           :
           |                            :           :
           |                            :           :
           +----------------------------:-----------:---------------------------------------------->  v^2
                                         interesting
                                         spd interval = [speed-speed_buffer_size/2, speed+speed_buffer_size/2]
     
      Author: magnus.nilsson@viktoria.se
        Date: Wednesday, February 18, 2015"""

    N = 100

    Nv = length(speed)
    a0_vec = np.zeros(Nv)
    a1_vec = np.zeros(Nv)
    for idx in range(Nv):

        spd = speed[idx] # [m/s]
        spd_min = spd - speed_buffer_size/2
        spd_max = spd + speed_buffer_size/2

        v = linspace(spd_min, spd_max, N)
        vsq = v**2
        t_true = sd/v

        # Solve a simple least squares problem to approximate curve with a line
        A = np.vstack([np.ones(v.shape), vsq]).T
        b = t_true
        x = np.linalg.lstsq(A,b)[0] # A\b in Matlab notation
 
        a_0 = x[0]
        a_1 = x[1]
        
        a0_vec[idx] = a_0
        a1_vec[idx] = a_1
        
    return a0_vec, a1_vec
