# This script sets up the energy-management problem for a truck platoon
#

# Requirements:
# To run this script, you need at least the following programs installed on your machine: 
# Python, IPOPT, CasADi

from casadi import * 
from casadi.tools import *  
import numpy as NP
from scipy import interpolate
from matplotlib.pyplot import plot, draw, show, ion
from classes import *

# TODO: free position of the followers, force the separation in the beginning and the end to match

#
# Load parameters for EM problem (cleaner code to separate it out from this script)
# 


class SetPlatoon:
    def __init__(self, Ntruck = 2, NFourrierCoeff = 0, nstep = 2, nk = 540):

        env            = {'airdens' : 1.2, 'g' : 9.81}
        
        DragRedSlope   = [0., -0.45, -0.48]
        DragRedSlope  += (Ntruck-3)*[-0.48]
        DragRedOffset  = [0., 43, 52]
        DragRedOffset += (Ntruck-3)*[52]
    
        #Define trucks constants and assign default values
        self.Constants = {'c1':  Ntruck*[17.94229464],
                          'c2':  Ntruck*[1.] ,
                          'cr':  Ntruck*[0.0047],
                          'rw':  Ntruck*[0.491],
                          'r_t': Ntruck*[2.6429],
                          'CdA': Ntruck*[5.141],
                          'm':   Ntruck*[40298.],
                          'DragRedSlope' : DragRedSlope,
                          'DragRedOffset': DragRedOffset}

        #Define platoon constants and assign default values
        self.ProblemParameters      = {    'Sin'            : [0. for i in range(NFourrierCoeff)],
                                           'Cos'            : [0. for i in range(NFourrierCoeff)],
                                           'Distance'       : 3e4,
                                           'MaxVel'         : 120.,
                                           'MinVel'         : 80.,
                                           'MinSeparation'  : 10.,
                                           'MaxTorque'      : 5e3,
                                           'MaxTime'        : 1080.}
        
        self.Ntruck         = Ntruck
        self.NFourrierCoeff = NFourrierCoeff
        
        # Numerical parameters
        self.nk       = nk
        self.nstep    = nstep

        # Dictionary for collecting solvers outputs
        self.SolversOut = {'Cost':[], 'Status':[]}

        # Plot parameters
        self._ScalePlot = {'pos':1e-3,'vel':3.6}

        #
        # Create basic variables
        #

        self.states = struct_symSX([
                               entry("pos"),
                               entry("vel")
                               ])

        self.inputs = struct_symSX([
                               entry("TdE"),
                               entry("Tdbrk")
                               ])

        parameters_list = [ entry(i) for i in self.Constants.keys()]
        parameters_list.append( entry('Sin',  repeat = NFourrierCoeff ) )
        parameters_list.append( entry('Cos',  repeat = NFourrierCoeff ) )
        parameters_list.append( entry('Distance')) #: Ntruck*[3e4]
        

        
        self.parameters = struct_symSX( parameters_list )


        
        #
        # Truck variables
        #


        self._AllPos_struct =  struct_symSX([
                                        entry('pos', repeat = self.nstep)
                                       ])


        self.Vtruck = struct_symSX([
                                        entry('State',     struct=self.states,      repeat=self.nk+1),
                                        entry('Input',     struct=self.inputs,      repeat=self.nk),
                                        entry('Tf')
                                    ])

        self.Ptruck = struct_symSX([
                                        entry('Parameter',         struct = self.parameters ),
                                        entry('PosLeader',         struct = self._AllPos_struct, repeat=self.nk+1),  #OBS: used only in greedy optimization, disregarded in full platoon
                                        entry('HazLeader')                                               #OBS: used only in greedy optimization, disregarded in full platoon
                                    ])

        self.Vplatoon = struct_symSX([ entry('Truck', struct=self.Vtruck, repeat=self.Ntruck) ])
        self.Pplatoon = struct_symSX([ entry('Truck', struct=self.Ptruck, repeat=self.Ntruck) ])

        #self.PlatoonParameters = self.Pplatoon()
        

        
        #
        # Obtain aliases with the Ellipsis index:
        #
        pos, vel    = self.states[...]
        TdE, Tdbrk  = self.inputs[...]

        PosLeader   = SX.sym('PosLeader')
        HazLeader   = SX.sym('leader')

        Distance    = self.parameters['Distance']
        
        
        ###################################################
        #  Create dynamics (this is a unique declaration) #
        ###################################################
        CdModifLeader   = 1 - HazLeader*( self.parameters['DragRedSlope']*( PosLeader - pos ) + self.parameters['DragRedOffset'] )/100.

        Faero = 0.5 * env['airdens'] * self.parameters['CdA'] * CdModifLeader * vel**2     # Aerodyn force
        Td    = TdE - Tdbrk                                                                # Torque at gearbox output

        SinSlope = 0                                                                       # Build Fourrier series for the slope
        for k in range(NFourrierCoeff):
            SinSlope +=  (2*pi*(k+1)/Distance)*self.parameters['Sin'][k]*cos(2*pi*k*pos/Distance)
        for k in range(NFourrierCoeff):
            SinSlope -=  (2*pi*(k+1)/Distance)*self.parameters['Cos'][k]*sin(2*pi*k*pos/Distance)


        Fg      = env['g']*self.parameters['m']*SinSlope                                                               # Force due to slopes
        Froll   = self.parameters['cr']*self.parameters['m']*9.81*sqrt(1 - SinSlope**2)                                # Force from roll
        Facc    = Td * self.parameters['r_t']/self.parameters['rw'] - Faero - Fg - Froll                               # Acceleration force
        dvel    = Facc/self.parameters['m']                                                                            # Velocity change

        rhs = struct_SX([
                         entry("pos", expr = vel),
                         entry("vel", expr = dvel)
                         ])


        ## Build the height for plotting
        Height = 0                                                                                         # Build Fourrier series for the height
        for k in range(NFourrierCoeff):
            Height   += self.parameters['Sin'][k]*sin(2*pi*(k+1)*pos/Distance)
        for k in range(NFourrierCoeff):
            Height   += self.parameters['Cos'][k]*cos(2*pi*(k+1)*pos/Distance)


        self.HeightFunc = SXFunction('height',[self.parameters,pos],[Height,SinSlope])


        #########################################################
        #  Create the integrator (this is a unique declaration) #
        #########################################################
        Tf = SX.sym('Tf')
        fode = SXFunction('ode',[self.states,self.inputs,self.parameters,PosLeader,HazLeader],[rhs])

        self.dt    = 1/float(self.nk)

        k1         = self.states
        AllPosExpr = []
        for k in range(self.nstep):
            AllPosExpr.append(self.states(k1)['pos'])
            [f]  = fode([k1 , self.inputs, self.parameters, self._AllPos_struct['pos'][k], HazLeader])
            k1   = k1 + Tf*self.dt*f/float(self.nstep)

        AllPosExpr = struct_SX([
                                entry('pos', expr = AllPosExpr)
                                ])

        self._euler_struct = struct_SX([
                                  entry('final',              expr = k1),
                                  entry('intermediate_pos',   expr = AllPosExpr)
                                  ])

        self._euler = SXFunction('euler',[self.states, self.inputs, self.parameters, self._AllPos_struct, HazLeader, Tf],[self._euler_struct])


    def SetSolvers(self, DicIpopt = {'max_iter':3000,'tol':1e-6}):
        
        if not(hasattr(self,'ObjFunc')):
            print 'Object misses objective function, please define first'
            return

        ####################################################################################

        ###########################      BUILD SOLVERS      ################################

        ####################################################################################

        #####################
        #                   #
        #   TRUCK SOLVER    #
        #                   #
        #####################

        #
        # Create shooting constraints for truck
        #
        shooting = []
        for time in range(self.nk):
            state_truck     = self.Vtruck['State',    time]
            input_truck     = self.Vtruck['Input',    time]
            param_truck     = self.Ptruck['Parameter'     ]
            Tf              = self.Vtruck['Tf'            ]
            PosLeader_truck = self.Ptruck['PosLeader',time]
            HazLeader_truck = self.Ptruck['HazLeader'     ]

            [shoot]  = self._euler([state_truck, input_truck, param_truck, PosLeader_truck, HazLeader_truck, Tf])
            shoot    = self._euler_struct(shoot)

            # append continuity constraints
            shooting.append(self.Vtruck['State',time+1] - shoot['final']) # The state evolution gets connected through the constraint shooting

        [fobj_truck] = self.ObjFunc([self.Vtruck,self.Ptruck])

        self._g_truck = struct_SX([
                                   entry('shooting',         expr = shooting),
                                   ])

        g_truck_func = SXFunction('g_truck',[self.Vtruck, self.Ptruck],[self._g_truck]) #OBS: for debugging purposes

        nlp=SXFunction("nlp", nlpIn(x=self.Vtruck, p=self.Ptruck),nlpOut(f=fobj_truck, g = self._g_truck))
        self.solver_truck = NlpSolver("solver", "ipopt", nlp,DicIpopt)


        #######################
        #                     #
        #   PLATOON SOLVER    #
        #                     #
        #######################

        #
        # Platoon variables
        #

        ## Platoon structure: Vplatoon['Truck', truck number ,'State'/'Input'/'Parameter', time , state label]



        #
        # Create shooting constraints for platooning
        #
        shooting = []
        ordering_const = []

        # Multiple shooting - Platoon
        for time in range(self.nk):
            AllPos = self._AllPos_struct(0)
    
            for truck in range(self.Ntruck):
                state_truck = self.Vplatoon['Truck',truck,'State',    time]
                input_truck = self.Vplatoon['Truck',truck,'Input',    time]
                param_truck = self.Pplatoon['Truck',truck,'Parameter'     ]
                Tf          = self.Vplatoon['Truck',truck,'Tf'            ]
        
                if truck == 0:
                    HazLeader = 0.
                else:
                    HazLeader = 1.
        
                [shoot]  = self._euler([state_truck, input_truck, param_truck, AllPos, HazLeader, Tf])
                shoot    = self._euler_struct(shoot)
                AllPos   = shoot['intermediate_pos']

                # append continuity constraints
                shooting.append(self.Vplatoon['Truck',truck,'State',time+1] - shoot['final']) # The state evolution gets connected through the constraint shooting

        #
        # Create cost for platooning
        #
        fobj = 0
        for truck in range(self.Ntruck):
            [fobj_truck] = self.ObjFunc([self.Vplatoon['Truck',truck],self.Pplatoon['Truck',truck]])
            fobj += fobj_truck

        #
        # Create ordering constraints for platooning
        #
        for truck in range(self.Ntruck-1):
            for time in range(self.nk):
                ordering_const.append(self.Vplatoon['Truck',truck+1,'State',time,'pos'] - self.Vplatoon['Truck',truck,'State',time,'pos'])

        #
        # Match final times
        #
        TfConst = []
        for truck in range(self.Ntruck-1):
            TfConst.append(self.Vplatoon['Truck',truck+1,'Tf'] - self.Vplatoon['Truck',truck,'Tf'])

        self._g_platoon = struct_SX([
                                     entry('shooting',         expr = shooting),
                                     entry('ordering',         expr = ordering_const),
                                     entry('final_times',      expr = TfConst)
                                     ])

        nlp=SXFunction("nlp", nlpIn(x=self.Vplatoon, p=self.Pplatoon),nlpOut(f=fobj, g = self._g_platoon))
        self.solver_platoon = NlpSolver("solver", "ipopt", nlp)


    def Optimize(self, Separation0 = 10.5, Vel0 = 100, DicIpopt = {'max_iter':3000,'tol':1e-6}):
        
        ####################################################################################

        ###########################     NUMERICAL PART      ################################

        ####################################################################################

        self.resolution = self.ProblemParameters['MaxTime']/self.nk

        if not(len(self.ProblemParameters['Sin']) == self.NFourrierCoeff) or not(len(self.ProblemParameters['Cos']) == self.NFourrierCoeff):
            print "Incorrect length of the Fourrier coefficients, abort"
            return

        ############################
        #                          #
        #   GREEDY OPTIMIZATION    #
        #                          #
        ############################

        vel_guess = 100/3.6

        truck_lb   = self.Vtruck(-inf)
        truck_ub   = self.Vtruck( inf)
        truck_init = self.Vtruck()

        truck_lb['Input']             = 0.
        truck_ub['Input',:,'TdE']     = self.ProblemParameters['MaxTorque']
        truck_lb['State',:,'vel']     = self.ProblemParameters['MinVel']/3.6
        truck_ub['State',:,'vel']     = self.ProblemParameters['MaxVel']/3.6
        truck_lb['Tf']                = 0.1
        truck_ub['Tf']                = self.ProblemParameters['MaxTime']

        truck_init['State',:,'vel']   = vel_guess

        truck_init['Tf']              = self.ProblemParameters['MaxTime']

        truck_lbg  = self._g_truck()
        truck_ubg  = self._g_truck()
        Ptruck_num = self.Ptruck()

        self.Sol_greedy = self.Vplatoon()

        self.SolversOut['Cost'] = []
        self.SolversOut['Cost'].append(0)

        for truck in range(self.Ntruck):
    
            #Truck-specific stuff
            truck_init['State',:,'pos'] = [vel_guess*i*self.ProblemParameters['MaxTime']*self.dt-Separation0*truck for i in range(self.nk+1)]
    
            truck_lb['State',0,'pos'] = -Separation0*truck
            truck_ub['State',0,'pos'] = -Separation0*truck
            truck_lb['State',0,'vel'] =  Vel0/3.6
            truck_ub['State',0,'vel'] =  Vel0/3.6
    
            truck_lb['State',-1,'pos']  = self.ProblemParameters['Distance'] - Separation0*truck
    
            if truck == 0:
                Ptruck_num['HazLeader'] = 0.
                Ptruck_num['PosLeader'] = truck_init['State',:,'pos']
            else:
                Ptruck_num['HazLeader'] = 1.
                #Ordering constraints as bounds
                for time in range(1,self.nk+1):
                    truck_ub['State',time,'pos'] = self.Sol_greedy['Truck',truck-1,'State',time,'pos'] - self.ProblemParameters['MinSeparation']

                #Next truck inherits final time
                truck_lb['Tf'] = self.Sol_greedy['Truck',truck-1,'Tf']
                truck_ub['Tf'] = self.Sol_greedy['Truck',truck-1,'Tf']

            for label in self.Constants.keys():
                print 'Truck',truck,' ', label, self.Constants[label][truck]
                Ptruck_num['Parameter',label] = self.Constants[label][truck]

            for label in ['Sin','Cos','Distance']:
                print 'Truck',truck,' ', label, self.ProblemParameters[label]
                Ptruck_num['Parameter',label] = self.ProblemParameters[label]
            
            self.GreedyParamCheck = Ptruck_num
            self.truck_lb = truck_lb
            self.truck_ub = truck_ub
            self.truck_init = truck_init
            
            self.solver_truck.setInput(truck_lb,"lbx")
            self.solver_truck.setInput(truck_ub,"ubx")
            self.solver_truck.setInput(Ptruck_num,"p")
            self.solver_truck.setInput(truck_lbg,"lbg")
            self.solver_truck.setInput(truck_ubg,"ubg")
            self.solver_truck.setInput(truck_init,"x0")
            self.solver_truck.evaluate()

    
            Sol_truck = self.Vtruck(self.solver_truck.getOutput())
            #assert(0==1)

            #Extract intermediate positions of the current truck -> leader of the next truck
            for time in range(self.nk):
                [shoot] = self._euler([Sol_truck['State',time],Sol_truck['Input',time],Ptruck_num['Parameter'],Ptruck_num['PosLeader',time],Ptruck_num['HazLeader'],Sol_truck['Tf']])
                shoot   = self._euler_struct(shoot)
                Ptruck_num['PosLeader',time] = shoot['intermediate_pos']



            self.Sol_greedy['Truck',truck] = Sol_truck.cat

            self.SolversOut['Cost'][0] += self.solver_truck.getOutput('f')


        ##############################
        #                            #
        #   HOLISTIC  OPTIMIZATION   #
        #                            #
        ##############################

        #
        # Define explicit bounds on variables
        #
        platoon_lb   = self.Vplatoon(-inf)
        platoon_ub   = self.Vplatoon( inf)
        platoon_init = self.Vplatoon()

        platoon_lb['Truck',:]   = truck_lb
        platoon_ub['Truck',:]   = truck_ub
        platoon_init['Truck',:] = truck_init

        platoon_ub['Truck',:,'State',:,'pos'] = inf

        for truck in range(self.Ntruck):
            platoon_init['Truck',truck,'State',:,'pos'] = [vel_guess*i*self.ProblemParameters['MaxTime']*self.dt-Separation0*truck for i in range(self.nk+1)]

        # Set states and its bounds
        platoon_init['Truck',:,'Tf'] = truck_init['Tf']

        lbg = self._g_platoon()
        ubg = self._g_platoon()

        lbg['ordering'] =  -inf
        ubg['ordering'] =  -self.ProblemParameters['MinSeparation']

        Pplatoon_num = self.Pplatoon()
        for label in self.Constants.keys():
            for truck in range(self.Ntruck):
                print label,self.Constants[label][truck]
                Pplatoon_num['Truck',truck,'Parameter',label] = self.Constants[label][truck]
            
        for label in ['Sin','Cos','Distance']:
            for truck in range(self.Ntruck):
                print label,self.ProblemParameters[label]
                Pplatoon_num['Truck',truck,'Parameter',label] = self.ProblemParameters[label]

        for truck in range(self.Ntruck):
            platoon_lb['Truck',truck,'State',0,'pos']  = -Separation0*truck
            platoon_ub['Truck',truck,'State',0,'pos']  = -Separation0*truck
            platoon_lb['Truck',truck,'State',0,'vel']  =  Vel0/3.6
            platoon_ub['Truck',truck,'State',0,'vel']  =  Vel0/3.6
            platoon_lb['Truck',truck,'State',-1,'pos'] =  self.ProblemParameters['Distance'] - Separation0*truck


        self.PlatoonParamCheck = Pplatoon_num
        self.platoon_lb = platoon_lb
        self.platoon_ub = platoon_ub
        self.platoon_init = platoon_init

        self.solver_platoon.setInput(platoon_lb,"lbx")
        self.solver_platoon.setInput(platoon_ub,"ubx")
        self.solver_platoon.setInput(Pplatoon_num,"p")
        self.solver_platoon.setInput(lbg,"lbg")
        self.solver_platoon.setInput(ubg,"ubg")
        self.solver_platoon.setInput(platoon_init,"x0")


        self.solver_platoon.evaluate()

        self.Sol_platoon = self.Vplatoon(self.solver_platoon.getOutput())

        self.SolversOut['Cost'].append(self.solver_platoon.getOutput('f'))


    def HazPlot(self):
        
        plt.figure(1)
        for index_sub, label_sub in enumerate(self.inputs.keys()):
            plt.subplot(1,2,index_sub+1)
            for truck in range(self.Ntruck):
                plt.hold('on')
                plt.step(self.Sol_greedy['Truck',truck,'Tf']*range(self.nk)/float(self.nk),self.Sol_greedy['Truck',truck,'Input',:,label_sub])
                plt.ylabel(label_sub)
        plt.figure(2)
        for index_sub, label_sub in enumerate(self.states.keys()):
            plt.subplot(1,2,index_sub+1)
            for truck in range(self.Ntruck):
                plt.hold('on')
                plt.plot(self.Sol_greedy['Truck',truck,'Tf']*range(self.nk+1)/float(self.nk),self._ScalePlot[label_sub]*np.array(self.Sol_greedy['Truck',truck,'State',:,label_sub]))
            plt.ylabel(label_sub)
        
        plt.figure(3)
        plt.hold('on')
        for truck in range(1,self.Ntruck):
            plt.plot(self.Sol_greedy['Truck',truck,'Tf']*range(self.nk+1)/float(self.nk),np.array(self.Sol_greedy['Truck',truck,'State',:,'pos'])-np.array(self.Sol_greedy['Truck',truck-1,'State',:,'pos']))
            plt.ylabel('Separations')

        plt.figure(4)
        for index_sub, label_sub in enumerate(self.inputs.keys()):
            plt.subplot(1,2,index_sub+1)
            for truck in range(self.Ntruck):
                plt.hold('on')
                plt.step(self.Sol_platoon['Truck',truck,'Tf']*range(self.nk)/float(self.nk),self.Sol_platoon['Truck',truck,'Input',:,label_sub])
                plt.ylabel(label_sub)
        plt.figure(5)
        for index_sub, label_sub in enumerate(self.states.keys()):
            plt.subplot(1,2,index_sub+1)
            for truck in range(self.Ntruck):
                plt.hold('on')
                plt.plot(self.Sol_platoon['Truck',truck,'Tf']*range(self.nk+1)/float(self.nk),self._ScalePlot[label_sub]*np.array(self.Sol_platoon['Truck',truck,'State',:,label_sub]))
                plt.ylabel(label_sub)

        plt.figure(6)
        plt.hold('on')
        for truck in range(1,self.Ntruck):
            plt.plot(self.Sol_platoon['Truck',truck,'Tf']*range(self.nk+1)/float(self.nk),np.array(self.Sol_platoon['Truck',truck,'State',:,'pos'])-np.array(self.Sol_platoon['Truck',truck-1,'State',:,'pos']))
        plt.ylabel('Separations')

        


