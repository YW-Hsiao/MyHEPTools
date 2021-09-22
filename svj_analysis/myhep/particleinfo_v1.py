"""
Author: You-Wei Hsiao
Institute: Department of Physics, National Tsing Hua University, Hsinchu, Taiwan
Mail: hsiao.phys@gapp.nthu.edu.tw
"""


import numpy as np
import uproot

class BranchGenParticles:
    def __init__(self, data):
        self.data = data
        self.length = len(data.array('Particle.Status'))
        self.Status = data.array('Particle.Status')
        self.PID = data.array('Particle.PID')
        self.M1 = data.array('Particle.M1')
        self.M2 = data.array('Particle.M2')
        self.D1 = data.array('Particle.D1')
        self.D2 = data.array('Particle.D2')
        self.PT = data.array('Particle.PT')
        self.Eta = data.array('Particle.Eta')
        self.Phi = data.array('Particle.Phi')
        self.Mass = data.array('Particle.Mass')
        self.Labels = ['Status', 'PID', 'M1', 'M2', 'D1', 'D2', 'PT', 'Eta', 'Phi', 'Mass']
        
        
#     To get the GenParticles information array in the i-th event.
    def length_i(self, i):
        return len(self.Status[i])
    def Status_i(self, i):
        return self.Status[i]
    def PID_i(self, i):
        return self.PID[i]
    def M1_i(self, i):
        return self.M1[i]
    def M2_i(self, i):
        return self.M2[i]
    def D1_i(self, i):
        return self.D1[i]
    def D2_i(self, i):
        return self.D2[i]
    def PT_i(self, i):
        return self.PT[i]
    def Eta_i(self, i):
        return self.Eta[i]
    def Phi_i(self, i):
        return self.Phi[i]
    def Mass_i(self, i):
        return self.Mass[i]
    
    
    
class BranchJet:
    def __init__(self, data):
        self.data = data
        self.length = len(data.array('Jet.PT'))
        self.PT = data.array('Jet.PT')
        self.Eta = data.array('Jet.Eta')
        self.Phi = data.array('Jet.Phi')
        self.Mass = data.array('Jet.Mass')
        
    def length_i(self, i):
        return len(self.PT[i])
    def PT_i(self, i):
        return self.PT[i]
    def Eta_i(self, i):
        return self.Eta[i]
    def Phi_i(self, i):
        return self.Phi[i]
    def Mass_i(self, i):
        return self.Mass[i]
    
    
    
class Event_Weight:
    def __init__(self, data):
        self.data = data
        self.length = len(data.array('Event.Weight'))
        self.Event_Weight = np.array(data.array('Event.Weight'))
        
    def Event_Weight_i(self, i):
        return self.Event_Weight[i]