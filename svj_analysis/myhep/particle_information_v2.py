"""
Program: This module is about .root file particle information as OOP and
         using pandas outputs DataFrame format.
Author: You-Wei Hsiao
Institute: Department of Physics, National Tsing Hua University, Hsinchu, Taiwan
Mail: hsiao.phys@gapp.nthu.edu.tw
History: 2021/04/17
         First release
Version: v.2.1
"""


import numpy as np
import pandas as pd
import uproot


# 1. class GenParticle
class classGenParticle():
    def __init__(self, data):
        self.data = data
        self.event = data.array('Event')  # ??
        self.PID = data.array('Particle.PID')
        self.Status = data.array('Particle.Status')
        self.M1 = data.array('Particle.M1')
        self.M2 = data.array('Particle.M2')
        self.D1 = data.array('Particle.D1')
        self.D2 = data.array('Particle.D2')
        self.Charge = data.array('Particle.Charge')
        self.Mass = data.array('Particle.Mass')
        self.E = data.array('Particle.E')
        self.Px = data.array('Particle.Px')
        self.Py = data.array('Particle.Py')
        self.Pz = data.array('Particle.Pz')
        self.P = data.array('Particle.P')
        self.PT = data.array('Particle.PT')
        self.Eta = data.array('Particle.Eta')
        self.Phi = data.array('Particle.Phi')
        self.length = len(data.array('Particle.PID'))

#     To get the class GenParticle informations array in the i-th(index) event.
    def dataframelize(self, i):
        idx = np.linspace(0, len(self.PID[i])-1, num=len(self.PID[i]))
        dict_particle = {"PID": self.PID[i],
                         "Status": self.Status[i],
                         "M1": self.M1[i],
                         "M2": self.M2[i],
                         "D1": self.D1[i],
                         "D2": self.D2[i],
                         "Mass": self.Mass[i],
                         "PT": self.PT[i],
                         "Eta": self.Eta[i],
                         "Phi": self.Phi[i]
                         }
        df_particle = pd.DataFrame(dict_particle)
        return df_particle


"""
def dataframelize(self, i):
        idx = np.linspace(0, len(self.PID[i])-1, num=len(self.PID[i]))
        dict_particle = {"Index": idx,
                         "PID": self.PID[i],
                         "Status": self.Status[i],
                         "M1st": self.M1[i],
                         "M2nd": self.M2[i],
                         "D1st": self.D1[i],
                         "Dlast": self.D2[i],
                         "Mass": self.Mass[i],
                         "PT": self.PT[i],
                         "Eta": self.Eta[i],
                         "Phi": self.Phi[i]
                        }
        df_particle = pd.DataFrame(dict_particle)
        print("(# events, # info. = {})".format(df_particle.shape))
        return df_particle
"""


# 2. class Jet
class classJet():
    def __init__(self, data):
        self.data = data
        self.PT = data.array('Jet.PT')
        self.Eta = data.array('Jet.Eta')
        self.Phi = data.array('Jet.Phi')
        self.Mass = data.array('Jet.Mass')
        self.length = len(data.array('Jet.PT'))

#     To get the class Jet informations array in the i-th(index) event.
    def dataframelize(self, i):
        #         idx = np.linspace(0, len(self.PT[i])-1, num=len(self.PT[i]))
        dict_jet = {"Mass": self.Mass[i],
                    "PT": self.PT[i],
                    "Eta": self.Eta[i],
                    "Phi": self.Phi[i]
                    }
        df_jet = pd.DataFrame(dict_jet)
        return df_jet


# 3. class Event
class classEvent():
    def __init__(self, data):
        self.data = data
        self.Weight = data.array('Event.Weight')
        self.length = len(data.array('Event.Weight'))

#     To get the class Event informations array in the i-th(index) event.
    def dataframelize(self, i):
        #         idx = np.linspace(0, len(self.Weight[i])-1, num=len(self.Weight[i]))
        dict_event = {"Weight": self.Weight[i]
                      }
        df_event = pd.DataFrame(dict_event)
        return df_event
