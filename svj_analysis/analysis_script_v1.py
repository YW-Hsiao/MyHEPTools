#!/usr/bin/env python3
"""
Program: This .py is the analysis script with version 1 to analyze SVJ data.
         I use MyHEPTools/svj_anlaysis/myhep/analysis_v3.py
         and particle_information_v2.py.
         For this .py, I follow
         https://github.com/YW-Hsiao/MyHEPTools/blob/main/svj_analysis/analysis-1.py
         and there is a test in svj_setting_master/test_analysis/test_8_analysis_script.
         Executable command is
         python3 analysis_script_v1.py input 0.4 -1 0 20 2.5 20000 output 0.3 5 remark
         
Author: You-Wei Hsiao
Institute: Department of Physics, National Tsing Hua University, Hsinchu, Taiwan
Mail: hsiao.phys@gapp.nthu.edu.tw
History (v.1.0): 2022/04/21 First release.
History (v.1.1): 2022/05/23 Debug for analyze_xdxdx and filter of selectStableFinalStateParticle.
                 Add dark_sector, neutrino, jet, jet_MET.
"""

################################################################################
#                                Import Packages                               #
################################################################################
# The Python Standard Library
import os
import sys
import time
import datetime
import glob
import multiprocessing as mp

# The Third-Party Library
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import prettytable
import uproot
import pyjet
import importlib

# My Packages
import myhep.particle_information_v2 as mypInfo_v2
import myhep.analytical_function_v2 as myaFun_v2
import myhep.analysis_v3 as myAnal_v3
# import myhep.particleinfo_v1 as mypiv1
# import myhep.particlefun_v1 as myafv1

# increase figure showing resolution
# %config InlineBackend.figure_format = 'retina'


print(time.strftime('%Y/%m/%d %a, %H:%M:%S %Z', time.localtime()))
start = datetime.datetime.now()
####################################  Start  ###################################
print("Start to execute this analysis script.")
################################################################################
#                              1. Input Arguments                              #
################################################################################
print("*------  1. Check your input argument  ------*")
n_argv = len(sys.argv)
# check input arguments
print("You have {} inputs.".format(n_argv - 1))
if n_argv != 12:
    print("The number of arguments is incorrect. Please check input parameters.")
    sys.exit()
print("Input list = {}.".format(sys.argv))

# define input arguments
analysis_script = str(sys.argv[0])
INPUT_FILE = str(sys.argv[1])
R = float(sys.argv[2])
JetClusteringAlgorithm = int(sys.argv[3])
pTmin_pyjet = int(sys.argv[4])
pT_min = int(sys.argv[5])
eta_max = float(sys.argv[6])
nevents = int(sys.argv[7])
OUTPUT_PATH = str(sys.argv[8])
rinv = float(sys.argv[9])
Lambda_d = int(sys.argv[10])
remark = str(sys.argv[11])

print("The analysis script = {}".format(analysis_script))
print("The input file = {}".format(INPUT_FILE))
print("The jet cone size = {}".format(R))
if JetClusteringAlgorithm == -1:
    print("The jet clustering algorithm = anti-kt")
elif JetClusteringAlgorithm == 0:
    print("The jet clustering algorithm = Cambridge-Aachen(C/A)")
elif JetClusteringAlgorithm == 1:
    print("The jet clustering algorithm = kT")
else:
    print("\033[7;31m****** Please check your jet clustering algorithm input. "
          "******\033[0m")
    sys.exit()
print("The minimal pT for jet clustering = {}".format(pTmin_pyjet))
print("The minimal pT for preselection = {}".format(pT_min))
print("The maximum eta for preselection = {}".format(eta_max))
print("Number of events = {}".format(nevents))
print("The output path = {}".format(OUTPUT_PATH))
print("The invisible rate = {}".format(rinv))
print("The confinement scale = {}".format(Lambda_d))
print("My remark = {}".format(remark))
print('\n')


################################################################################
#               2. Import .root Files and Load the Data via class              #
################################################################################
print("*------  2. Load the data  ------*")
t1 = datetime.datetime.now()

DATA = uproot.open(INPUT_FILE)['Delphes;1']
GP = mypInfo_v2.classGenParticle(DATA)
Jet = mypInfo_v2.classJet(DATA)
Event = mypInfo_v2.classEvent(DATA)

t2 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print("\n*------  2-1. Check the number of event for each branch  ------*")
if GP.length == Jet.length == Event.length:
    print("{} events in the .root file.".format(GP.length))
else:
    print("\033[7;31m****** There is the problem for the number of event "
          "in the .root file. ******\033[0m")
    print("\033[7;31m****** Please check your .root file. ******\033[0m")
print('\n')


################################################################################
#           3. Analyze the Dark Sector in the Parton and Truth Levels          #
################################################################################
print("*------  3. Analyze the dark sector in the parton "
      "and truth levels  ------*")
print("\n*------  3-1. Dark quark pair  ------*")
t1 = datetime.datetime.now()

df_xdxdx_23 = myAnal_v3.analyze_xdxdx(GP, status=23)
df_xdxdx_71 = myAnal_v3.analyze_xdxdx(GP, status=71)

t2 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print("\n*------  3-2. Dark sector  ------*")
t1 = datetime.datetime.now()

myAnal_v3.dark_sector(GP)

t2 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print("\n*------  3-3. Neutrino  ------*")
t1 = datetime.datetime.now()

p12_s1, p14_s1, p16_s1 = myAnal_v3.neutrino(GP)

t2 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print('\n')


################################################################################
#                               4. Jet Clustering                              #
################################################################################
print("*------  4. Jet clustering  ------*")
print("\n*------  4-1. Select stable final state particles without/with "
      "filtering out dark sector  ------*")
t1 = datetime.datetime.now()

SFSP, SFSP_filterDM = myAnal_v3.selectStableFinalStateParticle(
    GP, filter=[51, -51, 53, -53, 4900211, -4900211, 4900213, -4900213,
                4900101, -4900101, 4900021,
                12, -12, 14, -14, 16, -16])

t2 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print("\n*------  4-2. Let's do the jet clustering!!  ------*")
t1 = datetime.datetime.now()

PseudoJet = myAnal_v3.jetClustering_v1(SFSP, R=R,
                                       p=JetClusteringAlgorithm,
                                       pTmin=pTmin_pyjet)
PseudoJet_filterDM = myAnal_v3.jetClustering_v1(SFSP_filterDM, R=R,
                                                p=JetClusteringAlgorithm,
                                                pTmin=pTmin_pyjet)

t2 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print('\n')


################################################################################
#                 5. Analyze the Jet and MET in the Truth Level                #
################################################################################
print("*------  5. Analyze the jet and MET in the truth level  ------*")
print("\n*------  5-1. Preselection  ------*")
t1 = datetime.datetime.now()

presel_bef, presel_pt, presel_pt_eta, presel_idx = myAnal_v3.preselection_v1(
    PseudoJet_filterDM, pT_min=pT_min, eta_max=eta_max)

t2 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print("\n*------  5-2. Missing transverse energy vector  ------*")
t1 = datetime.datetime.now()

arr_MET, df_MET = myAnal_v3.MET_visParticles_v1(SFSP_filterDM)

t2 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print("\n*------  5-3. Jet  ------*")
t1 = datetime.datetime.now()

df_jet = myAnal_v3.jet(presel_pt_eta)

t2 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print("\n*------  5-4. Jet and MET  ------*")
t1 = datetime.datetime.now()

df_jet_MET = myAnal_v3.jet_MET(presel_pt_eta, arr_MET)

t2 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print("\n*------  5-5. Analyze truth jet with scheme 1 & version 2  ------*")
t1 = datetime.datetime.now()

df_N_jet, df_jj, df_jjj = myAnal_v3.analyze_truthJet_scheme1_v2(presel_pt_eta)

t2 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print("\n*------  5-6. Analyze truth jet and MET with scheme 1 "
      "& version 1  ------*")
t1 = datetime.datetime.now()

df_jj_MET, df_jjj_MET = myAnal_v3.analyze_truthJet_MET_scheme1_v1(
    presel_pt_eta, arr_MET)

t2 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print('\n')


################################################################################
#                                6. Event Weight                               #
################################################################################
print("*------  6. Event weight  ------*")
# Method 1:
event_weight_1 = np.array(Event.Weight)

# Method 2:
_weight = []
for i in range(Event.length):
    _weight.append(Event.Weight[i][0])

event_weight_2 = np.array(_weight)
event_weight_2
print("Done for event weight.")
print('\n')


################################################################################
#                        7. Save in .csv and .npz Files                        #
################################################################################
print("*------  7. Save in .csv and .npz files  ------*")
# default OUTPUT_PATH = '/youwei_u3/svj_data_master/scheme_1/analysis_script_v1'
# output_file = f"status23_rinv{int(rinv * 10)}_Lambdad{Lambda_d}_{remark}"
file = f"_rinv{int(rinv * 10)}_Lambdad{Lambda_d}_{remark}"
df_xdxdx_23.to_csv(OUTPUT_PATH + "/status23" + file + ".csv", index=False)
df_xdxdx_71.to_csv(OUTPUT_PATH + "/status71" + file + ".csv", index=False)
df_MET.to_csv(OUTPUT_PATH + "/met" + file + ".csv", index=False)
df_jet.to_csv(OUTPUT_PATH + "/jet" + file + ".csv", index=False)
df_jet_MET.to_csv(OUTPUT_PATH + "/jet_met" + file + ".csv", index=False)
df_N_jet.to_csv(OUTPUT_PATH + "/n_jet" + file + ".csv", index=False)
df_jj.to_csv(OUTPUT_PATH + "/jj" + file + ".csv", index=False)
df_jjj.to_csv(OUTPUT_PATH + "/jjj" + file + ".csv", index=False)
df_jj_MET.to_csv(OUTPUT_PATH + "/jj_met" + file + ".csv", index=False)
df_jjj_MET.to_csv(OUTPUT_PATH + "/jjj_met" + file + ".csv", index=False)
np.savez_compressed(OUTPUT_PATH + "/neutrinos" + file + ".npz",
                    nu12=p12_s1, nu14=p14_s1, nu16=p16_s1)
np.savez_compressed(OUTPUT_PATH + "/weight" + file + ".npz",
                    weight_1=event_weight_1, weight_2=event_weight_2)
print('\n')


print("The end of the program!!")
#####################################  End  ####################################
end = datetime.datetime.now()
print("\033[33mProgram Execution Time = {}\033[0m".format(end - start))
