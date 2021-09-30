#!/usr/bin/env python3
"""
Program: This .py is just for the analyzing the CKKW-L of DP8.
         
         For this .py, I do a demo to do the test in
         /youwei_home/SVJ_CKKWL/s-channel_ckkwl-v2/Analysis
         /Test_analysis_ckkwl_DP8.ipynb.
         
Author: You-Wei Hsiao
Institute: Department of Physics, National Tsing Hua University, Hsinchu, Taiwan
Mail: hsiao.phys@gapp.nthu.edu.tw
History: 2021/09/27
         First release
Version: v.1.0
"""

################################################################################
###### Import Packages
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
import myhep.particle_information_v2 as mypiv2
import myhep.analytical_function_v2 as myafv2
import myhep.particleinfo_v1 as mypiv1
import myhep.particlefun_v1 as myafv1
import myhep.parse_ckkwl_DP8 as myparDP8

# increase figure showing resolution
# %config InlineBackend.figure_format = 'retina'



print(time.strftime('%Y/%m/%d %a, %H:%M:%S %Z', time.localtime()))
start = datetime.datetime.now()
################################################################################
###### 1. Input Argument
################################################################################
print("+----+ 1. Check your input argument. +----+")
print("You have {} inputs which are {}.".format(len(sys.argv), sys.argv))

analysis_script = str(sys.argv[0])
INPUT_FILE = str(sys.argv[1])
R = float(sys.argv[2])
jetClusteringAlgorithm = int(sys.argv[3])
pTmin_pyjet = int(sys.argv[4])
N_jet = int(sys.argv[5])
pT_min = int(sys.argv[6])
pT1_min = int(sys.argv[7])
pT2_min = int(sys.argv[8])
eta12_max = float(sys.argv[9])
eta1_max = float(sys.argv[10])
eta2_max = float(sys.argv[11])
nevents = int(sys.argv[12])
OUTPUT_PATH = str(sys.argv[13])
rinv = float(sys.argv[14])
Lambda_d = int(sys.argv[15])
vers = str(sys.argv[16])

print("The analysis script is {}.".format(analysis_script))
print("The input file = {}".format(INPUT_FILE))
print("The jet cone size = {}".format(R))
if jetClusteringAlgorithm == -1:
    print("The jet clustering algorithm = anti-kT")
elif jetClusteringAlgorithm == 0:
    print("The jet clustering algorithm = Cambridge-Aachen(C/A)")
elif jetClusteringAlgorithm == 1:
    print("The jet clustering algorithm = kT")
else:
    print("\033[7;31m****** Please check your jet clustering algorithm input. "
          "******\033[0m")
#     Add sys.exit(1) ??
print("The minimum pT for jet clustering = {}".format(pTmin_pyjet))
print("The minimum number of jets = {}".format(N_jet))
print("The min pT for the all events = {}".format(pT_min))
print("The min pT1 for the leading jet = {}".format(pT1_min))
print("The min pT2 for the sub-leading jet = {}".format(pT2_min))
print("The max eta between jet1 and jet2 = {}".format(eta12_max))
print("The max eta for the leading jet = {}".format(eta1_max))
print("The max eta for the sub-leading jet = {}".format(eta2_max))
print("The number of events = {}".format(nevents))
print("The output path = {}".format(OUTPUT_PATH))
print("The invisible rate = {}".format(rinv))
print("Lambda_d = {}".format(Lambda_d))
print("Which analysis = {}".format(vers))
print("\n")


################################################################################
###### 2. Import .root Files and Load the Data via class
################################################################################
print("+----+ 2. Load the data. +----+")
t1 = datetime.datetime.now()

DATA = uproot.open(INPUT_FILE)['Delphes;1']
GP = mypiv2.classGenParticle(DATA)
Jet = mypiv2.classJet(DATA)
Event = mypiv2.classEvent(DATA)

t2 = datetime.datetime.now()
# print('Time =', end - start)
print("\033[33mTime = {}\033[0m".format(t2 - t1))
print("")

print("2-1. Check the number of event for each branch.")
if GP.length == Jet.length == Event.length:
    print("There are {} events in the .root file.".format(GP.length))
else:
    print("""\033[7;31m****** There is the problem for the number of event \
    in the .root file. ******\033[0m""")
#     print("\033[7;31m****** There is the problem for the number of event "
#           "in the .root file. ******\033[0m")
    print("\033[7;31m****** Please check your .root file. ******\033[0m")
print("\n")
    
print("2-2. Create the event list without event weight = 0.")
event_weight_not0 = []

for i in range(GP.length):
    if Event.Weight[i] != 0:
        event_weight_not0.append(i)
        
print(len(event_weight_not0))
print("\n")


################################################################################
###### 3. Analyze the Dark Quark Pair
################################################################################
print("+----+ 3. Analyze the dark quark pair. +----+")
t3 = datetime.datetime.now()

df_xdxdx_23 = myparDP8.parse_xdxdx_v2_ckkwl_DP8(GP, event_weight_not0, status=23)
df_xdxdx_71 = myparDP8.parse_xdxdx_v2_ckkwl_DP8(GP, event_weight_not0, status=71)

t4 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t4 - t3))
print("\n")


################################################################################
###### 4. Select Stable Final State Particle and Filter out Dark Matter
################################################################################
print("+----+ 4. Select stable final state particle and "
      "filter out dark matter. +----+")
t5 = datetime.datetime.now()

list_SFSP, list_SFSP_filterDM = myparDP8.selectStableFinalStateParticle_filterDM(
    GP, event_weight_not0, FILTER=[51, -51, 53, -53])

t6 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t6 - t5))
print("\n")


################################################################################
###### 5. Let's Do the Jet Clustering!!
################################################################################
print("+----+ 5. Let's do the jet clustering!! +----+")
t7 = datetime.datetime.now()

list_PseudoJet = myafv2.jetClustering(list_SFSP, R=R,
                                      p=jetClusteringAlgorithm,
                                      pTmin=pTmin_pyjet)
list_PseudoJet_filterDM = myafv2.jetClustering(list_SFSP_filterDM, R=R,
                                               p=jetClusteringAlgorithm,
                                               pTmin=pTmin_pyjet)

t8 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t8 - t7))
print("\n")


################################################################################
###### 6. Analyze the Jet in the Truth Level
################################################################################
print("+----+ 6. Analyze the jet in the truth level. +----+")
t9 = datetime.datetime.now()

list_truth_jet = myafv2.preselectTruthJet(list_PseudoJet, N_JET_MIN=N_jet)
list_truth_jet_filterDM = myafv2.preselectTruthJet(list_PseudoJet_filterDM,
                                                   N_JET_MIN=N_jet)

t10 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t10 - t9))
print("\n")


################################################################################
###### 7. Analyze the MET in the Truth Level
################################################################################
print("+----+ 7. Analyze the MET in the truth level. +----+")
t11 = datetime.datetime.now()

list_truth_MET = myafv2.parseMETTruthJet(list_SFSP_filterDM,
                                         list_PseudoJet_filterDM,
                                         N_jet_min=N_jet)

t12 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t12 - t11))
print("\n")


################################################################################
###### 8. Analyze the Jet in the Detector Level
################################################################################
print("+----+ 8. Analyze the jet in the detector level. +----+")
t13 = datetime.datetime.now()

list_detector_jet = myparDP8.preselectDetectorJet(Jet,
                                                  event_weight_not0,
                                                  N_JET_MIN=N_jet,
                                                  PT1_MIN=pT1_min,
                                                  PT2_MIN=pT2_min,
                                                  ETA_MAX=eta12_max)

t14 = datetime.datetime.now()
print("\033[33mTime = {}\033[0m".format(t14 - t13))
print("\n")


################################################################################
###### 9. Define the Event Weight
################################################################################
print("+----+ 9. Define the event weight. +----+")
event_weight = np.array(Event.Weight)/nevents


################################################################################
###### 10. Save to the .npz and .csv Files
################################################################################
print("+----+ 10. Save to the .npz and .csv files. +----+")
# OUTPUT_PATH = '/youwei_home/SVJ_Model/s-channel/Analysis'
OUTPUT_FILE = (analysis_script[-13:-10] + analysis_script[-4]
               + "_rinv" + str(int(rinv * 10))
               + "_Lambdad" + str(Lambda_d)
               + "_" + vers)

df_xdxdx_23.to_csv(OUTPUT_PATH + "/" + OUTPUT_FILE + "_S23.csv", index=False)
df_xdxdx_71.to_csv(OUTPUT_PATH + "/" + OUTPUT_FILE + "_S71.csv", index=False)
np.savez_compressed(OUTPUT_PATH + "/" + OUTPUT_FILE + ".npz",
                    truth_jet=list_truth_jet,
                    truth_jet_filterDM=list_truth_jet_filterDM,
                    truth_MET=list_truth_MET,
                    detector_jet=list_detector_jet)
np.savez_compressed(OUTPUT_PATH + "/" + OUTPUT_FILE + "_Weight.npz",
                    weight=event_weight)


print("The end of the program!!")


end = datetime.datetime.now()
# print('Time =', end - start)
print("\033[33mProgram Execution Time = {}\033[0m".format(end - start))
