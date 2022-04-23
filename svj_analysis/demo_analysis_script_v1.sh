#!/bin/bash
# Program:
#     This program is to do the analysis for python script analysis_script_v1.py.
#     
# Author: You-Wei Hsiao
# Institute: Department of Physics, National Tsing Hua University, Hsinchu, Taiwan
# Mail: hsiao.phys@gapp.nthu.edu.tw
# History (v.1.0): 2022/04/23 First release.


# Setup environment and variables
path_sh="/youwei_home/svj_setting_master/scheme_1/analysis_script_v1"
path_python_script="/youwei_home/MyHEPTools/svj_analysis"
python_script="analysis_script_v1.py"
path_data="/youwei_u3/svj_data_master/scheme_1/root"
R=0.4
JetClusteringAlgorithm=-1
pTmin_pyjet=0
pT_min=20
eta_max=2.5
nevents=20000
path_output="/youwei_u3/svj_data_master/scheme_1/analysis_script_v1"
rinv=0.3
Lambdad=5
remark="ckkwl_wo"

# Check your .root file path and all Python parameters
echo "Date: $(date)"
echo "*------  Start Running  ------*"
echo "Path of the shell script : $path_sh"
echo "Path of the python script: $path_python_script"
echo "Path of the .root file   : $path_data"
echo "Path of the output data  : $path_output"

echo "Using the $python_script analyzes"
echo "the rinv = $rinv and Lambda_d = $Lambdad data"
# python3 $path_python_script/analysis-1.py $path_data/svj0_run2.root 0.4 -1 0 2 0 440 60 1.2 0 0 20000 $path_output $rinv $Lambdad $version > $path_output/ana1_rinv3_Lambdad5_1.log 2>&1
# python3 analysis_script_v1.py input 0.4 -1 0 20 2.5 20000 output 0.3 5 remark
python3 $path_python_script/$python_script $path_data/ckkwl_wo.root 0.4 -1 0 20 2.5 20000 $path_output 0.3 5 $remark > $path_output/rinv3_Lambdad5_ckkwl_wo.log 2>&1

echo "*------  Done  ------*"
echo "Date: $(date)"
