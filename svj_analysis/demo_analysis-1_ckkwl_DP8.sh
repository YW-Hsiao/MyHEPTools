#!/bin/bash
# Program:
#     This program is to do the analysis about Python script analysis-1_ckkwl_DP8.py.
#     rinv=0.3, Lambda_d=5
#     
# Author: You-Wei Hsiao
# Institute: Department of Physics, National Tsing Hua University, Hsinchu, Taiwan
# Mail: hsiao.phys@gapp.nthu.edu.tw
# History: 2021/09/27
#     First release
# Version: v.1.0


# Setup environment and variables
today=$(date +"%Y/%m/%d")
path_data="/youwei_home/SVJ_CKKWL/s-channel_ckkwl-v2/DP8"
path_python_script="/youwei_home/MyHEPTools/svj_analysis"
python_script="analysis-1_ckkwl_DP8.py"
path_output="/youwei_home/SVJ_CKKWL/s-channel_ckkwl-v2/Analysis/analysis-1_ckkwl84"
rinv=0.3
Lambdad=5
version=1


# Check your .root file path and all Python parameters
echo "Date: $(date)"
echo "Start Running"
echo "Path of the shell script: $path_output"
echo "Path of the .root file: $path_data"
echo "Path of the Python script: $path_python_script"
echo "Path of the output data: $path_output"

echo "Analyze the rinv = $rinv and Lambda_d = $Lambdad data using the $python_script."
python3 $path_python_script/analysis-1_ckkwl_DP8.py $path_data/svj_ckkwl-4.root 0.4 -1 0 2 0 440 60 1.2 0 0 10000 $path_output $rinv $Lambdad $version > $path_output/ana1_rinv3_Lambdad5_1.log 2>&1


echo "The end of the shell script."
echo "Date: $(date)"
