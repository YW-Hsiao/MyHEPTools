"""
Program: This module is to analyze particle data and
         using math, numpy, and prettytable.
         This module is just for the analyzing CKKW-L of DP8.
         
         For this module, I do the many test in
         /youwei_home/SVJ_CKKWL/s-channel_ckkwl-v2/Analysis/Test_analysis_DP8.ipynb.
         
Author: You-Wei Hsiao
Institute: Department of Physics, National Tsing Hua University, Hsinchu, Taiwan
Mail: hsiao.phys@gapp.nthu.edu.tw
History: 2021/09/27
         First release, this is the version 2 of analysis.
Version: v.2.0
"""

################################################################################
###### 1. Import Packages
################################################################################
# The Python Standard Library


# The Third-Party Library
import math
import numpy as np
import pandas as pd
import prettytable
import pyjet


################################################################################
###### 2. Design My Analysis Function Base
################################################################################
# 1. Invariant Mass and Transverse Mass
def M(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1*np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2*np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    return np.sqrt((e1+e2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2)

def MT(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1*np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2*np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    ET1, ET2 = np.sqrt(m1**2 + pt1**2), np.sqrt(m2**2 + pt2**2)
    return np.sqrt((ET1+ET2)**2 - (px1+px2)**2 - (py1+py2)**2)


# 9. Transverse Mass mT is invariant under Lorentz boost along the z direction.
def mT12(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1*np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2*np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    return np.sqrt((e1+e2)**2 - (pz1+pz2)**2)


# 14. Transverse Energy ET
def ET(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1*np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2*np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    m12 = np.sqrt((e1+e2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2)
    return np.sqrt(m12**2 + (px1+px2)**2 + (py1+py2)**2)


# 15. Parse the Dark Quark Pair xd and xdx
def parse_xdxdx_v2_ckkwl_DP8(GP, event_weight_not0, status=23):
    """
    GP=GenParticle; i=i-th event; df=dataframe; Tem=temporary; acc=accumulate
    """
    _list_M, _list_MT, _list_mT, _list_ET = [], [], [], []
    _list_Dphi, _list_Deta = [], []
    _error = []
    acc = 0
    for i in event_weight_not0:
        dfGP = GP.dataframelize(i)
        dfGP_DQ_Status = dfGP[(abs(dfGP['PID']) == 4900101) & (dfGP['Status'] == status)]
        m1 = dfGP_DQ_Status.iloc[0,6]
        pt1 = dfGP_DQ_Status.iloc[0,7]
        eta1 = dfGP_DQ_Status.iloc[0,8]
        phi1 = dfGP_DQ_Status.iloc[0,9]
        m2 = dfGP_DQ_Status.iloc[1,6]
        pt2 = dfGP_DQ_Status.iloc[1,7]
        eta2 = dfGP_DQ_Status.iloc[1,8]
        phi2 = dfGP_DQ_Status.iloc[1,9]
        
        _list_M.append(M(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2))
        _list_MT.append(MT(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2))
        _list_mT.append(mT12(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2))
        _list_ET.append(ET(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2))
        Dphi, Deta = abs(phi1 - phi2), abs(eta1 - eta2)
        if Dphi > np.pi:
            _list_Dphi.append(2*np.pi - Dphi)
        else:
            _list_Dphi.append(Dphi)
        _list_Deta.append(Deta)
        
        
        if dfGP_DQ_Status.shape[0] != 2:
            acc += 1
            _error.append(i)
            
    M_xdxdx, MT_xdxdx = np.array(_list_M), np.array(_list_MT)
    mT_xdxdx, ET_xdxdx = np.array(_list_mT), np.array(_list_ET)
    Dphi_xdxdx, Deta_xdxdx = np.array(_list_Dphi), np.array(_list_Deta)
    data_xdxdx = {"M_xdxdx": M_xdxdx, "MT_xdxdx": MT_xdxdx,
                  "mT_xdxdx": mT_xdxdx, "ET_xdxdx": ET_xdxdx,
                  "Dphi_xdxdx": Dphi_xdxdx, "Deta_xdxdx": Deta_xdxdx}
    df_xdxdx = pd.DataFrame(data_xdxdx)
    if acc == 0:
        print("All events only include 2 particles.")
    else:
        print("There are {} events with over 2 particles.".format(acc))
        print(_error)
    return df_xdxdx


# 10. Select Stable Final State Particle and Filter out DM
def selectStableFinalStateParticle_filterDM(GP, event_weight_not0, FILTER=[51, -51, 53, -53]):
#     GP=GenParticle; i=i-th event; df=dataframe;
#     Status=1 is stable final state.
    _list_SFSP, _list_SFSP_filterDM = [], []
    if len(FILTER) == 0:
        print("There is no dark matter.")
    elif len(FILTER) == 1:
        print("The PID of dark matter is {}.".format(FILTER))
    else:
        print("The PID of dark matter are {}.".format(FILTER))
    for i in event_weight_not0:
        dfGP = GP.dataframelize(i)
        dfGP_Status1 = dfGP[(dfGP['Status'] == 1)]  # Stable final state particle
        dfGP_Status1_tonumpy = dfGP_Status1.to_numpy()
        dfGP_Status1_tonumpy_trans = np.transpose(dfGP_Status1_tonumpy)
        pid = dfGP_Status1_tonumpy_trans[0]
        m = dfGP_Status1_tonumpy_trans[6]
        pT = dfGP_Status1_tonumpy_trans[7]
        eta = dfGP_Status1_tonumpy_trans[8]
        phi = dfGP_Status1_tonumpy_trans[9]
        _nparray_pT_eta_phi_m_pid = np.stack((pT, eta, phi, m, pid))
        _list_SFSP.append(_nparray_pT_eta_phi_m_pid)
        
        dfGP_Status1_filter = dfGP_Status1[~dfGP_Status1['PID'].isin(FILTER)]
        dfGP_Status1_filter_tonumpy = dfGP_Status1_filter.to_numpy()
        dfGP_Status1_filter_tonumpy_trans = np.transpose(dfGP_Status1_filter_tonumpy)
        pid = dfGP_Status1_filter_tonumpy_trans[0]
        m = dfGP_Status1_filter_tonumpy_trans[6]
        pT = dfGP_Status1_filter_tonumpy_trans[7]
        eta = dfGP_Status1_filter_tonumpy_trans[8]
        phi = dfGP_Status1_filter_tonumpy_trans[9]
        _nparray_pT_eta_phi_m_pid = np.stack((pT, eta, phi, m, pid))
        _list_SFSP_filterDM.append(_nparray_pT_eta_phi_m_pid)
        
        
    print("There are {} events with stable final state.".format(len(_list_SFSP)))
    print("There are {} events with stable final state and without DM.".format(len(_list_SFSP_filterDM)))
    return _list_SFSP, _list_SFSP_filterDM


# 12. Preselect Detector Jet
# _list_results stores the 0=M_jj, 1=MT_jj, 2=mT_jj, 3=selected events,
# 4=pT1, 5=pT2, 6=Dphi_jj
def preselectDetectorJet(JET, event_weight_not0, N_JET_MIN=2, PT1_MIN=440, PT2_MIN=60, ETA_MAX=1.2):
    _list_M_jj, _list_MT_jj, _list_mT12_jj, _list_Selected = [], [], [], []
    _list_pT1, _list_pT2, _list_Dphi = [], [], []
    _error_pt_order = []
    for i in event_weight_not0:
        dfJET = JET.dataframelize(i)
        if dfJET.shape[0] < N_JET_MIN:
            continue
        elif dfJET['PT'][0] < PT1_MIN or dfJET['PT'][1] < PT2_MIN:
            continue
        elif np.abs(dfJET['Eta'][0]-dfJET['Eta'][1]) > ETA_MAX:
            continue
        _list_M_jj.append(M(dfJET['Mass'][0], dfJET['PT'][0], dfJET['Eta'][0], dfJET['Phi'][0],
                            dfJET['Mass'][1], dfJET['PT'][1], dfJET['Eta'][1], dfJET['Phi'][1]))
        _list_MT_jj.append(MT(dfJET['Mass'][0], dfJET['PT'][0], dfJET['Eta'][0], dfJET['Phi'][0],
                              dfJET['Mass'][1], dfJET['PT'][1], dfJET['Eta'][1], dfJET['Phi'][1]))
        _list_mT12_jj.append(mT12(dfJET['Mass'][0], dfJET['PT'][0], dfJET['Eta'][0], dfJET['Phi'][0],
                                  dfJET['Mass'][1], dfJET['PT'][1], dfJET['Eta'][1], dfJET['Phi'][1]))
        _list_Selected.append(i)
        _list_pT1.append(dfJET['PT'][0])
        _list_pT2.append(dfJET['PT'][1])
        Dphi = dfJET['Phi'][0] - dfJET['Phi'][1]
        if abs(Dphi) > np.pi:
            _list_Dphi.append(2*np.pi - abs(Dphi))
        else:
            _list_Dphi.append(abs(Dphi))
            
        dfJET_order = dfJET.sort_values(by=['PT'], ascending=False)
        if dfJET_order.index[0] != 0 or dfJET_order.index[1] != 1:
            _error_pt_order.append(i)
    if len(_error_pt_order) == 0:
        print("The order of jets all are no error!!")
    else:
        print("Errors are in events {}.".format(_error_pt_order))
        
        
    print("There are {} selected events.".format(len(_list_Selected)))
    _list_results = [np.array(_list_M_jj), np.array(_list_MT_jj), np.array(_list_mT12_jj), np.array(_list_Selected),
                     np.array(_list_pT1), np.array(_list_pT2), np.array(_list_Dphi)]
    return _list_results
