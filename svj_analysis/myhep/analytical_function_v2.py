"""
Program: This module is to analyze particle data and
         using math, numpy, and prettytable.
         There are invariant mass, transverse mass, find the mass of dark quarks,
         check the rinv, and count the number of dark hadrons.
         
         For this module, I do the many test in
         /youwei_home/SVJ_Model/s-channel/Check_truth-record.ipynb.
         
Author: You-Wei Hsiao
Institute: Department of Physics, National Tsing Hua University, Hsinchu, Taiwan
Mail: hsiao.phys@gapp.nthu.edu.tw
History: 2021/04/17
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


# 2. Find the Mass of xd and xd~
def findMxdxdbar(GP):
#     GP=GenParticle; i=i-th event; df=dataframe; Tem=temporary; acc=accumulate
    listM, _list = [], []
    acc = 0
    for i in range(GP.length):
        dfGP = GP.dataframelize(i)
        dfTem = dfGP[(abs(dfGP['PID']) == 4900101) & (dfGP['Status'] == 23)]
        m1 = dfTem.iloc[0,6]
        pt1 = dfTem.iloc[0,7]
        eta1 = dfTem.iloc[0,8]
        phi1 = dfTem.iloc[0,9]
        
        m2 = dfTem.iloc[1,6]
        pt2 = dfTem.iloc[1,7]
        eta2 = dfTem.iloc[1,8]
        phi2 = dfTem.iloc[1,9]
        listM.append(M(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2))
        
        if dfTem.shape[0] != 2:
            acc += 1
            _list.append(i)
            
            
    m_xdxdbar = np.array(listM)
    dataM = {"m_xdxdbar": m_xdxdbar}
    dfMxdxdbar = pd.DataFrame(dataM)
    if acc == 0:
        print("All events are including 2 particles.")
    else:
        print("There are {} events with over 2 particles.".format(acc))
        print(_list)
    return m_xdxdbar, dfMxdxdbar


# 3. Check r_inv = Branching Ratio
def checkrinvBRatio(GP, darkhadron=4900111, dm=51):
#     GP=GenParticle; i=i-th event; df=dataframe; Tem=temporary; acc=accumulate;
#     DH=darkhadron; dm=dark matter; tb=table
#     Status=1 is stable final state.
#     dm=51 & 53 are stable dark matter which is status=1.
    countInVis, countVis, errorDaughter, errorDecay, errorDecayAsymm = 0, 0, 0, 0, 0
    _listDaughter, _listDecay, _listDecayAsymm = [], [], []
    for i in range(GP.length):
        dfGP = GP.dataframelize(i)
        dfDH = dfGP[(abs(dfGP['PID']) == darkhadron)]
        for j, d1 in enumerate(dfDH['D1']):
            if (d1 == -1) or (dfDH['D2'].values[j] == -1):
                errorDaughter += 1
                _listDaughter.append([i,d1,dfDH['D2'].values[j]])
            elif dfGP.loc[d1,'PID'] == -dfGP.loc[dfDH['D2'].values[j],'PID']:
                if abs(dfGP.loc[d1,'PID']) < 6:
                    countVis += 1
                elif abs(dfGP.loc[d1,'PID']) == dm:
                    countInVis += 1
                else:
                    errorDecay += 1
                    _listDecay.append([i,d1])
            else:
                errorDecayAsymm += 1
                _listDecayAsymm.append([i,d1,dfDH['D2'].values[j]])
                
                
    rinv = countInVis/(countInVis+countVis)
                
    if errorDaughter == errorDecay == errorDecayAsymm == 0:
        print("No error!")
    else:
        print("There are {} events about daughters equal to 0.".format(errorDaughter))
        print(_listDaughter)
        print("There are {} events decayed to other particle rather than {} and {}.".format(errorDecay, darkhadron, dm))
        print(_listDecay)
        print("There are {} events which decay to asymmetric event.".format(errorDecayAsymm))
        print(_listDH)
    tb = prettytable.PrettyTable()
    tb.field_names = ["r_inv Branching Ratio"]
    tb.add_rows([
        ["There are {} Dark Mesons decayed into invisible particle.".format(countInVis)],
        ["There are {} Dark Mesons decayed into visible particle.".format(countVis)],
        ["r_inv = {:^6.4f}".format(rinv)]])
    tb.align = 'l'
    print(tb)
    return rinv, countInVis, countVis


# 4. Count Dark Hadron
def countDarkHadron(GP, _listDH=[4900111, -4900111, 4900113, -4900113, 4900211, -4900211, 4900213, -4900213]):
#     GP=GenParticle; i=i-th event; df=dataframe; Tem=temporary; acc=accumulate;
#     DH=darkhadron; conditionDH=PID of DH
    countDH = 0
#     conditionDH = (abs(dfGP['PID']) == 4900111) | (abs(dfGP['PID']) == 4900113) | (abs(dfGP['PID']) == 4900211) | (abs(dfGP['PID']) == 4900213)
    for i in range(GP.length):
        dfGP = GP.dataframelize(i)
        dfDH = dfGP[dfGP['PID'].isin(_listDH)]
        countDH += dfDH.shape[0]
        
    ratioDH = countDH/GP.length
    return countDH, ratioDH






"""
Program: I add 4 functions for analysis.
         They focus on to analyze the jets.
         They are select stable final state particle, do the jet clustering,
         preselect jets in truth level, and preselect jets in detector level.
History: 2021/05/15
         Second release
Version: v.2.1
"""
# 5. Select Stable Final State Particles
def selectStableFinalStateParticle(GP):
#     GP=GenParticle; i=i-th event; df=dataframe;
#     Status=1 is stable final state.
    _listSFSP = []
    for i in range(GP.length):
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
        
        _listSFSP.append(_nparray_pT_eta_phi_m_pid)
    print("There are {} events.".format(len(_listSFSP)))
    return _listSFSP


# 6. Let's Do Jet Clustering
def jetClustering(_listSFSP, R, p=-1, pTmin=200):
#     _listSFSP is truth record with Status=1 preselection.
#     R=The cone size of the jet
#     p=The jet clustering algorithm: -1=anti-kt, 0=Cambridge-Aachen(C/A), 1=kt
#     pTmin=The minimum pT of jet
    _listPseudoJet = []
    for i in range(len(_listSFSP)):
        vectors_i = np.core.records.fromarrays(_listSFSP[i],
                                               dtype=np.dtype([('pT', np.float64), ('eta', np.float64), ('phi', np.float64), ('mass', np.float64), ('PID', np.float64)]))
        sequence_i = pyjet.cluster(vectors_i, R=R, p=p)
        jets_i = sequence_i.inclusive_jets(pTmin)
        _listPseudoJet.append(jets_i)
        
    return _listPseudoJet


# 7. Preselect Jet (Detector Level)
def preselectJet(JET, N_JET=2, PT1_MIN=440, PT2_MIN=60, ETA_MAX=1.2):
    _listM_Jet, _listMT_Jet, _listSurvived = [], [], []
#     _nparray_N_JET = np.arange(N_JET+1)
    for i in range(JET.length):
        dfJET = JET.dataframelize(i)
        if dfJET.shape[0] < N_JET:
            continue
        elif dfJET['PT'][0] < PT1_MIN or dfJET['PT'][1] < PT2_MIN:
            continue
        elif np.abs(dfJET['Eta'][0]-dfJET['Eta'][1]) > ETA_MAX:
            continue
        _listM_Jet.append(M(dfJET['Mass'][0], dfJET['PT'][0], dfJET['Eta'][0], dfJET['Phi'][0],
                            dfJET['Mass'][1], dfJET['PT'][1], dfJET['Eta'][1], dfJET['Phi'][1]))
        _listMT_Jet.append(MT(dfJET['Mass'][0], dfJET['PT'][0], dfJET['Eta'][0], dfJET['Phi'][0],
                              dfJET['Mass'][1], dfJET['PT'][1], dfJET['Eta'][1], dfJET['Phi'][1]))
        _listSurvived.append(i)
        
    print("There are {} survived events.".format(len(_listSurvived)))
    return np.array(_listM_Jet), np.array(_listMT_Jet), np.array(_listSurvived)


# 8. Preselect Di-jets (Truth Level)
def preselectDiJets(_listPseudoJet):
    _listMjj, _listMTjj, _listSelected = [], [], []
    for i in range(len(_listPseudoJet)):
        if len(_listPseudoJet[i]) >= 2:  # At least 2 jets
            jet1 = _listPseudoJet[i][0]  # Leading jet
            jet2 = _listPseudoJet[i][1]  # Subleading jet
            _listMjj.append(M(jet1.mass, jet1.pt, jet1.eta, jet1.phi,
                              jet2.mass, jet2.pt, jet2.eta, jet2.phi))
            _listMTjj.append(MT(jet1.mass, jet1.pt, jet1.eta, jet1.phi,
                                jet2.mass, jet2.pt, jet2.eta, jet2.phi))
            _listSelected.append(i)
            
    print("There are {} selected events.".format(len(_listSelected)))
    return np.array(_listMjj), np.array(_listMTjj), np.array(_listSelected)






"""
Program: I add the 5 functions.
         One of functions is the other definition of transverse mass.
         They are advanced functions to analyze the jets.
         
         For the fourth release, I modify the FILTER of the function
         'selectStableFinalStateParticle_filterDM' FILTER=[51, 53]
         to FILTER=[51, -51, 53, -53].
History: 2021/05/25; 2021/07/14
         Third release; Fourth release
Version: v.2.2; v.2.3
"""
# 9. Transverse Mass mT is invariant under Lorentz boost along the z direction.
def mT12(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1*np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2*np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    return np.sqrt((e1+e2)**2 - (pz1+pz2)**2)


# 10. Select Stable Final State Particle and Filter out DM
def selectStableFinalStateParticle_filterDM(GP, FILTER=[51, -51, 53, -53]):
#     GP=GenParticle; i=i-th event; df=dataframe;
#     Status=1 is stable final state.
    _list_SFSP, _list_SFSP_filterDM = [], []
    if len(FILTER) == 0:
        print("There is no dark matter.")
    elif len(FILTER) == 1:
        print("The PID of dark matter is {}.".format(FILTER))
    else:
        print("The PID of dark matter are {}.".format(FILTER))
    for i in range(GP.length):
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


# 11. Preselect Truth Jet
# _list_results stores the 0=M_jj, 1=MT_jj, 2=mT_jj, 3=selected events,
# 4=pT1, 5=pT2, 6=Dphi_jj
def preselectTruthJet(_list_PseudoJet, N_JET_MIN=2):
    _list_M_jj, _list_MT_jj, _list_mT12_jj, _list_Selected = [], [], [], []
    _list_pT1, _list_pT2, _list_Dphi = [], [], []
    _error_pt_order = []
    for i in range(len(_list_PseudoJet)):
        if len(_list_PseudoJet[i]) >= N_JET_MIN:
            jet1 = _list_PseudoJet[i][0] # Leading jet
            jet2 = _list_PseudoJet[i][1] # Sub-leading jet
            _list_M_jj.append(M(jet1.mass, jet1.pt, jet1.eta, jet1.phi,
                                jet2.mass, jet2.pt, jet2.eta, jet2.phi))
            _list_MT_jj.append(MT(jet1.mass, jet1.pt, jet1.eta, jet1.phi,
                                  jet2.mass, jet2.pt, jet2.eta, jet2.phi))
            _list_mT12_jj.append(mT12(jet1.mass, jet1.pt, jet1.eta, jet1.phi,
                                      jet2.mass, jet2.pt, jet2.eta, jet2.phi))
            _list_Selected.append(i)
            _list_pT1.append(jet1.pt)
            _list_pT2.append(jet2.pt)
            Dphi = jet1.phi-jet2.phi
            if abs(Dphi) > np.pi:
                _list_Dphi.append(2*np.pi - abs(Dphi))
            else:
                _list_Dphi.append(abs(Dphi))
                
            _list_pt = []
            for jet in _list_PseudoJet[i]:
                _list_pt.append(jet.pt)
            _nparray_pt = np.array(_list_pt)
            _nparray_order = np.argsort(_nparray_pt)
            if _nparray_order[-1] != 0 or _nparray_order[-2] != 1:
                _error_pt_order.append(i)
    if len(_error_pt_order) == 0:
        print("The order of jets all are no error!!")
    else:
        print("Errors are in events {}.".format(_error_pt_order))
        
        
    print("There are {} selected events.".format(len(_list_Selected)))
    _list_results = [np.array(_list_M_jj), np.array(_list_MT_jj), np.array(_list_mT12_jj), np.array(_list_Selected),
                     np.array(_list_pT1), np.array(_list_pT2), np.array(_list_Dphi)]
    return _list_results


# 12. Preselect Detector Jet
# _list_results stores the 0=M_jj, 1=MT_jj, 2=mT_jj, 3=selected events,
# 4=pT1, 5=pT2, 6=Dphi_jj
def preselectDetectorJet(JET, N_JET_MIN=2, PT1_MIN=440, PT2_MIN=60, ETA_MAX=1.2):
    _list_M_jj, _list_MT_jj, _list_mT12_jj, _list_Selected = [], [], [], []
    _list_pT1, _list_pT2, _list_Dphi = [], [], []
    _error_pt_order = []
    for i in range(JET.length):
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

"""
# 13. Parse MET in Truth Level
def parseMETTruthJet(_list_SFSP_filterDM, _list_PseudoJet_filterDM, N_jet_min=2):
    _list_pTNet_invis, _list_phiNet_invis = [], []
    _list_MT_jj_MET = []
    _list_Dphi_j1_MET = []
    for i in range(len(_list_SFSP_filterDM)):
        px = _list_SFSP_filterDM[i][0] * np.cos(_list_SFSP_filterDM[i][2])
        py = _list_SFSP_filterDM[i][0] * np.sin(_list_SFSP_filterDM[i][2])
        pxNet_invis = -np.sum(px)
        pyNet_invis = -np.sum(py)
        pTNet_invis = np.sqrt(pxNet_invis**2 + pyNet_invis**2)
        if pxNet_invis > 0 and pyNet_invis > 0:
            phiNet_invis = np.arctan(pyNet_invis/pxNet_invis)
        elif pxNet_invis < 0 and pyNet_invis > 0:
            phiNet_invis = np.arctan(pyNet_invis/pxNet_invis) + np.pi
        elif pxNet_invis < 0 and pyNet_invis < 0:
            phiNet_invis = np.arctan(pyNet_invis/pxNet_invis) + np.pi
        else:
            phiNet_invis = np.arctan(pyNet_invis/pxNet_invis) + 2*np.pi
        _list_pTNet_invis.append(pTNet_invis)
        _list_phiNet_invis.append(phiNet_invis)
        
        if len(_list_PseudoJet_filterDM[i]) >= N_jet_min:
            jet1 = _list_PseudoJet_filterDM[i][0]  # j1 = [E1, px1, py1, pz1]
            jet2 = _list_PseudoJet_filterDM[i][1]  # j2 = [E2, px2, py2, pz2]
            px1, px2 = jet1.pt * np.cos(jet1.phi), jet2.pt * np.cos(jet2.phi)
            py1, py2 = jet1.pt * np.sin(jet1.phi), jet2.pt * np.sin(jet2.phi)
            pz1 = np.sqrt((jet1.mass)**2 + (jet1.pt)**2) * np.sinh(jet1.eta)
            pz2 = np.sqrt((jet2.mass)**2 + (jet2.pt)**2) * np.sinh(jet2.eta)
            E1 = np.sqrt((jet1.mass)**2 + px1**2 + py1**2 + pz1**2)
            E2 = np.sqrt((jet2.mass)**2 + px2**2 + py2**2 + pz2**2)
            jj = [E1+E2, px1+px2, py1+py2, pz1+pz2]  # jj = [Ejj, pxjj, pyjj, pzjj]
            ET1 = np.sqrt((jet1.pt)**2 + (jet1.mass)**2)
            ET2 = np.sqrt((jet2.pt)**2 + (jet2.mass)**2)
            mjj = np.sqrt((E1+E2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2)
            ETjj = np.sqrt((px1+px2)**2 + (py1+py2)**2 + mjj**2)
            METx = pxNet_invis  # = pTNet_invis * np.cos(phiNet_invis)
            METy = pyNet_invis  # = pTNet_invis * np.sin(phiNet_invis)
            METz = 0
            MET = np.sqrt(METx**2 + METy**2 + METz**2)  # MET = [METx, METy, METz]
            MT_jj_MET = np.sqrt((ETjj+MET)**2 - (px1+px2+METx)**2 - (py1+py2+METy)**2)
            _list_MT_jj_MET.append(MT_jj_MET)
            
            Dphi_j1_MET = jet1.phi - phiNet_invis
            if abs(Dphi_j1_MET) > np.pi:
                _list_Dphi_j1_MET.append(2*np.pi - abs(Dphi_j1_MET))
            else:
                _list_Dphi_j1_MET.append(abs(Dphi_j1_MET))
                
    _list_results = [np.array(_list_pTNet_invis), np.array(_list_phiNet_invis),
                     np.array(_list_MT_jj_MET), np.array(_list_Dphi_j1_MET)]
    print("There are {} invisible events.".format(len(_list_pTNet_invis)))
    print("There are {} jet events.".format(len(_list_MT_jj_MET)))
    return _list_results
"""





"""
Program: I add the 2 functions.
         ET and the function analyzes the dark quark pair.
History: 2021/07/14
         Fourth release
Version: v.2.3
"""
# 14. Transverse Energy ET
def ET(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1*np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2*np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    m12 = np.sqrt((e1+e2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2)
    return np.sqrt(m12**2 + (px1+px2)**2 + (py1+py2)**2)


# 15. Parse the Dark Quark Pair xd and xdx
def parse_xdxdx_v2(GP, status=23):
    """
    GP=GenParticle; i=i-th event; df=dataframe; Tem=temporary; acc=accumulate
    """
    _list_M, _list_MT, _list_mT, _list_ET = [], [], [], []
    _list_Dphi, _list_Deta = [], []
    _error = []
    acc = 0
    for i in range(GP.length):
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






"""
Program: I modify a function which is parseMETTruthJet.
         Add the _list_Selected.
History: 2021/07/23
         Fifth release
Version: v.2.4
"""
# 16. Parse MET in Truth Level
def parseMETTruthJet(_list_SFSP_filterDM, _list_PseudoJet_filterDM, N_jet_min=2):
    _list_pTNet_invis, _list_phiNet_invis = [], []
    _list_MT_jj_MET = []
    _list_Dphi_j1_MET = []
    _list_Selected = []
    for i in range(len(_list_SFSP_filterDM)):
        px = _list_SFSP_filterDM[i][0] * np.cos(_list_SFSP_filterDM[i][2])
        py = _list_SFSP_filterDM[i][0] * np.sin(_list_SFSP_filterDM[i][2])
        pxNet_invis = -np.sum(px)
        pyNet_invis = -np.sum(py)
        pTNet_invis = np.sqrt(pxNet_invis**2 + pyNet_invis**2)
        if pxNet_invis > 0 and pyNet_invis > 0:
            phiNet_invis = np.arctan(pyNet_invis/pxNet_invis)
        elif pxNet_invis < 0 and pyNet_invis > 0:
            phiNet_invis = np.arctan(pyNet_invis/pxNet_invis) + np.pi
        elif pxNet_invis < 0 and pyNet_invis < 0:
            phiNet_invis = np.arctan(pyNet_invis/pxNet_invis) + np.pi
        else:
            phiNet_invis = np.arctan(pyNet_invis/pxNet_invis) + 2*np.pi
        _list_pTNet_invis.append(pTNet_invis)
        _list_phiNet_invis.append(phiNet_invis)
        
        if len(_list_PseudoJet_filterDM[i]) >= N_jet_min:
            jet1 = _list_PseudoJet_filterDM[i][0]  # j1 = [E1, px1, py1, pz1]
            jet2 = _list_PseudoJet_filterDM[i][1]  # j2 = [E2, px2, py2, pz2]
            px1, px2 = jet1.pt * np.cos(jet1.phi), jet2.pt * np.cos(jet2.phi)
            py1, py2 = jet1.pt * np.sin(jet1.phi), jet2.pt * np.sin(jet2.phi)
            pz1 = np.sqrt((jet1.mass)**2 + (jet1.pt)**2) * np.sinh(jet1.eta)
            pz2 = np.sqrt((jet2.mass)**2 + (jet2.pt)**2) * np.sinh(jet2.eta)
            E1 = np.sqrt((jet1.mass)**2 + px1**2 + py1**2 + pz1**2)
            E2 = np.sqrt((jet2.mass)**2 + px2**2 + py2**2 + pz2**2)
            jj = [E1+E2, px1+px2, py1+py2, pz1+pz2]  # jj = [Ejj, pxjj, pyjj, pzjj]
            ET1 = np.sqrt((jet1.pt)**2 + (jet1.mass)**2)
            ET2 = np.sqrt((jet2.pt)**2 + (jet2.mass)**2)
            mjj = np.sqrt((E1+E2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2)
            ETjj = np.sqrt((px1+px2)**2 + (py1+py2)**2 + mjj**2)
            METx = pxNet_invis  # = pTNet_invis * np.cos(phiNet_invis)
            METy = pyNet_invis  # = pTNet_invis * np.sin(phiNet_invis)
            METz = 0
            MET = np.sqrt(METx**2 + METy**2 + METz**2)  # MET = [METx, METy, METz]
            MT_jj_MET = np.sqrt((ETjj+MET)**2 - (px1+px2+METx)**2 - (py1+py2+METy)**2)
            _list_MT_jj_MET.append(MT_jj_MET)
            
            Dphi_j1_MET = jet1.phi - phiNet_invis
            if abs(Dphi_j1_MET) > np.pi:
                _list_Dphi_j1_MET.append(2*np.pi - abs(Dphi_j1_MET))
            else:
                _list_Dphi_j1_MET.append(abs(Dphi_j1_MET))
                
            _list_Selected.append(i)
            
    _list_results = [np.array(_list_pTNet_invis), np.array(_list_phiNet_invis),
                     np.array(_list_MT_jj_MET), np.array(_list_Dphi_j1_MET),
                     np.array(_list_Selected)]
    print("There are {} invisible events.".format(len(_list_pTNet_invis)))
    print("There are {} jet events.".format(len(_list_MT_jj_MET)))
    return _list_results










