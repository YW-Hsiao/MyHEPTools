"""
Program: This module is to analyze the SVJ data and
         using math, numpy, and prettytable.
         There are transverse momentum, invariant mass, 2 transverse mass,
         features of dark quarks.

         For this module, I follow
         https://github.com/YW-Hsiao/MyHEPTools/blob/main/svj_analysis/myhep/analytical_function_v2.py
         and use this format of particle information (v2)
         https://github.com/YW-Hsiao/MyHEPTools/blob/main/svj_analysis/myhep/particle_information_v2.py

Author: You-Wei Hsiao
Institute: Department of Physics, National Tsing Hua University, Hsinchu, Taiwan
Mail: hsiao.phys@gapp.nthu.edu.tw
History: 2022/03/18 First release and the version 3.
Version: v.3.0
"""


################################################################################
#                              1. Import Packages                              #
################################################################################
# The Python Standard Library

# The Third-Party Library
import math
import numpy as np
import pandas as pd
import prettytable
import pyjet


################################################################################
#                        2. Define Physical Quantities                         #
################################################################################
# 2-1. Invariant mass M
def M(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1 * \
        np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2 * \
        np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    return np.sqrt((e1+e2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2)


# 2-2. Transverse mass MT
def MT(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1 * \
        np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2 * \
        np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    ET1, ET2 = np.sqrt(m1**2 + pt1**2), np.sqrt(m2**2 + pt2**2)
    return np.sqrt((ET1+ET2)**2 - (px1+px2)**2 - (py1+py2)**2)


# 2-3. Transverse mass mT is invariant under Lorentz boost along the z direction.
def mT(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1 * \
        np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2 * \
        np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    return np.sqrt((e1+e2)**2 - (pz1+pz2)**2)


# 2-4. Transverse energy ET
def ET(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1 * \
        np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2 * \
        np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    m12 = np.sqrt((e1+e2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2)
    return np.sqrt(m12**2 + (px1+px2)**2 + (py1+py2)**2)


################################################################################
#                    3. Analyze Parton and Truth Level Data                    #
################################################################################
# 3-1. Analyze the dark quark pair, xd and xdx
def analyze_xdxdx(GP, status=23):
    """
    GP=GenParticle, _=list, i=i-th event, df=dataframe,
    acc=accumulate, tem=temporary
    """
    _M, _MT, _mT, _ET = [], [], [], []
    _Dphi, _Deta, _eta_xd, _eta_xdx = [], [], [], []
    _error = []
    acc = 0
    for i in range(GP.length):
        dfGP = GP.dataframelize(i)
        dfGP_DQ_Status = dfGP[(abs(dfGP['PID']) == 4900101)
                              & (dfGP['Status'] == status)]
        m1 = dfGP_DQ_Status.iloc[0, 6]
        pt1 = dfGP_DQ_Status.iloc[0, 7]
        eta1 = dfGP_DQ_Status.iloc[0, 8]
        phi1 = dfGP_DQ_Status.iloc[0, 9]
        m2 = dfGP_DQ_Status.iloc[1, 6]
        pt2 = dfGP_DQ_Status.iloc[1, 7]
        eta2 = dfGP_DQ_Status.iloc[1, 8]
        phi2 = dfGP_DQ_Status.iloc[1, 9]
        eta_xd = dfGP_DQ_Status[(dfGP_DQ_Status['PID'] == 4900101)].iat[0, 8]
        eta_xdx = dfGP_DQ_Status[(dfGP_DQ_Status['PID'] == -4900101)].iat[0, 8]

        _M.append(M(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2))
        _MT.append(MT(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2))
        _mT.append(mT(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2))
        _ET.append(ET(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2))
        Dphi, Deta = abs(phi1 - phi2), abs(eta1 - eta2)
        if Dphi > np.pi:
            _Dphi.append(2*np.pi - Dphi)
        else:
            _Dphi.append(Dphi)
        _Deta.append(Deta)
        _eta_xd.append(eta_xd)
        _eta_xdx.append(eta_xdx)

        if dfGP_DQ_Status.shape[0] != 2:
            acc += 1
            _error.append(i)

    # M_xdxdx, MT_xdxdx = np.array(_M), np.array(_MT)
    # mT_xdxdx, ET_xdxdx = np.array(_mT), np.array(_ET)
    # Dphi_xdxdx, Deta_xdxdx = np.array(_Dphi), np.array(_Deta)
    data_xdxdx = {"M_xdxdx": _M, "MT_xdxdx": _MT,
                  "mT_xdxdx": _mT, "ET_xdxdx": _ET,
                  "Dphi_xdxdx": _Dphi, "Deta_xdxdx": _Deta,
                  "eta_xd": _eta_xd, "eta_xdx": _eta_xdx}
    df_xdxdx = pd.DataFrame(data_xdxdx)
    if acc == 0:
        print("For status = {}, all events only include 2 particles.".format(status))
    else:
        print("{} events are over 2 particles.".format(acc))
        print(_error)
    return df_xdxdx


################################################################################
#                              4. Jet Clustering                               #
################################################################################
# 4-1. Select stable final state particle and filter out DM
def selectStableFinalStateParticle(GP,
                                   filter=[51, -51, 53, -53, 4900211, -4900211, 4900213, -4900213]):
    """
    GP=GenParticle, _=list, i=i-th event, df=dataframe,
    acc=accumulate, tem=temporary
    Status=1 is stable final state.
    """
    _SFSP, _SFSP_filterDM = [], []
    if len(filter) == 0:
        print("There is no dark matter.")
    elif len(filter) == 1:
        print("The PID of dark matter is {}.".format(filter))
    else:
        print("The PID of dark matter are {}.".format(filter))
    for i in range(GP.length):
        dfGP = GP.dataframelize(i)
        # stable final state particle
        dfGP_Status1 = dfGP[(dfGP['Status'] == 1)]
        # without filtering DM
        dfGP_Status1_tonumpy = dfGP_Status1.to_numpy()
        dfGP_Status1_tonumpy_trans = np.transpose(dfGP_Status1_tonumpy)
        pid = dfGP_Status1_tonumpy_trans[0]
        m = dfGP_Status1_tonumpy_trans[6]
        pT = dfGP_Status1_tonumpy_trans[7]
        eta = dfGP_Status1_tonumpy_trans[8]
        phi = dfGP_Status1_tonumpy_trans[9]
        _arr_pT_eta_phi_m_pid = np.stack((pT, eta, phi, m, pid))
        _SFSP.append(_arr_pT_eta_phi_m_pid)

        # with filtering DM
        dfGP_Status1_filter = dfGP_Status1[~dfGP_Status1['PID'].isin(filter)]
        dfGP_Status1_filter_tonumpy = dfGP_Status1_filter.to_numpy()
        dfGP_Status1_filter_tonumpy_trans = np.transpose(
            dfGP_Status1_filter_tonumpy)
        pid = dfGP_Status1_filter_tonumpy_trans[0]
        m = dfGP_Status1_filter_tonumpy_trans[6]
        pT = dfGP_Status1_filter_tonumpy_trans[7]
        eta = dfGP_Status1_filter_tonumpy_trans[8]
        phi = dfGP_Status1_filter_tonumpy_trans[9]
        _arr_pT_eta_phi_m_pid = np.stack((pT, eta, phi, m, pid))
        _SFSP_filterDM.append(_arr_pT_eta_phi_m_pid)

    print("{} events are stable final state.".format(len(_SFSP)))
    print("{} events are stable final state without DM.".format(len(_SFSP_filterDM)))
    return _SFSP, _SFSP_filterDM


# 4-2. Jet clustering
def jetClustering(list_SFSP, R, p=-1, pTmin=200):
    """
    list_SFSP=from selecting stable final state particle,
    _=list, i=i-th event,
    R=the cone size of the jet
    p=the jet clustering algorithm: -1=anti-kt, 0=Cambridge-Aachen(C/A), 1=kt
    pTmin=the minimum pT of jet
    """
    _PseudoJet = []
    for i in range(len(list_SFSP)):
        vectors_i = np.core.records.fromarrays(list_SFSP[i],
                                               dtype=np.dtype([('pT', np.float64),
                                                               ('eta', np.float64),
                                                               ('phi', np.float64),
                                                               ('mass', np.float64),
                                                               ('PID', np.float64)]))
        sequence_i = pyjet.cluster(vectors_i, R=R, p=p)
        jets_i = sequence_i.inclusive_jets(pTmin)  # list of PseudoJets
        _PseudoJet.append(jets_i)

    return _PseudoJet


################################################################################
#                    5. Analyze the Jets in the Truth Level                    #
################################################################################
# 5-1. Preselection
def preselection_v1(PseudoJet, pT_min, eta_max):
    """
    'preselection' function is to do the event preselection, such as minimal pT and
    maximal eta,
    * output formats are LISTs of before preselection, after pT preselection,
    * and after pT & eta preselection.
    * v1 uses the simple algorithm.
    ! If there is no PseudoJet in _event_i, I put the empty list in _event_i.
    PseudoJet=[ event_i(PseudoJet) ] is a list to store all events.
    pT_min=minial pT, eta_max=maximal eta
    _=list, i=i-th event,
    _presel_events=preselection events,
    _num_events=number of events,
    _event_i=temporary list of components (pt, eta, phi, mass) of each event
    _idx=record i-th event when number of PseudoJet = 0
    """
    _presel_events_bef, _presel_events_pt, _presel_events_pt_eta = [], [], []
    _num_events_bef, _num_events_pt, _num_events_pt_eta = [], [], []
    _idx_bef, _idx_pt, _idx_pt_eta = [], [], []
    for i in range(len(PseudoJet)):
        _event_i_bef, _event_i_pt, _event_i_pt_eta = [], [], []
        for j, jet in enumerate(PseudoJet[i]):
            # before preselection
            _event_i_bef.append((jet.pt, jet.eta, jet.phi, jet.mass))
            #
            # pT preselection
            if jet.pt > pT_min:
                _event_i_pt.append((jet.pt, jet.eta, jet.phi, jet.mass))
                # print(j, jet)
                #
                # eta preselection
                if abs(jet.eta) < eta_max:
                    _event_i_pt_eta.append(
                        (jet.pt, jet.eta, jet.phi, jet.mass))

        # record the event when number of PseudoJet = 0
        if len(_event_i_pt_eta) == 0:
            _idx_pt_eta.append(i)
            if len(_event_i_pt) == 0:
                _idx_pt.append(i)
                if len(_event_i_bef) == 0:
                    _idx_bef.append(i)

        # define np.array() with dtype
        arr_event_i_bef = np.array(_event_i_bef, dtype=[('pT', '<f8'), ('eta', '<f8'),
                                                        ('phi', '<f8'), ('mass', '<f8')])
        arr_event_i_pt = np.array(_event_i_pt, dtype=[('pT', '<f8'), ('eta', '<f8'),
                                                      ('phi', '<f8'), ('mass', '<f8')])
        arr_event_i_pt_eta = np.array(_event_i_pt_eta, dtype=[('pT', '<f8'), ('eta', '<f8'),
                                                              ('phi', '<f8'), ('mass', '<f8')])
        # append number of events and np.array() to list
        _num_events_bef.append(arr_event_i_bef.shape[0])
        _num_events_pt.append(arr_event_i_pt.shape[0])
        _num_events_pt_eta.append(arr_event_i_pt_eta.shape[0])
        _presel_events_bef.append(arr_event_i_bef)
        _presel_events_pt.append(arr_event_i_pt)
        _presel_events_pt_eta.append(arr_event_i_pt_eta)

    _idx = [_idx_bef, _idx_pt, _idx_pt_eta]
    print("{} events before preselection".format(len(_presel_events_bef)))
    print("{} events after pT preselection".format(len(_presel_events_pt)))
    print("{} events after pT & eta preselections".format(
        len(_presel_events_pt_eta)))
    print("-"*80)
    print("{} events without PseudoJet before preselection".format(len(_idx_bef)))
    print("{} events without PseudoJet after pT preselection".format(len(_idx_pt)))
    print("{} events without PseudoJet after pT & eta preselections".format(
        len(_idx_pt_eta)))

    return _presel_events_bef, _presel_events_pt, _presel_events_pt_eta, _idx


# Change mT12 to mT
# Add M_123, M_1234, MT_123, MT_1234, mT_123, mT_1234.
