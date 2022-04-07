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
# 2-1. Mass quantities of two objects
# A. Invariant mass M
def M(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1 * \
        np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2 * \
        np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    return np.sqrt((e1+e2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2)


# B. Transverse mass MT
def MT(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1 * \
        np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2 * \
        np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    ET1, ET2 = np.sqrt(m1**2 + pt1**2), np.sqrt(m2**2 + pt2**2)
    return np.sqrt((ET1+ET2)**2 - (px1+px2)**2 - (py1+py2)**2)


# C. Transverse mass mT
# * mT is invariant under Lorentz boost along the z direction.
def mT(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1 * \
        np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2 * \
        np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    return np.sqrt((e1+e2)**2 - (pz1+pz2)**2)


# D. Transverse energy ET
def ET(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1 * \
        np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2 * \
        np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    m12 = np.sqrt((e1+e2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2)
    return np.sqrt(m12**2 + (px1+px2)**2 + (py1+py2)**2)


# 2-2. Mass quantities of three objects
# A. Invariant mass M
def M_123(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2, m3, pt3, eta3, phi3):
    """Invariant mass M of 3 objects, M is defined by M^2 = E^2 - p^2.

    Parameters
    ----------
    m1, m2, m3 : float
        Mass of object i.
    pt1, pt2, pt3 : float
        Transverse momentum of object i.
    eta1, eta2, eta3 : float
        Pseudorapidity of object i.
    phi1, phi2, phi3 : float
        Azimuthal angle of object i.

    Returns
    -------
    float
        Invariant mass M.
    """
    # object 1
    px1, py1 = pt1 * np.cos(phi1), pt1 * np.sin(phi1)
    pz1 = np.sqrt(m1**2 + pt1**2) * np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    # object 2
    px2, py2 = pt2 * np.cos(phi2), pt2 * np.sin(phi2)
    pz2 = np.sqrt(m2**2 + pt2**2) * np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    # object 3
    px3, py3 = pt3 * np.cos(phi3), pt3 * np.sin(phi3)
    pz3 = np.sqrt(m3**2 + pt3**2) * np.sinh(eta3)
    e3 = np.sqrt(m3**2 + px3**2 + py3**2 + pz3**2)
    return np.sqrt((e1 + e2 + e3)**2
                   - (px1 + px2 + px3)**2
                   - (py1 + py2 + py3)**2
                   - (pz1 + pz2 + pz3)**2)


# B. Transverse mass MT
def MT_123(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2, m3, pt3, eta3, phi3):
    """Transverse mass MT of 3 objects, MT is defined by
    M_T^2 = (E_T(1) + E_T(2))^2 - (p_T(1) + p_T(2))^2.

    Parameters
    ----------
    m1, m2, m3 : float
        Mass of object i.
    pt1, pt2, pt3 : float
        Transverse momentum of object i.
    eta1, eta2, eta2 : float
        Pseudorapidity of object i.
    phi1, phi2, phi3 : float
        Azimuthal angle of object i.

    Returns
    -------
    float
        Transverse mass MT.
    """
    # object 1
    px1, py1 = pt1 * np.cos(phi1), pt1 * np.sin(phi1)
    pz1 = np.sqrt(m1**2 + pt1**2) * np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    et1 = np.sqrt(m1**2 + pt1**2)
    # object 2
    px2, py2 = pt2 * np.cos(phi2), pt2 * np.sin(phi2)
    pz2 = np.sqrt(m2**2 + pt2**2) * np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    et2 = np.sqrt(m2**2 + pt2**2)
    # object 3
    px3, py3 = pt3 * np.cos(phi3), pt3 * np.sin(phi3)
    pz3 = np.sqrt(m3**2 + pt3**2) * np.sinh(eta3)
    e3 = np.sqrt(m3**2 + px3**2 + py3**2 + pz3**2)
    et3 = np.sqrt(m3**2 + pt3**2)
    return np.sqrt((et1 + et2 + et3)**2
                   - (px1 + px2 + px3)**2
                   - (py1 + py2 + py3)**2)


# C. Transverse mass mT with definition 1
# * mT is invariant under Lorentz boost along the z direction.
def mT_123_def_1(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2, m3, pt3, eta3, phi3):
    """Transverse mass mT of 3 objects, mT is defined by m_T^2 = E^2 - p_z^2.

    Parameters
    ----------
    m1, m2, m3 : float
        Mass of object i.
    pt1, pt2, pt3 : float
        Transverse mass of object i.
    eta1, eta2, eta3 : float
        Pseudorapidity of object i.
    phi1, phi2, phi3 : float
        Azimuthal angle of object i.

    Returns
    -------
    float
        Transverse mass mT.
    """
    # object 1
    px1, py1 = pt1 * np.cos(phi1), pt1 * np.sin(phi1)
    pz1 = np.sqrt(m1**2 + pt1**2) * np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    # object 2
    px2, py2 = pt2 * np.cos(phi2), pt2 * np.sin(phi2)
    pz2 = np.sqrt(m2**2 + pt2**2) * np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    # object 3
    px3, py3 = pt3 * np.cos(phi3), pt3 * np.sin(phi3)
    pz3 = np.sqrt(m3**2 + pt3**2) * np.sinh(eta3)
    e3 = np.sqrt(m3**2 + px3**2 + py3**2 + pz3**2)
    return np.sqrt((e1 + e2 + e3)**2 - (pz1 + pz2 + pz3)**2)


# D. Transverse mass mT with definition 2 or Transverse energy ET
# * mT is invariant under Lorentz boost along the z direction.
def mT_123_def_2(m1, pt1, eta1, phi1, m2, pt2, eta2, phi2, m3, pt3, eta3, phi3):
    """Transverse mass mT of 3 objects, mT is defined by m_T^2 = m^2 + p_T^2.

    Parameters
    ----------
    m1, m2, m3 : float
        Mass of object i.
    pt1, pt2, pt3 : float
        Transverse momentum of object i.
    eta1, eta2, eta3 : float
        Pseudorapidity of object i.
    phi1, phi2, phi3 : float
        Azimuthal angle of object i.

    Returns
    -------
    float
        Transverse mass mT.
    """
    # object 1
    px1, py1 = pt1 * np.cos(phi1), pt1 * np.sin(phi1)
    pz1 = np.sqrt(m1**2 + pt1**2) * np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    # object 2
    px2, py2 = pt2 * np.cos(phi2), pt2 * np.sin(phi2)
    pz2 = np.sqrt(m2**2 + pt2**2) * np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    # object 3
    px3, py3 = pt3 * np.cos(phi3), pt3 * np.sin(phi3)
    pz3 = np.sqrt(m3**2 + pt3**2) * np.sinh(eta3)
    e3 = np.sqrt(m3**2 + px3**2 + py3**2 + pz3**2)
    # invariant mass of 3 objects
    m123 = np.sqrt((e1 + e2 + e3)**2 - (px1 + px2 + px3)**2
                   - (py1 + py2 + py3)**2 - (pz1 + pz2 + pz3)**2)
    return np.sqrt(m123**2 + (px1 + px2 + px3)**2 + (py1 + py2 + py3)**2)


# 2-3. Mass quantities of four objects


################################################################################
#                    3. Analyze Parton and Truth Level Data                    #
################################################################################
# 3-1. Analyze the dark quark pair, xd and xdx
def analyze_xdxdx(GP, status=23):
    """
    Analyze the dark quark pair with different status.
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
# A. Version 1
def jetClustering_v1(list_SFSP, R, p=-1, pTmin=20):
    """
    Do the jet clustering for each event.
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


# B. Version 2
# ! Use np.array([(), (), ...], dtype=[(), (), ...])


################################################################################
#                    5. Analyze the Jets in the Truth Level                    #
################################################################################
# 5-1. Preselection
# A. Version 1
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


# B. Version 2
# ! Use pt < pT_min and continue in loop


# 5-2. MET
# ! Use GenParticle
# ! Use PseudoJet before and after preselections


# 5-3. Analyze the truth jet (Scheme 1)
# A. Version 1
def analyze_truthJet_scheme1_v1(data_presel):
    """This function is my scheme 1 with version 1 which analyzes jet in truth level.

    Parameters
    ----------
    data_presel : list
        Record all preselected events in list data_presel.

    Returns
    -------
    (array 1, array 2, array 3)
        array 1 is number of jet, array 2 is observables of dijet,
        and array 3 is observables of trijet.
    """
    # * _=list, i=i-th event
    _N_jet = []
    _pT_1_jj, _pT_2_jj, _eta_1_jj, _eta_2_jj = [], [], [], []
    _M_jj, _MT_jj, _mT_jj, _Dphi, _Deta = [], [], [], [], []
    _pT_1_jjj, _pT_2_jjj, _pT_3_jjj, _eta_1_jjj, _eta_2_jjj, _eta_3_jjj = [], [], [], [], [], []
    _M_jjj, _MT_jjj, _mT_jjj = [], [], []
    _idx_jj, _idx_jjj = [], []
    for i in range(len(data_presel)):
        # number of jets
        _N_jet.append(data_presel[i].shape[0])
        # at least dijet
        if np.shape(data_presel[i])[0] >= 2:
            pt, eta = data_presel[i]['pT'], data_presel[i]['eta']
            phi, mass = data_presel[i]['phi'], data_presel[i]['mass']
            _pT_1_jj.append(pt[0])
            _pT_2_jj.append(pt[1])
            _eta_1_jj.append(eta[0])
            _eta_2_jj.append(eta[1])
            _M_jj.append(M(mass[0], pt[0], eta[0], phi[0],
                           mass[1], pt[1], eta[1], phi[1]))
            _MT_jj.append(MT(mass[0], pt[0], eta[0], phi[0],
                             mass[1], pt[1], eta[1], phi[1]))
            _mT_jj.append(mT(mass[0], pt[0], eta[0], phi[0],
                             mass[1], pt[1], eta[1], phi[1]))
            Dphi = abs(phi[0] - phi[1])
            if Dphi > np.pi:
                _Dphi.append(2*np.pi - Dphi)
            else:
                _Dphi.append(Dphi)
            _Deta.append(abs(eta[0] - eta[1]))
            # index of selected event
            _idx_jj.append(i)
        # at least trijet
        if np.shape(data_presel[i])[0] >= 3:
            pt, eta = data_presel[i]['pT'], data_presel[i]['eta']
            phi, mass = data_presel[i]['phi'], data_presel[i]['mass']
            _pT_1_jjj.append(pt[0])
            _pT_2_jjj.append(pt[1])
            _pT_3_jjj.append(pt[2])
            _eta_1_jjj.append(eta[0])
            _eta_2_jjj.append(eta[1])
            _eta_3_jjj.append(eta[2])
            _M_jjj.append(M_123(mass[0], pt[0], eta[0], phi[0],
                                mass[1], pt[1], eta[1], phi[1],
                                mass[2], pt[2], eta[2], phi[2]))
            _MT_jjj.append(MT_123(mass[0], pt[0], eta[0], phi[0],
                                  mass[1], pt[1], eta[1], phi[1],
                                  mass[2], pt[2], eta[2], phi[2]))
            _mT_jjj.append(mT_123_def_1(mass[0], pt[0], eta[0], phi[0],
                                        mass[1], pt[1], eta[1], phi[1],
                                        mass[2], pt[2], eta[2], phi[2]))
            # index of selected event
            _idx_jjj.append(i)
    # collect observables to np.array()
    arr_N_jet = np.array(_N_jet)
    arr_jj = np.array([_pT_1_jj, _pT_2_jj, _eta_1_jj, _eta_2_jj, _M_jj,
                       _MT_jj, _mT_jj, _Dphi, _Deta, _idx_jj])
    arr_jjj = np.array([_pT_1_jjj, _pT_2_jjj, _pT_3_jjj, _eta_1_jjj, _eta_2_jjj,
                        _eta_3_jjj, _M_jjj, _MT_jjj, _mT_jjj, _idx_jjj])

    print("{} events in the array of number of jets.".format(len(_N_jet)))
    print("{} selected events in dijet.".format(len(_idx_jj)))
    print("{} selected events in trijet.".format(len(_idx_jjj)))
    return arr_N_jet, arr_jj, arr_jjj


# 5-4. Analyze the truth jet with MET


################################################################################
#                   6. Analyze the Jets in the Detector Level                  #
################################################################################
