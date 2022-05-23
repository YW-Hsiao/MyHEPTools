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
History: 2022/03/18 First release, the version 3, packages, and basic mass functions.
Version: v.3.0
History (v.3.1): 2022/03/21 analyze_xdxdx.
History (v.3.2): 2022/03/23 Jet clustering, selectStableFinalStateParticle and jetClustering.
History (v.3.3): 2022/04/02 preselection_v1.
History (v.3.4): 2022/04/07 analyze_truthJet_scheme1_v1.
History (v.3.5): 2022/04/08 analyze_truthJet_scheme1_v2.
History (v.3.6): 2022/04/15 MET_visParticle_v1.
History (v.3.7): 2022/04/19 analyze_truthJet_MET_scheme1_v1.
History (v.3.8): 2022/05/04 add min_all,6,9_Dphi_j_MET.
History (v.3.9): 2022/05/16 Debug analyze_xdxdx when xd and xd~ are stable final state.
History (v.3.10): 2022/05/17 dark_sector and neutrino.
History (v.3.11): 2022/05/18 jet.
History (v.3.12): 2022/05/20 jet_MET.
"""


################################################################################
#                              1. Import Packages                              #
################################################################################
# The Python Standard Library
import sys

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
    _Dphi, _Deta = [], []
    _pT_xd, _pT_xdx, _eta_xd, _eta_xdx = [], [], [], []
    _error = []
    acc = 0
    for i in range(GP.length):
        dfGP = GP.dataframelize(i)
        dfGP_DQ_Status = dfGP[(abs(dfGP['PID']) == 4900101)
                              & (dfGP['Status'] == status)]
        if dfGP_DQ_Status.shape[0] != 2:
            acc += 1
            _error.append(i)
            dfGP_DQ_Status = dfGP[(abs(dfGP['PID']) == 4900101)
                                  & (dfGP['Status'] == 1)]
            if dfGP_DQ_Status.shape[0] != 2:
                sys.exit(
                    f"{i} event has no +-4900101 with status = {status} and 1.")
            print(f"* Skip event {i}")
            continue
        m1 = dfGP_DQ_Status[dfGP_DQ_Status['PID'] == 4900101].iat[0, 6]
        pt1 = dfGP_DQ_Status[dfGP_DQ_Status['PID'] == 4900101].iat[0, 7]
        eta1 = dfGP_DQ_Status[dfGP_DQ_Status['PID'] == 4900101].iat[0, 8]
        phi1 = dfGP_DQ_Status[dfGP_DQ_Status['PID'] == 4900101].iat[0, 9]
        m2 = dfGP_DQ_Status[dfGP_DQ_Status['PID'] == -4900101].iat[0, 6]
        pt2 = dfGP_DQ_Status[dfGP_DQ_Status['PID'] == -4900101].iat[0, 7]
        eta2 = dfGP_DQ_Status[dfGP_DQ_Status['PID'] == -4900101].iat[0, 8]
        phi2 = dfGP_DQ_Status[dfGP_DQ_Status['PID'] == -4900101].iat[0, 9]
        # eta_xd = dfGP_DQ_Status[(dfGP_DQ_Status['PID'] == 4900101)].iat[0, 8]
        # eta_xdx = dfGP_DQ_Status[(dfGP_DQ_Status['PID'] == -4900101)].iat[0, 8]

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
        _pT_xd.append(pt1)
        _pT_xdx.append(pt2)
        _eta_xd.append(eta1)
        _eta_xdx.append(eta2)

    # M_xdxdx, MT_xdxdx = np.array(_M), np.array(_MT)
    # mT_xdxdx, ET_xdxdx = np.array(_mT), np.array(_ET)
    # Dphi_xdxdx, Deta_xdxdx = np.array(_Dphi), np.array(_Deta)
    data_xdxdx = {"M_xdxdx": _M, "MT_xdxdx": _MT,
                  "mT_xdxdx": _mT, "ET_xdxdx": _ET,
                  "Dphi_xdxdx": _Dphi, "Deta_xdxdx": _Deta,
                  "pT_xd": _pT_xd, "pT_xdx": _pT_xdx,
                  "eta_xd": _eta_xd, "eta_xdx": _eta_xdx}
    df_xdxdx = pd.DataFrame(data_xdxdx)
    if acc == 0:
        print(
            f"For status = {status}, all events only include dark quark pair with status {status}.")
    else:
        print(f"The length of {acc} events is not equal to 2 particles.")
        print(_error)
    return df_xdxdx


# 3-2. Check the constituent of dark sector
def dark_sector(GP):
    """Check the constituent of dark sector, such as the existence of
    stable particles 4900101, 4900021, 4900111, 4900113 in final state and
    the existence of particles 4900102, 4900022.

    Parameters
    ----------
    GP : DataFrame
        The DataFrame of each event in a dataset.
    """
    # p=particle
    _p4900101_S1, _p4900021_S1, _p4900111_S1, _p4900113_S1 = [], [], [], []
    _p4900102, _p4900022 = [], []
    for i in range(GP.length):
        dfGP = GP.dataframelize(i)
        dfGP_Status1 = dfGP[dfGP['Status'] == 1]
        pid = dfGP['PID'].to_numpy()
        pid_S1 = dfGP_Status1['PID'].to_numpy()
        # specify PID
        pid_4900101_S1 = pid_S1[abs(pid_S1) == 4900101]
        pid_4900021_S1 = pid_S1[pid_S1 == 4900021]
        pid_4900111_S1 = pid_S1[abs(pid_S1) == 4900111]
        pid_4900113_S1 = pid_S1[abs(pid_S1) == 4900113]
        pid_4900102 = pid[abs(pid) == 4900102]
        pid_4900022 = pid[pid == 4900022]
        # 4900101 & 4900021
        if pid_4900101_S1.shape[0] != 0:
            _p4900101_S1.append(i)
        if pid_4900021_S1.shape[0] != 0:
            _p4900021_S1.append(i)
        # 4900111 & 4900113
        if pid_4900111_S1.shape[0] != 0:
            _p4900111_S1.append(i)
        if pid_4900113_S1.shape[0] != 0:
            _p4900113_S1.append(i)
        # 4900102 & 4900022
        if pid_4900102.shape[0] != 0:
            _p4900102.append(i)
        if pid_4900022.shape[0] != 0:
            _p4900022.append(i)
    # 4900101 & 4900021
    if len(_p4900101_S1) != 0:
        print(
            f"! Event {_p4900101_S1}, there is the existence of particle 4900101 with Status = 1.")
    else:
        print("* There is NO stable 4900101 in final state.")
    if len(_p4900021_S1) != 0:
        print(
            f"! Event {_p4900021_S1}, there is the existence of particle 4900021 with Status = 1.")
    else:
        print("* There is NO stable 4900021 in final state.")
    # 4900111 & 4900113
    if len(_p4900111_S1) != 0:
        print(
            f"! Event {_p4900111_S1}, there is the existence of particle 4900111 with Status = 1.")
    else:
        print("* There is NO stable 4900111 in final state.")
    if len(_p4900113_S1) != 0:
        print(
            f"! Event {_p4900113_S1}, there is the existence of particle 4900113 with Status = 1.")
    else:
        print("* There is NO stable 4900113 in final state.")
    # 4900102 & 4900022
    if len(_p4900102) != 0:
        print(
            f"! Event {_p4900102}, there is the existence of particle 4900102.")
    else:
        print("* There is NO 4900102 in this dataset.")
    if len(_p4900022) != 0:
        print(
            f"! Event {_p4900022}, there is the existence of particle 4900022.")
    else:
        print("* There is NO 4900022 in this dataset.")


# 3-3. Check the existence of neutrinos in a dataset
def neutrino(GP):
    """Check the existence of neutrinos (12, 14, 16) in a dataset.

    Parameters
    ----------
    GP : DataFrame
        The DataFrame of each event in a dataset.

    Returns
    -------
    p12_s1 : ndarray
    p14_s1 : ndarray
    p16_s1 : ndarray
        Which event includes neutrinos (12, 14, 16) in a dataset.
    """
    # p=particle
    _p12_S1, _p14_S1, _p16_S1 = [], [], []
    for i in range(GP.length):
        dfGP = GP.dataframelize(i)
        dfGP_Status1 = dfGP[dfGP['Status'] == 1]
        pid_S1 = dfGP_Status1['PID'].to_numpy()
        # specify PID
        pid_12_S1 = pid_S1[abs(pid_S1) == 12]
        pid_14_S1 = pid_S1[abs(pid_S1) == 14]
        pid_16_S1 = pid_S1[abs(pid_S1) == 16]
        # 12 electron neutrino
        if pid_12_S1.shape[0] != 0:
            _p12_S1.append(i)
        # 14 muon neutrino
        if pid_14_S1.shape[0] != 0:
            _p14_S1.append(i)
        # 16 tau neutrino
        if pid_16_S1.shape[0] != 0:
            _p16_S1.append(i)
    # 12
    if len(_p12_S1) != 0:
        print(
            f"! {len(_p12_S1)} events have the existence of electron neutrino (12) in final state.")
    else:
        print("* There is NO stable electron neutrino (12) in final state.")
    # 14
    if len(_p14_S1) != 0:
        print(
            f"! {len(_p14_S1)} events have the existence of muon neutrino (14) in final state.")
    else:
        print("* There is NO stable muon neutrino (14) in final state.")
    # 16
    if len(_p16_S1) != 0:
        print(
            f"! {len(_p16_S1)} events have the existence of tau neutrino (16) in final state.")
    else:
        print("* There is NO tau neutrino (16) in final state.")

    return np.array(_p12_S1), np.array(_p14_S1), np.array(_p16_S1)


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
def MET_visParticles_v1(sfsp):
    """Calculate missing transverse energy vector,
    ndarray is input and convenient to analyze truth jet and MET,
    and DataFrame is convenient to plot the histogram.

    Parameters
    ----------
    sfsp : list with array elements
        Each array is (pT, eta, phi, mass, pid) 

    Returns
    -------
    MET : ndarray and DataFrame
        ndarray with field names of (MET, phi, METx, METy)
    """
    # * _=list, i=i-th event
    _met_i = []
    for i in range(len(sfsp)):
        pt, eta = sfsp[i][0], sfsp[i][1]
        phi, mass = sfsp[i][2], sfsp[i][3]
        px, py = pt * np.cos(phi), pt * np.sin(phi)
        pxnet_invis, pynet_invis = -np.sum(px), -np.sum(py)
        ptnet_invis = np.sqrt(pxnet_invis**2 + pynet_invis**2)
        phinet_invis = np.arctan2(pynet_invis, pxnet_invis)
        _met_i.append((ptnet_invis, phinet_invis, pxnet_invis, pynet_invis))
    # construct numpy ndarray with dtype
    arr_met = np.array(_met_i, dtype=[('MET', '<f8'), ('phi', '<f8'),
                                      ('METx', '<f8'), ('METy', '<f8')])
    # construct pandas DataFrame
    df_met = pd.DataFrame(arr_met, columns=['MET', 'phi', 'METx', 'METy'])

    print("{} events in MET data.".format(arr_met.shape[0]))
    return arr_met, df_met


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


# B. Version 2
def analyze_truthJet_scheme1_v2(data_presel):
    _N_jet = []
    _observable_jj, _observable_jjj = [], []
    for i in range(len(data_presel)):
        # number of jets
        _N_jet.append(data_presel[i].shape[0])
        # at least dijet
        if data_presel[i].shape[0] >= 2:
            pt, eta = data_presel[i]['pT'], data_presel[i]['eta']
            phi, mass = data_presel[i]['phi'], data_presel[i]['mass']
            dphi = abs(phi[0] - phi[1])
            if dphi > np.pi:
                Dphi = 2*np.pi - dphi
            else:
                Dphi = dphi
            Deta = abs(eta[0] - eta[1])
            _observable_jj.append([pt[0], pt[1], eta[0], eta[1],
                                   M(mass[0], pt[0], eta[0], phi[0],
                                     mass[1], pt[1], eta[1], phi[1]),
                                   MT(mass[0], pt[0], eta[0], phi[0],
                                      mass[1], pt[1], eta[1], phi[1]),
                                   mT(mass[0], pt[0], eta[0], phi[0],
                                      mass[1], pt[1], eta[1], phi[1]),
                                   Dphi, Deta, i])
        # at least trijet
        if data_presel[i].shape[0] >= 3:
            pt, eta = data_presel[i]['pT'], data_presel[i]['eta']
            phi, mass = data_presel[i]['phi'], data_presel[i]['mass']
            _observable_jjj.append([pt[0], pt[1], pt[2], eta[0], eta[1], eta[2],
                                    M_123(mass[0], pt[0], eta[0], phi[0],
                                          mass[1], pt[1], eta[1], phi[1],
                                          mass[2], pt[2], eta[2], phi[2]),
                                    MT_123(mass[0], pt[0], eta[0], phi[0],
                                           mass[1], pt[1], eta[1], phi[1],
                                           mass[2], pt[2], eta[2], phi[2]),
                                    mT_123_def_1(mass[0], pt[0], eta[0], phi[0],
                                                 mass[1], pt[1], eta[1], phi[1],
                                                 mass[2], pt[2], eta[2], phi[2]),
                                    i])
    # transform to np.array() and print the information of array
    arr_N_jet = np.array(_N_jet)
    arr_observable_jj = np.array(_observable_jj)
    arr_observable_jjj = np.array(_observable_jjj)
    print("{} events in the array of number of jets.".format(
        arr_N_jet.shape[0]))
    print("{} selected events and {} observables in dijet.".format(arr_observable_jj.shape[0],
                                                                   arr_observable_jj.shape[1]))
    print("{} selected events and {} observables in trijet.".format(arr_observable_jjj.shape[0],
                                                                    arr_observable_jjj.shape[1]))
    # construct DataFrame from numpy ndarray
    df_N_jet = pd.DataFrame(arr_N_jet, columns=['N_jet'])
    df_jj = pd.DataFrame(arr_observable_jj,
                         columns=['pT_1', 'pT_2', 'eta_1', 'eta_2', 'M_jj',
                                  'MT_jj', 'mT_jj', 'Dphi', 'Deta', 'selected'])
    df_jjj = pd.DataFrame(arr_observable_jjj, columns=['pT_1', 'pT_2', 'pT_3',
                                                       'eta_1', 'eta_2', 'eta_3',
                                                       'M_jjj', 'MT_jjj', 'mT_jjj', 'selected'])
    return df_N_jet, df_jj, df_jjj


# 5-4. Analyze the truth jet and MET (Scheme 1)
def analyze_truthJet_MET_scheme1_v1(data_presel, data_met):
    """This function is my scheme 1 with version 1 to
    analyze truth jet and MET in truth level.

    Parameters
    ----------
    data_presel : list
        Record all preselected events in list data_presel.
        Each event data_presel[i] records PseudoJet information.
    data_met : ndarray with dtype=['MET', 'phi', 'METx', 'METy']
        Record the MET informations of all events.

    Returns
    -------
    observables: DataFrame
        Observables of truth jet and MET.
    """
    _jj_met, _jjj_met = [], []
    for i in range(len(data_presel)):
        # dijet
        if data_presel[i].shape[0] >= 2:
            # data
            pt, eta = data_presel[i]['pT'], data_presel[i]['eta']
            phi, mass = data_presel[i]['phi'], data_presel[i]['mass']
            met, met_phi = data_met['MET'][i], data_met['phi'][i]
            metx, mety = data_met['METx'][i], data_met['METy'][i]
            # transformation and 4-momentum of dijet
            px, py = pt * np.cos(phi), pt * np.sin(phi)
            pz = np.sqrt(mass**2 + pt**2) * np.sinh(eta)
            e = np.sqrt(mass**2 + px**2 + py**2 + pz**2)
            # jj = np.array([(e[0]+e[1], px[0]+px[1], py[0]+py[1], pz[0]+pz[1])],
            #   dtype=[('E', '<f8'), ('px', '<f8'), ('py', '<f8'), ('pz', '<f8')])
            jj = np.array([e[0]+e[1], px[0]+px[1], py[0]+py[1], pz[0]+pz[1]])
            m_jj = np.sqrt(jj[0]**2 - jj[1]**2 - jj[2]**2 - jj[3]**2)
            et_jj = np.sqrt(m_jj**2 + jj[1]**2 + jj[2]**2)
            mt_jj_met = np.sqrt((et_jj + met)**2 -
                                (jj[1] + metx)**2 - (jj[2] + mety)**2)
            phi_jj = np.arctan2(jj[2], jj[1])
            dphi_jj_met = abs(phi_jj - met_phi)
            if dphi_jj_met > np.pi:
                Dphi_jj_met = 2*np.pi - dphi_jj_met
            else:
                Dphi_jj_met = dphi_jj_met
            Dphi_j_met = np.absolute(phi - met_phi)
            Dphi_j_met[Dphi_j_met > np.pi] = 2 * \
                np.pi - Dphi_j_met[Dphi_j_met > np.pi]
            _jj_met.append([mt_jj_met, Dphi_jj_met, Dphi_j_met[0],
                            Dphi_j_met[1], np.min(Dphi_j_met),
                            np.min(Dphi_j_met[:4]), np.min(Dphi_j_met[:6]),
                            np.min(Dphi_j_met[:9]), i])
        # trijet
        if data_presel[i].shape[0] >= 3:
            # data
            pt, eta = data_presel[i]['pT'], data_presel[i]['eta']
            phi, mass = data_presel[i]['phi'], data_presel[i]['mass']
            met, met_phi = data_met['MET'][i], data_met['phi'][i]
            metx, mety = data_met['METx'][i], data_met['METy'][i]
            # transformation and 4-momentum of trijet
            px, py = pt * np.cos(phi), pt * np.sin(phi)
            pz = np.sqrt(mass**2 + pt**2) * np.sinh(eta)
            e = np.sqrt(mass**2 + px**2 + py**2 + pz**2)
            jjj = np.array([e[0]+e[1]+e[2], px[0]+px[1]+px[2],
                            py[0]+py[1]+py[2], pz[0]+pz[1]+pz[2]])
            m_jjj = np.sqrt(jjj[0]**2 - jjj[1]**2 - jjj[2]**2 - jjj[3]**2)
            et_jjj = np.sqrt(m_jjj**2 + jjj[1]**2 + jjj[2]**2)
            mt_jjj_met = np.sqrt((et_jjj + met)**2 -
                                 (jjj[1] + metx)**2 - (jjj[2] + mety)**2)
            phi_jjj = np.arctan2(jjj[2], jjj[1])
            dphi_jjj_met = abs(phi_jjj - met_phi)
            if dphi_jjj_met > np.pi:
                Dphi_jjj_met = 2*np.pi - dphi_jjj_met
            else:
                Dphi_jjj_met = dphi_jjj_met
            Dphi_j_met = np.absolute(phi - met_phi)
            Dphi_j_met[Dphi_j_met > np.pi] = 2 * \
                np.pi - Dphi_j_met[Dphi_j_met > np.pi]
            _jjj_met.append([mt_jjj_met, Dphi_jjj_met, Dphi_j_met[0], Dphi_j_met[1],
                             Dphi_j_met[2], np.min(Dphi_j_met),
                             np.min(Dphi_j_met[:4]), np.min(Dphi_j_met[:6]),
                             np.min(Dphi_j_met[:9]), i])
    # construtct ndarray & DataFrame and print the information of array
    arr_jj_met = np.array(_jj_met)
    arr_jjj_met = np.array(_jjj_met)
    print("{} selected events and {} observables in dijet and MET.".format(arr_jj_met.shape[0],
                                                                           arr_jj_met.shape[1]))
    print("{} selected events and {} observables in trijet and MET.".format(arr_jjj_met.shape[0],
                                                                            arr_jjj_met.shape[1]))
    df_jj_met = pd.DataFrame(arr_jj_met,
                             columns=['MT_jj_MET', 'Dphi_jj_MET', 'Dphi_j1_MET',
                                      'Dphi_j2_MET', 'min_Dphi_j_MET', 'min4_Dphi_j_MET',
                                      'min6_Dphi_j_MET', 'min9_Dphi_j_MET', 'selected'])
    df_jjj_met = pd.DataFrame(arr_jjj_met,
                              columns=['MT_jjj_MET', 'Dphi_jjj_MET',
                                       'Dphi_j1_MET', 'Dphi_j2_MET', 'Dphi_j3_MET',
                                       'min_Dphi_j_MET', 'min4_Dphi_j_MET',
                                       'min6_Dphi_j_MET', 'min9_Dphi_j_MET', 'selected'])
    return df_jj_met, df_jjj_met


# 5-5. The N_jet and states (pT, eta, phi, mass) of 4 leading jet for each event
def jet(data_presel):
    """Collects number of jets (N_jet) and states (pT, eta, phi, mass) of
    4 leading jets for each event.

    Parameters
    ----------
    data_presel : array_like (list)
        Collects the states (pT, eta, phi, mass) of all preselected events
        into list data_presel.

    Returns
    -------
    df_jet : DataFrame
        Collects N_jet and (pT, eta, phi, mass) of 4 leading jets for each event.
    """
    _jet = []
    for i in range(len(data_presel)):
        N_jet = data_presel[i].shape[0]
        pt, eta = data_presel[i]['pT'], data_presel[i]['eta']
        phi, mass = data_presel[i]['phi'], data_presel[i]['mass']
        arr_j = np.array([N_jet])
        # supply -999 element to fill the each state
        if N_jet < 4:
            diff = 4 - N_jet
            arr_n999 = np.full(diff, -999, dtype=np.float64)
            pt = np.concatenate((pt, arr_n999), axis=None)
            eta = np.concatenate((eta, arr_n999), axis=None)
            phi = np.concatenate((phi, arr_n999), axis=None)
            mass = np.concatenate((mass, arr_n999), axis=None)
        arr_j = np.concatenate((arr_j, pt[:4], eta[:4], phi[:4], mass[:4]),
                               axis=None)
        _jet.append(arr_j)
    arr_jet = np.stack(_jet, axis=0)
    print(f"{arr_jet.shape[0]} events and "
          f"{int((arr_jet.shape[1] - 1)/4)} leading jet states (pT, eta, phi, mass).")
    # construct pandas.DataFrame from numpy.ndarray
    df_jet = pd.DataFrame(arr_jet,
                          columns=['N_jet', 'pT_1', 'pT_2', 'pT_3', 'pT_4',
                                   'eta_1', 'eta_2', 'eta_3', 'eta_4',
                                   'phi_1', 'phi_2', 'phi_3', 'phi_4',
                                   'mass_1', 'mass_2', 'mass_3', 'mass_4'])
    return df_jet


# 5-6. The N_jet, Dphi between 4 leading jets and MET,
#      and minimum Dphi between all jets and MET
def jet_MET(data_presel, data_met):
    """Collects number of jets (N_jet), azimuthal angle difference between
    4 leading jet and MET, and minimum azimuthal angle between all jets
    and MET for each event.

    Parameters
    ----------
    data_presel : array_like (list)
        Collects the states (pT, eta, phi, mass) of all preselected events
        into list data_presel.
    data_met : ndarray (np.array(dtype=['MET', 'phi', 'METx', 'METy']))
        Collects the MET informations (MET, phi, METx, METy) of all events.

    Returns
    -------
    df_jet_met : DataFrame
        Collects N_jet, Dphi_j_MET of 4 leading jets, and min(Dphi_j_MET)
        of all jet for each event.
    """
    _jet_met = []
    for i in range(len(data_presel)):
        N_jet = data_presel[i].shape[0]
        pt, eta = data_presel[i]['pT'], data_presel[i]['eta']
        phi, mass = data_presel[i]['phi'], data_presel[i]['mass']
        met, met_phi = data_met['MET'][i], data_met['phi'][i]
        arr_j_met = np.array([N_jet])
        # supply Dphi for 0-jet events
        if N_jet != 0:
            Dphi = np.absolute(phi - met_phi)
        else:
            Dphi = np.full(4, -999, dtype=np.float64)
        # take minimum angle Dphi for each jet
        Dphi[Dphi > np.pi] = 2*np.pi - Dphi[Dphi > np.pi]
        # take minimum Dphi for each event
        min_Dphi = np.array([np.amin(Dphi)])
        # supply -999 into Dphi for N_jet < 4 events
        if N_jet < 4:
            diff = 4 - N_jet
            arr_n999 = np.full(diff, -999, dtype=np.float64)
            Dphi = np.concatenate((Dphi, arr_n999), axis=None)
        arr_j_met = np.concatenate((arr_j_met, Dphi[:4], min_Dphi),
                                   axis=None)
        _jet_met.append(arr_j_met)
    arr_jet_met = np.stack(_jet_met, axis=0)
    print(f"{arr_jet_met.shape[0]} events.")
    print(f"Azimuthal angle difference between 4 leading jets and MET, and "
          f"minimum azimuthal angle between all jets and MET.")
    # construct pandas.DataFrame from numpy.ndarray
    df_jet_met = pd.DataFrame(arr_jet_met,
                              columns=['N_jet', 'Dphi_j1_MET',
                                       'Dphi_j2_MET', 'Dphi_j3_MET',
                                       'Dphi_j4_MET', 'min_Dphi_j_MET'])
    return df_jet_met


################################################################################
#                   6. Analyze the Jets in the Detector Level                  #
################################################################################
# ! coming soon
