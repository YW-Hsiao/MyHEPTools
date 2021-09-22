"""
Author: You-Wei Hsiao
Institute: Department of Physics, National Tsing Hua University, Hsinchu, Taiwan
Mail: hsiao.phys@gapp.nthu.edu.tw
"""


import numpy as np

# 1. Invariant Mass and Transverse Mass
def M(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1*np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2*np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    return np.sqrt((e1+e2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2)


def MT(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    px1, py1, pz1 = pt1*np.cos(phi1), pt1*np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2*np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    ET1, ET2 = np.sqrt(m1**2 + pt1**2), np.sqrt(m2**2 + pt2**2)
    return np.sqrt((ET1+ET2)**2 - (px1+px2)**2 - (py1+py2)**2)


# 2. Finding Dark Quarks
def Find_xdxdbar(GP):
#     GP = GenParticles
    m_xdxdbar = []
    for i in range(GP.length):
        acc = 0
        for j in range(GP.length_i(i)):
            PID = GP.PID_i(i)[j]
            M1 = GP.M1_i(i)[j]
            M2 = GP.M2_i(i)[j]
            D1 = GP.D1_i(i)[j]
            D2 = GP.D2_i(i)[j]
            Status = GP.Status_i(i)[j]
            
            if PID == 5000001:
                if abs(GP.PID_i(i)[D1]) == abs(GP.PID_i(i)[D2]) == 4900101 and GP.Status_i(i)[D1] == GP.Status_i(i)[D2] == 23:
                    tmp_pt1 = GP.PT_i(i)[D1]
                    tmp_eta1 = GP.Eta_i(i)[D1]
                    tmp_phi1 = GP.Phi_i(i)[D1]
                    tmp_m1 = GP.Mass_i(i)[D1]
                    
                    tmp_pt2 = GP.PT_i(i)[D2]
                    tmp_eta2 = GP.Eta_i(i)[D2]
                    tmp_phi2 = GP.Phi_i(i)[D2]
                    tmp_m2 = GP.Mass_i(i)[D2]
                    break
            elif PID == 4900101 and Status == 23:
                tmp_pt1 = GP.PT_i(i)[j]
                tmp_eta1 = GP.Eta_i(i)[j]
                tmp_phi1 = GP.Phi_i(i)[j]
                tmp_m1 = GP.Mass_i(i)[j]
                acc += 1
                if acc == 2:
                    break
            elif PID == -4900101 and Status == 23:
                tmp_pt2 = GP.PT_i(i)[j]
                tmp_eta2 = GP.Eta_i(i)[j]
                tmp_phi2 = GP.Phi_i(i)[j]
                tmp_m2 = GP.Mass_i(i)[j]
                acc += 1
                if acc == 2:
                    break
                    
        m_xdxdbar.append(M(tmp_pt1, tmp_eta1, tmp_phi1, tmp_m1, tmp_pt2, tmp_eta2, tmp_phi2, tmp_m2))
            
    return np.array(m_xdxdbar)


# 3. Checking rinv
def Check_rinv(GP):
#     GP = GenParticles
    invis_count, vis_count = 0, 0
    Nvmeson = 0
    for i in range(GP.length):
        for j in range(GP.length_i(i)):
            PID = GP.PID_i(i)[j]
            M1 = GP.M1_i(i)[j]
            M2 = GP.M2_i(i)[j]
            D1 = GP.D1_i(i)[j]
            D2 = GP.D2_i(i)[j]
            Status = GP.Status_i(i)[j]
            
            if abs(PID) == 4900111 and abs(GP.PID_i(i)[D1]) != 4900111 and abs(GP.PID_i(i)[D2]) != 4900111:
                if abs(GP.PID_i(i)[D1]) != 3 and abs(GP.PID_i(i)[D2]) != 3:
                    invis_count += 1
                elif abs(GP.PID_i(i)[D1]) == 3 or abs(GP.PID_i(i)[D2]) == 3:
                    vis_count += 1
                    
            elif abs(PID) == 4900113 and abs(GP.PID_i(i)[D1]) != 4900113 and abs(GP.PID_i(i)[D2]) != 4900113:
                if abs(GP.PID_i(i)[D1]) > 4900000 or abs(GP.PID_i(i)[D2]) > 4900000:  # and?
                    invis_count += 1
                elif abs(GP.PID_i(i)[D1]) < 6 or abs(GP.PID_i(i)[D2]) < 6:  # <=?; Ans: not decay to top quark
                    vis_count += 1
                    
            elif abs(PID) == 4900211 and Status == 1:
                Nvmeson += 1
                
            elif abs(PID) == 4900213 and Status == 1:
                Nvmeson += 1
                
    print('There are {} events.'.format(GP.length))
    print('There are {} Dark mesons decay into invisible particle.'.format(invis_count))
    print('There are {} Dark mesons decay into visible particle.'.format(vis_count))
    print('r_inv = {:^4.4f}'.format(invis_count/(invis_count+vis_count)))
    
    print('There are {} stable Dark mesons.'.format(Nvmeson))
    print('Average = {:^4.4f}'.format(Nvmeson/GP.length))
    
    return invis_count/(invis_count+vis_count), Nvmeson/GP.length


def Check_rinv_bRatio(GP, darkhadron=4900111, dm=51):
    invis_count, vis_count = 0, 0
    for i in range(GP.length):
        for j in range(GP.length_i(i)):
            PID = GP.PID_i(i)[j]
            M1 = GP.M1_i(i)[j]
            M2 = GP.M2_i(i)[j]
            D1 = GP.D1_i(i)[j]
            D2 = GP.D2_i(i)[j]
            Status = GP.Status_i(i)[j]
            
            if abs(PID) == darkhadron and abs(GP.PID_i(i)[D1]) != darkhadron and abs(GP.PID_i(i)[D2]) != darkhadron:
                if abs(GP.PID_i(i)[D1]) < 6 and abs(GP.PID_i(i)[D2]) < 6:  # and becomes to or??
                    vis_count += 1
                elif abs(GP.PID_i(i)[D1]) == dm and abs(GP.PID_i(i)[D2]) == dm:
                    invis_count += 1
            
    print('There are {} events.'.format(GP.length))
    print('There are {} Dark Mesons decay into invisible particle.'.format(invis_count))
    print('There are {} Dark Mesons decay into visible particle.'.format(vis_count))
    print('r_inv = {:^6.4f}'.format(invis_count/(invis_count+vis_count)))
    
    return invis_count/(invis_count+vis_count), invis_count, vis_count


# 4. Preselection
def Preselection(Jet):
    twojet_invariantmass = []
    survived_list = []
    for i in range(Jet.length):
        if Jet.length_i(i) < 2:
            continue
        elif Jet.PT_i(i)[0] < 440 or Jet.PT_i(i)[1] < 60:
            continue
        elif np.abs(Jet.Eta_i(i)[0]-Jet.Eta_i(i)[1]) > 1.2:
            continue
            
        twojet_invariantmass.append(M(Jet.PT_i(i)[0], Jet.Eta_i(i)[0], Jet.Phi_i(i)[0], Jet.Mass_i(i)[0],
                                      Jet.PT_i(i)[1], Jet.Eta_i(i)[1], Jet.Phi_i(i)[1], Jet.Mass_i(i)[1]))
        survived_list.append(i)
        
    print('There are {} events.'.format(len(twojet_invariantmass)))
    return np.array(twojet_invariantmass), np.array(survived_list)


# 5. Select Stable Final State Particles
def StableFinalStateParticles(GP):
    event_SFSP = []
    for i in range(GP.length):
        event_SFSP_tmp = []
        for j in range(GP.length_i(i)):
            if GP.Status_i(i)[j] == 1:  # Stable Final State Particles
                event_SFSP_tmp.append([GP.PT_i(i)[j], GP.Eta_i(i)[j], GP.Phi_i(i)[j], GP.Mass_i(i)[j], GP.PID_i(i)[j]])
                
        event_SFSP.append(np.array(event_SFSP_tmp))
        
    return event_SFSP
