import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import json 


def longest_increasing_subsequence_indices(arr):
    def increasing_subsequences():
        start, current_length = 0, 1
        for i in range(1, len(arr)):
            if arr[i] > arr[i - 1]:
                current_length += 1
            else:
                yield start, start + current_length - 1
                start = i
                current_length = 1
        yield start, start + current_length - 1

    all_subsequences = list(increasing_subsequences())
    longest_subsequence_indices = max(all_subsequences, key=lambda x: x[1] - x[0], default=(0, 0))
    return longest_subsequence_indices

def tangent_cone_pressure_coefficient(m_inf, m_surf, beta, gamma):
    # Equations from http://mae-nas.eng.usu.edu/MAE_6530_Web/New_Course/Section7/section2.2.1.pdf
    # CP from pressure ratio is 9.10 in Anderson Compressible
    mn_inf = m_inf * np.sin(beta)
    mn_post = np.sqrt((1 + (gamma - 1) / 2 * mn_inf ** 2) / (gamma * mn_inf ** 2 - (gamma - 1) / 2))
    deflection = np.arctan(2 * (m_inf ** 2 * np.sin(beta) ** 2 - 1) / 
                            (np.tan(beta) * (2 + m_inf ** 2 * (gamma + np.cos(2 * beta)))))
    m_post = mn_post / np.sin(beta - deflection)
    pressure_ratio = ((1 + (gamma - 1) / 2 * m_post ** 2) / (1 + (gamma - 1) / 2 * m_surf ** 2)) ** (
            gamma / (gamma - 1)) * (
                             1 + 2 * gamma / (gamma + 1) * (m_inf ** 2 * np.sin(beta) ** 2 - 1))
    cp = 2 / (gamma * m_inf ** 2) * (pressure_ratio - 1)
    return cp

def oblique_shock_relations(beta, m_inf, gamma=1.4):
    m_inf_normal = m_inf * np.sin(beta) #Anderson 4.7
    m2_normal = np.sqrt((m_inf_normal**2 + 2/(gamma-1)) / (2 * gamma / (gamma -1) * (m_inf_normal**2) - 1)) #Anderson 4.10
    deflection = np.arctan(2 / np.tan(beta) * (m_inf**2 * np.sin(beta) ** 2 - 1) / (m_inf ** 2 * (gamma + np.cos(2*beta)) +2) ) #Anderson 4.17
    m2 = m2_normal/(np.sin(beta-deflection)) #Anderson 4.12
    return [deflection, m2]

def postShock(m2, deflection, beta, gamma=1.4):
    vPrime = (2/((gamma - 1) * m2 * m2) + 1) ** (-0.5)
    vPrimeR = np.cos(beta-deflection) * vPrime 
    vPrimeTheta = np.sin(beta-deflection) * vPrime
    return [vPrimeR, vPrimeTheta]

def taylorMaccoll(theta, s, gamma):
    #theta is the polar coordinate, marching inwards
    #s is the state vector [v_r, v_theta]
    v_r, v_theta = s
    return np.array([v_theta, (v_theta ** 2 * v_r - (gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) * (2 * v_r + v_theta / np.tan(theta))) / ((gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) - v_theta ** 2)])

def stopCondition(theta, s, _):
    _, v_theta = s
    return -v_theta

def solveConeAngle(beta, mach, gamma = 1.4):
    [deflection, m2] = oblique_shock_relations(beta, mach, gamma)
    [vR, vTheta] = postShock(m2, deflection, beta, gamma)
    s_0 = np.array([vR, -vTheta])
    # Solve_IVP should use an implicit method. I have had luck with Radau and BDF with low tolerances. Very low tolerances should be used for generating datasets
    sol = solve_ivp(taylorMaccoll, (beta, 0.0005), s_0, events=[stopCondition], method="BDF", atol=1e-10, rtol=1e-10, args=(gamma,))
    theta_f = sol.t_events[0][0]
    v_r_f = sol.y_events[0][0][0]
    m_surf = np.sqrt((2/(gamma-1)) * (1/(1/(v_r_f**2) - 1)))
    return theta_f, m_surf

def findShockParameters(theta_c, mach, gamma=1.4): 
    #Uses slightly modified Newton-Raphson method to iterate to a cone angle. Highly dependent on initial conditions.
    beta_max = np.arccos(
        np.sqrt(
            (3 * mach**2 * gamma - np.sqrt((gamma + 1) * (8 * mach**2 * gamma +
            mach**4 * gamma - 8 * mach**2 + mach**4 + 16)) - mach**2 + 4) / gamma
        ) / (2 * mach)
    )
    b_u=np.deg2rad(90)
    b_l=np.deg2rad(0)
    for _ in range(30):
        try:
            b_h=(b_u+b_l)/2
            solveConeAngle(b_h, mach, gamma=gamma)
            b_u=b_h
        except (IndexError, ValueError) as e:
            b_l=b_h
    beta_min = b_u
    #beta_0=betamax-1
    #slope = (theta(betamax) - theta(betamax-2))/2
    #next_beta = beta_0-theta(beta_0)/slope
    #if next_beta < 0 next_beta = beta_0/2
    beta = beta_max*0.99
    print(f"Min beta: {beta_min} max beta: {beta_max}")
    i=0
    m_surf = 0
    while i<15:
        stepSize = 0.02 * 0.9**i
        t, m_surf = solveConeAngle(beta, mach, gamma)
        t-=theta_c
        t_more = solveConeAngle(beta+stepSize,mach, gamma)[0]-theta_c
        t_less = solveConeAngle(beta-stepSize,mach, gamma)[0]-theta_c
        slope = (t_more-t_less)/(2*stepSize)
        beta_next = beta - t/slope
        if(beta_next<beta_min):
            beta_next = beta_min+0.05
        print("Beta Guess: ", beta)
        print("Slope: ", slope)
        print("Guess at next Beta: ", beta_next)
        # input("Press enter to continue")
        beta=beta_next
        i+=1
    return beta, m_surf

def generateThetaBeta(mach, gamma=1.4, resolution=0.25):
    #Implementation of Bisection method to rootfind minimum beta
    b_u=np.deg2rad(90)
    b_l=np.deg2rad(0)
    for _ in range(25):
        try:
            b_h=(b_u+b_l)/2
            solveConeAngle(b_h, mach, gamma=gamma)
            b_u=b_h
        except (IndexError, ValueError) as e:
            b_l=b_h
    beta_min = b_u
    t,_=solveConeAngle(beta_min, mach)
    print(f"beta_min: {np.rad2deg(beta_min)}, theta_min: {np.rad2deg(t)}")
    min, max = np.emath.logn(12, np.rad2deg(beta_min)), np.emath.logn(12, 90)
    betas = np.deg2rad(np.logspace(min, max, base=12,num=100))
    solvedBetas = []
    thetas = []
    msurfs = []
    for beta in betas:
        try:
            (theta, msurf) = solveConeAngle(beta, mach, gamma=gamma)
            thetas.append(theta)
            msurfs.append(msurf)
            solvedBetas.append(beta)
        except (IndexError, ValueError) as e:
            pass
    betas = solvedBetas

    (start, end) = longest_increasing_subsequence_indices(thetas)
    longest_thetas = thetas[start:end]
    longest_betas = betas[start:end]
    longest_machs = msurfs[start:end]
    return (longest_betas, longest_thetas, longest_machs)

if __name__ == "main":
    results = {}
    for g in [1.1,1.2,1.3,1.4]:
        results[g]={}
        for m in [2,2.5,3,4,6,8,12,25]:
            print(f"Running at M{m}, gamma={g}")
            results[g][m]={}
            (results[g][m]['betas'], results[g][m]['thetas'], results[g][m]['surface_machs']) = generateThetaBeta(m, gamma=g)
            results[g][m]['cps'] = tangent_cone_pressure_coefficient(
                np.array([m] * len(results[g][m]['thetas'])),
                np.array(results[g][m]['surface_machs']),
                np.array(results[g][m]['betas']),
                np.array([g] * len(results[g][m]['thetas']))
            ).tolist()
    json_file_path = "Results_Polys.json"
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=2)

    print(f"Results exported to {json_file_path}")
 