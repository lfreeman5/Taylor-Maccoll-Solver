import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import itertools
import pandas as pd
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

def oblique_shock_relations(beta, m_inf, gamma=1.4):
    m_inf_normal = m_inf * np.sin(beta) #Anderson 4.7
    m2_normal = np.sqrt((m_inf_normal**2 + 2/(gamma-1)) / (2 * gamma / (gamma -1) * (m_inf_normal**2) - 1)) #Anderson 4.10
    deflection = np.arctan(2 / np.tan(beta) * (m_inf**2 * np.sin(beta) ** 2 - 1) / (m_inf ** 2 * (gamma + np.cos(2*beta)) +2) ) #Anderson 4.17
    # print(f"beta: {beta}, deflection: {deflection}")
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
    # print(f"For theta: {theta}, v_theta: {v_theta}")
    return np.array([v_theta, (v_theta ** 2 * v_r - (gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) * (2 * v_r + v_theta / np.tan(theta))) / ((gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) - v_theta ** 2)])

def stopCondition(theta, s, _):
    v_r, v_theta = s
    return -v_theta

def solveConeAngle(beta, mach, gamma = 1.4):
    # print(f"solveConeAngle Gamma {GAMMA}")
    [deflection, m2] = oblique_shock_relations(beta, mach, gamma)
    [vR, vTheta] = postShock(m2, deflection, beta, gamma)
    # print(deflection, '<--deflection  m2-->', m2)
    # print(vR, '<--vR  vTheta-->',vTheta)
    s_0 = np.array([vR, -vTheta])
    # Solve_IVP should use an implicit method. I have had luck with Radau and BDF with low tolerances
    sol = solve_ivp(taylorMaccoll, (beta, 0.0005), s_0, events=[stopCondition], method="Radau", atol=1e-10, rtol=1e-10, args=(gamma,))

    # plt.plot(sol.t, sol.y[1])
    # plt.grid(True)
    # plt.title(f"Theta-V_theta plot for B {np.rad2deg(beta)} and M{mach} and g{GAMMA}")
    # plt.show()
    # print(f"Termination condition: {sol.status}, message: {sol.message}")
    # print(sol.t_events[0][0])
    # print(sol.y_events)

    theta_f = sol.t_events[0][0]
    # print(f"SOL Y EVENTS {sol.y_events}")
    v_r_f = sol.y_events[0][0][0]
    # print(f'For M={mach} and B = {np.degrees(beta)}, the cone angle is: {np.degrees(theta_f)}')
    m_surf = np.sqrt((2/(gamma-1)) * (1/(1/(v_r_f**2) - 1)))
    # print(m_surf) Sanity check - are these the same thing? Answer: yes, I derived the M_surf thing correctly
    # m_surf = ((1/(v_r_f**2)-1)*((gamma-1)/2)) ** (-1/2)
    # print(m_surf)
    return theta_f, m_surf

def findShockParameters(theta_c, mach, gamma=1.4): #Uses slightly modified Newton-Raphson method to iterate to a cone angle. Highly dependent on initial conditions.
    beta_max = np.arccos(
        np.sqrt(
            (3 * mach**2 * gamma - np.sqrt((gamma + 1) * (8 * mach**2 * gamma +
            mach**4 * gamma - 8 * mach**2 + mach**4 + 16)) - mach**2 + 4) / gamma
        ) / (2 * mach)
    )

    #beta_0=betamax-1
    #slope = (theta(betamax) - theta(betamax-2))/2
    #next_beta = beta_0-theta(beta_0)/slope
    #if next_beta < 0 next_beta = beta_0/2
    beta = beta_max-0.02
    i=0
    while True:
        stepSize = 0.02 * 0.9**i
        t=solveConeAngle(beta, mach, gamma)[0]-theta_c
        t_more = solveConeAngle(beta+stepSize,mach, gamma)[0]-theta_c
        t_less = solveConeAngle(beta-stepSize,mach, gamma)[0]-theta_c
        slope = (t_more-t_less)/(2*stepSize)
        beta_next = beta - t/slope
        if(beta_next<0):
            beta_next=beta/2
        print("Beta Guess: ", beta)
        print("Slope: ", slope)
        print("Guess at next Beta: ", beta_next)
        input("Press enter to continue")
        beta=beta_next
        i+=1

def generateThetaBeta(mach, gamma=1.4, resolution=0.25):
    #Implementation of Bisection method to rootfind minimum beta
    # print(f"GTB Gamma: {gamma}")
    b_u=np.deg2rad(90)
    b_l=np.deg2rad(0)
    for _ in range(25):
        try:
            b_h=(b_u+b_l)/2
            # print(f"Trying Beta: {np.rad2deg(b_h)}")
            solveConeAngle(b_h, mach, gamma=gamma)
            #If code gets here, the halfway point is still valid. So the new upper is the halfway
            b_u=b_h
        except (IndexError, ValueError) as e:
            #If it fails, the halfway point is below beta_min, so it's the new lower bound
            b_l=b_h
    beta_min = b_u
    # print("Minimum Beta: ", np.rad2deg(beta_min))
    t,_=solveConeAngle(beta_min, mach)
    # print(f"Minimum theta: {np.rad2deg(t)}")

    betas = np.deg2rad(np.geomspace(np.rad2deg(beta_min), 90, num=140))
    solvedBetas = []
    thetas = []
    msurfs = []
    for beta in betas:
        # print(beta)
        try:
            (theta, msurf) = solveConeAngle(beta, mach, gamma=gamma)
            thetas.append(theta)
            msurfs.append(msurf)
            solvedBetas.append(beta)
        except (IndexError, ValueError) as e:
            pass
            # print(f"For beta: {beta}, mach: {mach}, gamma: {gamma}, the ODE did not solve")
    betas = solvedBetas

    
    # Find the longest increasing run indices
    (start, end) = longest_increasing_subsequence_indices(thetas)

    # Extract the data for the longest run
    longest_thetas = thetas[start:end]
    longest_betas = betas[start:end]
    longest_machs = msurfs[start:end]

    # plt.plot(longest_thetas, longest_betas, marker='o', color='g')
    # plt.plot(thetas, betas, marker='x',linestyle='-')
    # plt.show()
    return (longest_betas, longest_thetas, longest_machs)


# generateThetaBeta(5,gamma=1.2)



# results={}
# fig, axs = plt.subplots(1, 2, figsize=(8, 10))

# for m in [2.5,4,6,8,15]:
#     results[m]={}
#     (results[m]['betas'], results[m]['thetas'], results[m]['surface_machs']) = generateThetaBeta(m, resolution=0.05, gamma=1.1)
#     axs[0].plot(results[m]['thetas'], results[m]['betas'], label=f"Mach {m}")
#     axs[1].plot(results[m]['thetas'], results[m]['surface_machs'], label=f"Mach {m}")

# axs[0].set_title('Betas vs. Thetas')
# axs[0].legend()
# axs[1].set_title('Surface Machs vs. Thetas')
# axs[1].legend()
# plt.show()


# results = {}
# for g in [1.1,1.2,1.3,1.4]:
#     results[g]={}
#     for m in [2,4,6,8,10,12,15,20,25,30]:
#         print(f"Running at M{m}, gamma={g}")
#         results[g][m]={}
#         (results[g][m]['betas'], results[g][m]['thetas'], results[g][m]['surface_machs']) = generateThetaBeta(m, gamma=g)

# json_file_path = "results_Radau.json"
# with open(json_file_path, 'w') as json_file:
#     json.dump(results, json_file, indent=2)

# print(f"Results exported to {json_file_path}")
 
results = {}
for g in [1.04, 1.08, 1.12, 1.16, 1.2 , 1.24, 1.28, 1.32, 1.36, 1.4 ]:
    results[g]={}
    for m in [10]:
        print(f"Running at M{m}, gamma={g}")
        results[g][m]={}
        (results[g][m]['betas'], results[g][m]['thetas'], results[g][m]['surface_machs']) = generateThetaBeta(m, gamma=g)

json_file_path = "results_gamma_sweep.json"
with open(json_file_path, 'w') as json_file:
    json.dump(results, json_file, indent=2)

print(f"Results exported to {json_file_path}")
 
