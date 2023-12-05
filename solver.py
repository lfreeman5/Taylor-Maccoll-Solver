import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import itertools

def oblique_shock_relations(beta, m_inf):
    gamma = 1.4
    m_inf_normal = m_inf * np.sin(beta)
    m2_normal = np.sqrt((m_inf_normal**2 + 2/(gamma-1)) / (2 * gamma / (gamma -1) * (m_inf_normal**2) - 1))
    deflection = np.arctan(2 / np.tan(beta) * (m_inf**2 * np.sin(beta) ** 2 - 1) / (m_inf ** 2 * (gamma + np.cos(2*beta)) +2) )
    m2 = m2_normal/(np.sin(beta-deflection))
    return [deflection, m2]

def postShock(m2, deflection, beta):
    gamma = 1.4
    vPrime = (2/((gamma - 1) * m2 * m2) + 1) ** (-0.5)
    vPrimeR = np.cos(beta-deflection) * vPrime 
    vPrimeTheta = np.sin(beta-deflection) * vPrime
    return [vPrimeR, vPrimeTheta]

def taylorMaccoll(s, theta):
    #theta is the polar coordinate, marching inwards
    #s is the state vector [v_r, v_theta]
    gamma = 1.4
    v_r, v_theta = s
    return np.array([v_theta, (v_theta ** 2 * v_r - (gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) * (2 * v_r + v_theta / np.tan(theta))) / ((gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) - v_theta ** 2)])

def solveConeAngle(beta, mach, gamma = 1.4):
    [deflection, m2] = oblique_shock_relations(beta, mach)
    [vR, vTheta] = postShock(m2, deflection, beta)
    # print(deflection, '<--deflection  m2-->', m2)
    # print(vR, '<--vR  vTheta-->',vTheta)
    s_0 = np.array([vR, -vTheta])
    theta_values = np.linspace(beta, 0.005, 15000)
    results = odeint(taylorMaccoll, s_0, theta_values)
    v_r_values = results.T[0]
    v_theta_values = results.T[1]
    # Find the index where v_theta changes sign
    change_point_index = np.where(np.diff(np.sign(v_theta_values)))[0][0]
    # Print the corresponding theta value
    theta_change_point = 1/3 * (theta_values[change_point_index] + theta_values[change_point_index+1] + theta_values[change_point_index-1])
    # print(f'For M={mach} and B = {np.degrees(beta)}, the cone angle is: {np.degrees(theta_change_point)}')
    v_r_f = v_r_values[change_point_index]
    m_surf = np.sqrt((2/(gamma-1)) * (1/(1/(v_r_f**2) - 1)))

    return theta_change_point, m_surf

    
def findShockParameters(theta_c, mach, gamma=1.4):
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



thetas = []
betas = np.deg2rad(np.arange(7,80,0.25))
for beta in betas:
    theta = solveConeAngle(beta, 5)[0]
    thetas.append(theta)

# Find the longest increasing section
increasing_sections = [list(group) for _, group in itertools.groupby(enumerate(thetas), lambda x: x[1] > thetas[x[0] - 1] if x[0] > 0 else False)]
longest_section = max(increasing_sections, key=len, default=[])

# Extract the betas and thetas from the longest section
longest_betas, longest_thetas = zip(*longest_section)
# Extract the betas from the longest section
longest_betas_indices = [index for index, _ in longest_section]
longest_betas = [betas[index] for index in longest_betas_indices]

print(longest_thetas)
plt.plot(np.array(betas), np.array(thetas))
plt.plot(longest_betas, longest_thetas)
plt.title('Thetas vs. Betas')
plt.grid(True)
plt.show()

# findShockParameters(0.4, 5)
