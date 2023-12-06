import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import itertools

def oblique_shock_relations(beta, m_inf):
    gamma = 1.4
    m_inf_normal = m_inf * np.sin(beta) #Anderson 4.7
    m2_normal = np.sqrt((m_inf_normal**2 + 2/(gamma-1)) / (2 * gamma / (gamma -1) * (m_inf_normal**2) - 1)) #Anderson 4.10
    deflection = np.arctan(2 / np.tan(beta) * (m_inf**2 * np.sin(beta) ** 2 - 1) / (m_inf ** 2 * (gamma + np.cos(2*beta)) +2) ) #Anderson 4.17
    print(f"beta: {beta}, deflection: {deflection}")
    m2 = m2_normal/(np.sin(beta-deflection)) #Anderson 4.12
    return [deflection, m2]

def postShock(m2, deflection, beta):
    gamma = 1.4
    vPrime = (2/((gamma - 1) * m2 * m2) + 1) ** (-0.5)
    vPrimeR = np.cos(beta-deflection) * vPrime 
    vPrimeTheta = np.sin(beta-deflection) * vPrime
    return [vPrimeR, vPrimeTheta]

THETA = -1
def taylorMaccoll(theta, s, gamma=1.4):
    #theta is the polar coordinate, marching inwards
    #s is the state vector [v_r, v_theta]
    global THETA
    v_r, v_theta = s
    THETA=theta
    # print(f"For theta: {theta}, v_theta: {v_theta}")
    return np.array([v_theta, (v_theta ** 2 * v_r - (gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) * (2 * v_r + v_theta / np.tan(theta))) / ((gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) - v_theta ** 2)])

def stopCondition(theta, s):
    v_r, v_theta = s
    return -v_theta
# stopCondition.terminal=True

def solveConeAngle(beta, mach, gamma = 1.4):
    [deflection, m2] = oblique_shock_relations(beta, mach)
    [vR, vTheta] = postShock(m2, deflection, beta)
    # print(deflection, '<--deflection  m2-->', m2)
    # print(vR, '<--vR  vTheta-->',vTheta)
    s_0 = np.array([vR, -vTheta])
    theta_values = np.linspace(beta, 0.005, 15000)
    # Define the solve_ivp options
    sol = solve_ivp(taylorMaccoll, (beta, 0.005), s_0, events=[stopCondition], method="Radau")

    # plt.plot(sol.t, sol.y[1])
    # plt.show()
    # print(f"Termination condition: {sol.status}, message: {sol.message}")
    # print(sol.t_events[0][0])
    # print(sol.y_events)

    theta_f = sol.t_events[0][0]
    v_r_f = sol.y_events[0][0]
    # print(f'For M={mach} and B = {np.degrees(beta)}, the cone angle is: {np.degrees(theta_f)}')
    m_surf = np.sqrt((2/(gamma-1)) * (1/(1/(v_r_f**2) - 1)))
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
    global THETA
    beta_min = np.arcsin(np.sqrt((gamma-1)/(2*gamma) * 1/(mach**2)))
    betas = np.deg2rad(np.arange(np.rad2deg(beta_min+0.0001),90,resolution))
    solvedBetas = []
    thetas = []
    for beta in betas:
        print(beta)
        try:
            theta = solveConeAngle(beta, mach)[0]
            thetas.append(theta)
            solvedBetas.append(beta)
        except IndexError as e:
            print(f"For beta: {beta}, mach: {mach}, gamma: {gamma}, the ODE did not solve")
            thetas.append(THETA)
            THETA = -1
            solvedBetas.append(beta)

    betas = solvedBetas
    increasing_sections = [list(group) for _, group in itertools.groupby(enumerate(thetas), lambda x: x[1] > thetas[x[0] - 1] if x[0] > 0 else False)]
    longest_section = max(increasing_sections, key=len, default=[])
    longest_betas, longest_thetas = zip(*longest_section)
    longest_betas_indices = [index for index, _ in longest_section]
    longest_betas = [betas[index] for index in longest_betas_indices]
    return (longest_betas, longest_thetas)


try:
    solveConeAngle(np.deg2rad(0.1), 5)
except IndexError as e:
    print(f"The ODE did not solve")
    print(f"Try theta: {THETA}")

# results={}
# for m in [2, 5]:
#     results[m]={}
#     (results[m]['betas'], results[m]['thetas']) = generateThetaBeta(m, resolution=1)
#     plt.plot(results[m]['thetas'], results[m]['betas'], label=f"Mach {m}")

plt.title("Weak Shock Theta-Beta-M Plot for Conical flow")
plt.grid(True)
plt.legend()
plt.show()



# findShockParameters(0.4, 5)
