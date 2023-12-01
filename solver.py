import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

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
    gamma = 1.4
    v_r, v_theta = s
    # print("V_Theta: ", v_theta)
    # print("V_R: ", v_r)
    #theta is the polar coordinate, marching inwards
    #s is the state vector [v_r, v_theta]
    return np.array([v_theta, (v_theta ** 2 * v_r - (gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) * (2 * v_r + v_theta / np.tan(theta))) / ((gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) - v_theta ** 2)])

def solveConeAngle(beta, mach, gamma = 1.4):
    [deflection, m2] = oblique_shock_relations(beta, mach)
    [vR, vTheta] = postShock(m2, deflection, beta)
    # print(deflection, '<--deflection  m2-->', m2)
    # print(vR, '<--vR  vTheta-->',vTheta)
    s_0 = np.array([vR, -vTheta])
    theta_values = np.linspace(beta, 0.005, 200)
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

def findShockParameters(theta, mach, gamma=1.4):
    #Paramters: theta is the cone angle, mach is freestream mach number
    #Pass theta in radians, mach dimensionless
    angles = np.linspace(np.pi/2, 0, 1000) #Deflection from 90 degrees to zero 
    coneAngle = -1
    goingDown = False
    prevTheta = -1
    coneAngle, surfaceMach = 0,0
    beta=0
    for angle in angles: #Goes from 90 to zero
        #Run solveConeAngle with beta, if it fails then it didnt' converge to a solution. The first time it converges, great.
        try:
            # print('Trying angle ', np.degrees(angle), 'at M=', mach)
            coneAngle, surfaceMach = solveConeAngle(angle, mach)
            if not goingDown and coneAngle < prevTheta:
                # print("Descending")
                goingDown = True 
            
            if coneAngle < theta and goingDown: #The cone angles will start large because we start at a deflection of 90 degrees
                beta = angle
                break
        except IndexError as e:
            print(f"Error: {e}")
            continue
        prevTheta = coneAngle
    if coneAngle == -1:
        return "NO SOLUTION FOUND"
    else:
        #Using constitutive relationships find surface pressure
        pressureRatio = (1 + surfaceMach**2 * (gamma - 1)/2) ** (gamma / (gamma - 1))
        pressureCoefficient = 2/(gamma*mach**2) * (pressureRatio-1)
        print("Surface Mach Number: ", surfaceMach)
        print("Shock Angle: ", np.degrees(beta))
        print("Surface Pressure Ratio for cone: ", pressureRatio)
        print("Surface Pressure Coefficient for cone: ", pressureCoefficient)
        return
    

print(solveConeAngle(np.deg2rad(22.149), 4))
# print(solveConeAngle(1.4, 21))
# print(solveConeAngle(1.3, 21))
# print(solveConeAngle(1.2, 21))
# print(solveConeAngle(1.1, 21))
# findShockParameters(np.radians(20), 5)


























# # Plot the results
# plt.plot(theta_values, v_r_values, label='v_r')
# plt.plot(theta_values, v_theta_values, label='v_theta')
# plt.xlabel('Theta')
# plt.ylabel('Values')
# plt.legend()
# plt.show()

###Plotting oblique shock relations
# mach_numbers = [2, 3, 4, 5, 10]
# beta_range = np.radians(np.arange(0, 91, 0.1))  # Convert beta to radians

# # Plot the results for each Mach number
# for mach_number in mach_numbers:
#     deflection_results = [np.degrees(oblique_shock_relations(beta, mach_number)[0]) for beta in beta_range]

#     # Filter data for deflection > 0
#     # beta_filtered = [np.degrees(beta) for i, beta in enumerate(beta_range) if deflection_results[i] > 0]
#     # deflection_filtered = [deflection for deflection in deflection_results if deflection > 0]

#     # Plot the filtered results with flipped axes
#     plt.plot(deflection_results, beta_range, label=f'Mach {mach_number}')

# plt.title('Oblique Shock Deflection Angle vs. Beta')
# plt.xlabel('Deflection Angle (degrees)')
# plt.ylabel('Beta (degrees)')
# plt.legend()
# plt.grid(True)
# plt.show()
