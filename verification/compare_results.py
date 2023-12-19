import numpy as np
from gdtk import ideal_gas_flow, zero_solvers
import sys, os
current_script_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, os.pardir))
sys.path.append(parent_directory)
import solver

def tangent_cone_pressure_ratio(m_inf, m_surf, beta, gamma):
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
    return pressure_ratio, cp

R = 287 #specific gas constant, J/kgK
gamma = 1.4 #specific heat ratio
T = 230 #degrees kelvin K
c = np.sqrt(R*gamma*T) #speed of sound in m/s

beta = np.radians(35)
mach = 5
v_inf = c*mach #m/s
p_inf = 40000 #Pa

#(V1, p1, T1, beta, R=287.1, g=1.4, dtheta=-1.0e-5):
theta_c_gdtk, V_r_gdtk, p_gdtk, T_gdtk = ideal_gas_flow.theta_cone(v_inf, p_inf, T, beta, R, gamma, dtheta = -1e-6)

theta_c_me, m_surf = solver.solveConeAngle(beta, mach, gamma=gamma)

print(f"theta_c_gdtk: {np.degrees(theta_c_gdtk)} vs theta_c_me: {np.degrees(theta_c_me)}")
pr, cp = tangent_cone_pressure_ratio(mach, m_surf, beta, gamma)
print(f"p_gdtk: {p_gdtk} vs p_me {p_inf*pr}")

theta_c = np.radians(35)
beta_me, m_surf_me = solver.findShockParameters(theta_c, mach, gamma)
beta_gdtk = ideal_gas_flow.beta_cone2(mach, theta_c, R=R, g=gamma, dtheta=-1e-6)
print(f"beta_gdtk: {np.degrees(beta_gdtk)} vs beta_me: {np.degrees(beta_me)}")


