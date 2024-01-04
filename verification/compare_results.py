import numpy as np
import matplotlib.pyplot as plt
from gdtk import ideal_gas_flow, zero_solvers
import sys, os, csv
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

"""

    beta = np.radians(35)
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


"""

R = 287 #specific gas constant, J/kgK
gamma = 1.405 #specific heat ratio
mach = 5

#GDTK works in units
T = 230 #degrees kelvin K
c = np.sqrt(R*gamma*T) #speed of sound in m/s
v_inf = c*mach #m/s
p_inf = 40000 #Pa - in the end we need a pressure ratio so this will not matter

if __name__ == "__main__":
    csv_name = 'verification\\1135_digitized.csv'
    plot_dir = 'verification\\plots'
    with open(csv_name, 'r') as f:
        reader = csv.reader(f)
        header_1 = next(reader)
        header_2 = next(reader)
        data_rows = [row for row in reader]
    machs = [float(val) for val in header_1 if val != '']
    column1 = [float(row[0]) for row in data_rows if row[0]!='']
    column2 = [float(row[1]) for row in data_rows if row[0]!='']
    for i, mach in enumerate(machs):
        theta_naca = [np.radians(float(row[2*i])) for row in data_rows if row[2*i]!='']
        cp_naca = [float(row[2*i+1]) for row in data_rows if row[2*i]!='']

        theta_range = np.linspace(0, max(theta_naca)*0.995, 30)
        cp_range_gdtk = []
        cp_range_me = []
        for theta in theta_range:
            print(f"Me solving {theta}")
            beta, msurf = solver.findShockParameters(theta, mach, gamma)
            cp_range_me.append(tangent_cone_pressure_ratio(mach, msurf, beta, gamma)[1])
            # print(f"GDTK Solving: {theta}")
            # beta_gdtk = ideal_gas_flow.beta_cone2(mach, theta, R=R, g=gamma, dtheta=-1e-6)
            # _, _, p_gdtk, _ = ideal_gas_flow.theta_cone(v_inf, p_inf, T, beta_gdtk, R, gamma, dtheta=-1e-6)
            # print(p_gdtk)
            # cp = 2 / (gamma * mach ** 2) * (p_gdtk/p_inf  - 1)
            # cp_range_gdtk.append(cp)

        # Scatter NACA data
        plt.scatter(np.degrees(theta_naca), cp_naca, label='NACA Data', marker='o', color='blue')

        # Plot GDTK data as a line
        # plt.plot(np.degrees(theta_range), cp_range_gdtk, label='GDTK Data', color='red', linestyle='-')

        # Plot Me data as a line
        plt.plot(np.degrees(theta_range), cp_range_me, label='Me Data', color='green', linestyle='--')

        # Set labels and title
        plt.xlabel('Theta (degrees)')
        plt.ylabel('C_p')
        plt.title('NACA 1135 Data vs Taylor-Maccoll Data')
        plt.legend()
        plt.savefig(f'verification/plots/M{mach}_comparison.pdf')
        plt.show()


