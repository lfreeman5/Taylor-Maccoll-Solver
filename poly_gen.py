from poly_fits import *
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt

json_path = "results_Radau.json"
with open(json_path, 'r') as json_file:
    data = json.load(json_file)

#The two dependent variables we care about are M_surf and beta
#The three independent variables are M_inf, gamma, and theta_c
#Assume dict is of the form dict[gamma][mach]['requested_result']
independent_variables = []
dependent_surface_machs = []
dependent_betas = []


for g, g_dict in data.items():
    fig, axs = plt.subplots(1, 2, figsize=(8, 10))
    g=float(g)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', '*', 'X']
    for i, (m, m_dict) in enumerate(g_dict.items()):
        axs[0].plot(np.rad2deg(m_dict['thetas']), np.rad2deg(m_dict['betas']), label=f"Mach {m}", marker=markers[i], markevery=0.1)
        axs[1].plot(np.rad2deg(m_dict['thetas']), m_dict['surface_machs'], label=f"Mach {m}", marker=markers[i], markevery=0.1)
        m=float(m)
        thetas = m_dict['thetas']
        betas = m_dict['betas']
        surface_machs = m_dict['surface_machs']
        for i in range(len(thetas)):
            independent_variables.append([g,m,thetas[i]])
            dependent_surface_machs.append(surface_machs[i])
            dependent_betas.append(betas[i])
    axs[0].set_title(f'Betas vs. Thetas at Gamma = {g}')
    axs[0].set_xlabel('Theta (deg)')
    axs[0].set_ylabel('Beta (deg)')
    axs[0].legend()
    axs[1].set_title(f'Surface Machs vs. Thetas at Gamma = {g}')
    axs[1].set_xlabel('Theta (deg)')
    axs[1].set_ylabel('Mach Number on Cone Surface')
    axs[1].legend()
    plt.show()

exit()
orders = [3,4,7] #Order: gamma, mach, theta
coefficients, r2_value = multivariablePolynomialFit(orders, independent_variables, dependent_betas)
print(coefficients)
test_ind = [1.2,10,0.2]
predictedBeta = multivariablePolynomialFunction(coefficients, orders, test_ind)
print(predictedBeta)