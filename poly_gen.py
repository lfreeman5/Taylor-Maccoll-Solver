from poly_fits import *
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt

def create_coeffs(json_name, key, orders=[3,4,6]):
    with open(json_name, 'r') as json_file:
        data = json.load(json_file)
    
    independent_variables = []
    dependent_variables = []

    for g, g_dict in data.items():
        g=float(g)
        for j, (m, m_dict) in enumerate(g_dict.items()):
            m=float(m)
            thetas = m_dict['thetas']
            dependent_vars = m_dict[key]
            for i in range(len(thetas)):
                independent_variables.append([g,m,thetas[i]])
                dependent_variables.append(dependent_vars[i])


    coefficients, r2_value = multivariablePolynomialFit(orders, independent_variables, dependent_variables)
    return coefficients

def evaluate_function(coeffs, input_vector, orders=[3,4,6]): #input_vector takes the form [gamma, mach, theta]
    return multivariablePolynomialFunction(coeffs, orders, input_vector)

if __name__ == '__main__':
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