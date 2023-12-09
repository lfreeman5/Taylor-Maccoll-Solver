from poly_fits import *
import numpy as np
import pandas as pd
import json

json_path = "results.json"
with open(json_path, 'r') as json_file:
    data = json.load(json_file)

#The two dependent variables we care about are M_surf and beta
#The three independent variables are M_inf, gamma, and theta_c
#Assume dict is of the form dict[gamma][mach]['requested_result']
independent_variables = []
dependent_surface_machs = []
dependent_betas = []
for g, g_dict in data.items():
    g=float(g)
    for m, m_dict in g_dict.items():
        m=float(m)
        thetas = m_dict['thetas']
        betas = m_dict['betas']
        surface_machs = m_dict['surface_machs']
        for i in range(len(thetas)):
            independent_variables.append([g,m,thetas[i]])
            dependent_surface_machs.append(surface_machs[i])
            dependent_betas.append(betas[i])

orders = [2,3,6] #Order: gamma, mach, theta
coefficients, r2_value = multivariablePolynomialFit(orders, independent_variables, dependent_betas)
print(coefficients)
test_ind = [1.2,10,0.2]
predictedBeta = multivariablePolynomialFunction(coefficients, orders, test_ind)
print(predictedBeta)