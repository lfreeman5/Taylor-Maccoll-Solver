import numpy as np
from matplotlib import pyplot as plt
import json
from poly_fits import *

json_path = "results_gamma_sweep.json"
with open(json_path, 'r') as json_file:
    data = json.load(json_file)

independent_variables = []
dependent_surface_machs = []
dependent_betas = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for gamma, g_dict in data.items():
    gamma = float(gamma)
    if(gamma in [1.04, 1.08, 1.12, 1.16, 1.2 , 1.24]):
        continue
    m_dict=g_dict["10"]
    thetas = m_dict['thetas']
    betas = m_dict['betas']
    surface_machs = m_dict['surface_machs']
    ax.scatter(np.ones(len(thetas))*gamma, thetas, betas, label=f'M {gamma}')
    
    for i in range(len(thetas)):
        independent_variables.append([gamma,thetas[i]])
        dependent_surface_machs.append(surface_machs[i])
        dependent_betas.append(betas[i])

orders = [1,6] #Order: gamma, theta
coefficients, r2_value = multivariablePolynomialFit(orders, independent_variables, dependent_betas)


gen_thetas = np.linspace(0,1,30)
gen_gammas = np.linspace(1.04,1.4,30)
gen_thetas, gen_gammas = np.meshgrid(gen_thetas, gen_gammas)
gen_betas = multivariablePolynomialFunction(coefficients, orders, [gen_gammas, gen_thetas])
ax.contour3D(gen_gammas, gen_thetas, gen_betas, 50, cmap='binary')

ax.set_xlabel('Gamma')
ax.set_ylabel('Theta')
ax.set_zlabel('Beta')
plt.legend()
plt.show()
print(coefficients)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for gamma, g_dict in data.items():
    gamma = float(gamma)
    if(gamma in [1.04, 1.08, 1.12, 1.16, 1.2 , 1.24]):
        continue
    m_dict=g_dict["10"]
    thetas = m_dict['thetas']
    betas = m_dict['betas']
    surface_machs = m_dict['surface_machs']
    error = (betas - multivariablePolynomialFunction(coefficients, orders, [np.ones(len(thetas))*gamma, np.array(thetas)])) * 100 
    ax.scatter(np.ones(len(thetas))*gamma, thetas, error, label=f'M {gamma}')



ax.set_xlabel('Mach Number')
ax.set_ylabel('Theta')
ax.set_zlabel('Error (%)')
plt.show()