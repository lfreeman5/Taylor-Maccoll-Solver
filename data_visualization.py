import numpy as np
from matplotlib import pyplot as plt
import json
from poly_fits import *

json_path = "results_1.4.json"
with open(json_path, 'r') as json_file:
    data = json.load(json_file)
data = data["1.4"]

independent_variables = []
dependent_surface_machs = []
dependent_betas = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for mach, m_dict in data.items():
    mach = float(mach)
    print(mach)
    thetas = m_dict['thetas']
    betas = m_dict['betas']
    surface_machs = m_dict['surface_machs']
    ax.scatter(np.ones(len(thetas))*mach, thetas, surface_machs, label=f'M {mach}')
    
    for i in range(len(thetas)):
        independent_variables.append([mach,thetas[i]])
        dependent_surface_machs.append(surface_machs[i])
        dependent_betas.append(betas[i])

orders = [6,9] #Order: mach, theta
coefficients, r2_value = multivariablePolynomialFit(orders, independent_variables, dependent_surface_machs)


gen_thetas = np.linspace(0,1,30)
gen_machs = np.linspace(1,30,30)
gen_thetas, gen_machs = np.meshgrid(gen_thetas, gen_machs)
gen_betas = multivariablePolynomialFunction(coefficients, orders, [gen_machs, gen_thetas])
ax.contour3D(gen_machs, gen_thetas, gen_betas, 50, cmap='binary')

ax.set_xlabel('Mach Number')
ax.set_ylabel('Theta')
ax.set_zlabel('Beta')
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for mach, m_dict in data.items():
    mach = float(mach)
    print(mach)
    thetas = m_dict['thetas']
    betas = m_dict['betas']
    surface_machs = m_dict['surface_machs']
    error = (surface_machs - multivariablePolynomialFunction(coefficients, orders, [np.ones(len(thetas))*mach, np.array(thetas)]))*100
    ax.scatter(np.ones(len(thetas))*mach, thetas, error)

ax.set_xlabel('Mach Number')
ax.set_ylabel('Theta')
ax.set_zlabel('Error (%)')
plt.show()
    