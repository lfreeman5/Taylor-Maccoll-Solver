import numpy as np
from matplotlib import pyplot as plt
import json
from poly_gen import *
json_path = "results_1.4.json"
with open(json_path, 'r') as json_file:
    data = json.load(json_file)
data = data["1.4"]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for mach, m_dict in data.items():
    mach = float(mach)
    print(mach)
    thetas = m_dict['thetas']
    betas = m_dict['betas']
    surface_machs = m_dict['surface_machs']
    ax.scatter(np.ones(len(thetas))*mach, thetas, betas, label=f'M {mach}')

beta_coefficients, sm_coefficients = create_coeffs('results_Radau.json')

gen_thetas = np.linspace(0,1,30)
gen_machs = np.linspace(1,30,30)
gen_thetas, gen_machs = np.meshgrid(gen_thetas, gen_machs)
gen_betas = evaluate_function(beta_coefficients, [1.4*np.ones_like(gen_thetas), gen_machs, gen_thetas])
ax.contour3D(gen_machs, gen_thetas, gen_betas, 50, cmap='binary')

ax.set_xlabel('Mach Number')
ax.set_ylabel('Theta')
ax.set_zlabel('Surface Mach Number')
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
    error = (betas - evaluate_function(beta_coefficients, [np.ones(len(thetas))*1.4, np.ones(len(thetas))*mach, np.array(thetas)]))*100
    ax.scatter(np.ones(len(thetas))*mach, thetas, error)

ax.set_xlabel('Mach Number')
ax.set_ylabel('Theta')
ax.set_zlabel('Error (%)')
plt.show()
    