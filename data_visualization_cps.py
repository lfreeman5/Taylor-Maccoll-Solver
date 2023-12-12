import numpy as np
from matplotlib import pyplot as plt
import json
from poly_gen import *
json_path = "results_CP.json"
with open(json_path, 'r') as json_file:
    data = json.load(json_file)
data = data['1.4']

for mach, m_dict in data.items():
    mach = float(mach)
    print(mach)
    thetas = m_dict['thetas']
    betas = m_dict['betas']
    surface_machs = m_dict['surface_machs']
    cps = m_dict['cps']
    plt.plot(thetas, cps, label = f"Mach {mach}")

plt.xlabel('Theta (rad)')
plt.ylabel('Pressure Coefficient')
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
    cps = m_dict['cps']
    ax.scatter(np.ones(len(thetas))*mach, thetas, cps, label=f'M {mach}')

coefficients = create_coeffs('results_Big.json', 'cps', orders=[2,3,6])

gen_thetas = np.linspace(0,1,30)
gen_machs = np.linspace(1,30,30)
gen_thetas, gen_machs = np.meshgrid(gen_thetas, gen_machs)
gen_cps = evaluate_function(coefficients, [1.4*np.ones_like(gen_thetas), gen_machs, gen_thetas], orders=[2,3,6])
ax.contour3D(gen_machs, gen_thetas, gen_cps, 50, cmap='binary')

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
    cps = m_dict['cps']
    error = (cps - evaluate_function(coefficients, [np.ones(len(thetas))*1.4, np.ones(len(thetas))*mach, np.array(thetas)], orders=[2,3,6]))*100
    ax.scatter(np.ones(len(thetas))*mach, thetas, error)

ax.set_xlabel('Mach Number')
ax.set_ylabel('Theta')
ax.set_zlabel('Error (%)')
plt.show()