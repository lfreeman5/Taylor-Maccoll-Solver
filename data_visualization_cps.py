import numpy as np
from matplotlib import pyplot as plt
import json
from poly_gen import *
json_path = "results_CP.json"
with open(json_path, 'r') as json_file:
    data = json.load(json_file)
data = data['1.4']
orders = [6,3,3]
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

coefficients, r2 = create_coeffs('results_Big.json', 'cps', orders=orders)
print(f"R^2 Value for polynomial fit: {r2}")

gen_thetas = np.linspace(0,1,30)
gen_machs = np.linspace(1,30,30)
gen_thetas, gen_machs = np.meshgrid(gen_thetas, gen_machs)
gen_cps = evaluate_function(coefficients, [gen_thetas, gen_machs, 1.4*np.ones_like(gen_thetas)], orders=orders)
ax.contour3D(gen_machs, gen_thetas, gen_cps, 50, cmap='binary')

ax.set_xlabel('Mach Number')
ax.set_ylabel('Theta')
ax.set_zlabel('Surface Pressure Coefficient')
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for mach, m_dict in data.items():
    mach = float(mach)
    print(mach)
    thetas = m_dict['thetas']
    cps = np.array(m_dict['cps'])
    evaluated_cps = evaluate_function(coefficients, [np.array(thetas), np.ones(len(thetas))*mach, np.ones(len(thetas))*1.4], orders=orders)
    print(cps)
    print(evaluated_cps)
    # error = (cps - evaluated_cps)/cps*100
    error = np.log10(abs(cps - evaluated_cps))

    ax.scatter(np.ones(len(thetas))*mach, thetas, error)

ax.set_zlim(-10,10)
ax.set_xlabel('Mach Number')
ax.set_ylabel('Theta')
ax.set_zlabel('Percent error')
plt.show()