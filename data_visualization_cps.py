import numpy as np
from matplotlib import pyplot as plt
import json
from poly_gen import *
json_path = "Results_Mtest.json"
with open(json_path, 'r') as json_file:
    data = json.load(json_file)
data = data['1.4']
orders = [6,4,2] #theta, mach, gamma
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

# coefficients, r2 = create_coeffs('results_CP.json', 'cps', orders=orders)
coefficients, r2 = create_coeffs_threshold('results_CP.json', 'cps', orders, np.deg2rad(10))

print(f"R^2 Value for polynomial fit: {r2}")

gen_thetas = np.linspace(0,1,30)
gen_machs = np.linspace(1,30,30)
gen_thetas, gen_machs = np.meshgrid(gen_thetas, gen_machs)
# Create an empty array to store the results
gen_cps = np.empty_like(gen_thetas)

# Iterate through the meshgrid
for i in range(gen_thetas.shape[0]):
    for j in range(gen_thetas.shape[1]):
        # Evaluate the function for each combination of gen_thetas[i, j] and gen_machs[i, j]
        gen_cps[i, j] = evaluate_function_threshold(coefficients, [gen_thetas[i, j], gen_machs[i, j], 1.4], orders, np.deg2rad(10))

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
    # evaluated_cps = evaluate_function(coefficients, [np.array(thetas), np.ones(len(thetas))*mach, np.ones(len(thetas))*1.4], orders=orders)
    evaluated_cps = np.empty_like(thetas)

    # Iterate through the elements of thetas
    for i in range(len(thetas)):
        # Evaluate the function for each combination of thetas[i], mach, and a constant value
        evaluated_cps[i] = evaluate_function_threshold(coefficients, [thetas[i], mach, 1.4], orders, np.deg2rad(10))

    # error = (cps - evaluated_cps)/cps*100
    error = np.log10(abs(cps - evaluated_cps))

    ax.scatter(np.ones(len(thetas))*mach, thetas, error)

ax.set_zlim(-10,10)
ax.set_xlabel('Mach Number')
ax.set_ylabel('Theta')
ax.set_zlabel('Percent error')
plt.show()