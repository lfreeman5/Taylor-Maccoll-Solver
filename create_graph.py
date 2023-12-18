import numpy as np
from matplotlib import pyplot as plt
import json
from poly_gen import *

def plot_vs_trendline():
    json_path = "Results_Polys.json"
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    data = data['1.4']

    thetas=np.array(np.degrees(data['8']['thetas']))
    cps=np.array(data['8']['cps'])

    coefficients = np.polyfit(thetas, cps, 6)

    plt.scatter(thetas, cps, label='Taylor-Maccoll Results')
    thetas_fit = np.linspace(min(thetas), max(thetas), 100)
    cps_fit = np.polyval(coefficients, thetas_fit)
    plt.plot(thetas_fit, cps_fit, label='6th-order Polynomial Fit', color='red')

    # Calculate percent error
    abs_error = np.log10(abs((-cps + np.polyval(coefficients, thetas))))

    fig, ax1 = plt.subplots()
    ax1.scatter(thetas, cps, label='Original Data', color='blue')
    ax1.plot(thetas_fit, cps_fit, label='6th-order Polynomial Fit', color='red')
    ax1.set_xlabel('Cone Angle (deg)')
    ax1.set_ylabel('C_P', color='blue')
    ax1.tick_params('y', colors='blue')

    ax2 = ax1.twinx()
    # ax2.plot(thetas, abs_error, label='Percent Error', color='green', marker='v')
    # ax2.set_ylabel('Percent Error', color='green')
    # ax2.tick_params('y', colors='green')
    # ax2.set_ylim([-10, 10])

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plt.title('Taylor-Maccoll Pressure Coefficient results vs. 6th-order polynomial fit')
    plt.grid(True)
    plt.show()

def threeD_plot():
    json_path = "Results_Polys.json"
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    for g, g_dict in data.items():
        g = float(g)
        thetas=np.degrees(g_dict['8']['thetas'])
        cps = g_dict['8']['cps']
        plt.plot(thetas, cps, label = f"Gamma {g}")
    plt.title('Pressure Coefficient vs. Cone Angle at Mn=8')
    plt.xlabel('Cone Angle (deg)')
    plt.ylabel('C_P')
    plt.legend()
    plt.show()

    data = data['1.4']
    orders = [6,4,2] #theta, mach, gamma
    for mach, m_dict in data.items():
        mach = float(mach)
        print(mach)
        thetas = m_dict['thetas']
        betas = m_dict['betas']
        surface_machs = m_dict['surface_machs']
        cps = m_dict['cps']
        plt.plot(np.degrees(thetas), cps, label = f"Mach {mach}")

    plt.title('Pressure Coefficient vs. Cone Angle at Gamma=1.4')
    plt.xlabel('Cone Angle (deg)')
    plt.ylabel('C_P')
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

if __name__ == "__main__":
    plot_vs_trendline()
    threeD_plot()