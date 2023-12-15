import numpy as np
from matplotlib import pyplot as plt
import json
from poly_gen import *
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
ax1.set_xlabel('Thetas')
ax1.set_ylabel('Cps', color='blue')
ax1.tick_params('y', colors='blue')

ax2 = ax1.twinx()
ax2.plot(thetas, abs_error, label='Percent Error', color='green', marker='v')
ax2.set_ylabel('Percent Error', color='green')
ax2.tick_params('y', colors='green')
# ax2.set_ylim([-10, 10])

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')
plt.title('Taylor-Maccoll Results vs. Polynomial fit at gamma=1.4, M8.0')
plt.grid(True)
plt.show()