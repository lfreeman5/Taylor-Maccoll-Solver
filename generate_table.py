import numpy as np
import pandas as pd
import json
"""
This program takes JSON theta-gamma-m-CP results and formats them in strings
to be copy-pasted into fortran table
"""


json_path = "results_Polys.json"
with open(json_path, 'r') as json_file:
    data = json.load(json_file)

gammas = []
machs = []
results = []
for g, g_dict in data.items():
    g=float(g)
    gammas.append(g)
    arr = []
    machs = []
    for m, m_dict in g_dict.items():
        m=float(m)
        machs.append(m)
        thetas=m_dict['thetas']
        theta_max = max(thetas)
        cps=m_dict['cps']
        coefficients = np.polyfit(thetas,cps,6)
        a=[theta_max]
        a.extend(coefficients)
        arr.append(a)
        print(f"For g={g} and M={m}, thetamax is {theta_max} and the coeffs are:\n{coefficients}")
        print(f"The predicted CP at theta=0.4: {np.polyval(coefficients, 0.4)}")
    results.append(arr)

totalStr = "(/ "
for arr in results:
    for a in arr:
        for s in a:
            totalStr += (str(s)+ ', ')

totalStr = totalStr[:-2] + " /)"
lineLengthLimit = 70
newStr = ""
currentLine = 0
for c in totalStr:
    newStr += c
    currentLine += 1
    if(currentLine>lineLengthLimit-15 and c == " "):
        newStr += '&\n'
        currentLine = 0


machStr = "(/ "
for m in machs:
    machStr += (str(m) + ', ')
machStr = machStr[:-2] + " /)"

gammaStr = "(/ "
for g in gammas:
    gammaStr += (str(g) + ', ')
gammaStr = gammaStr[:-2] + " /)"

with open('table_output.txt', 'w+') as file:
    file.write("totalStr: " + newStr + '\n')
    file.write("machStr: " + machStr + '\n')
    file.write("gammaStr: " + gammaStr + '\n')