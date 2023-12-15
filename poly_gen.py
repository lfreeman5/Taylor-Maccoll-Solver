from poly_fits import *
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt

def create_coeffs(json_name, key, orders=[3,4,6]):
    with open(json_name, 'r') as json_file:
        data = json.load(json_file)
    
    independent_variables = []
    dependent_variables = []

    for g, g_dict in data.items():
        g=float(g)
        for j, (m, m_dict) in enumerate(g_dict.items()):
            m=float(m)
            thetas = m_dict['thetas']
            dependent_vars = m_dict[key]
            for i in range(len(thetas)):
                # independent_variables.append([g,m,thetas[i]])
                independent_variables.append([thetas[i],m,g])
                dependent_variables.append(dependent_vars[i])


    coefficients, r2_value = multivariablePolynomialFit(orders, independent_variables, dependent_variables)
    return coefficients, r2_value

def evaluate_function(coeffs, input_vector, orders=[3,4,6]): #input_vector takes the form [gamma, mach, theta]
    return multivariablePolynomialFunction(coeffs, orders, input_vector)

def create_coeffs_threshold(json_name, key, orders, threshold):
    with open(json_name, 'r') as json_file:
        data = json.load(json_file)
    
    independent_variables_low = []
    dependent_variables_low = []
    independent_variables_high = []
    dependent_variables_high = []

    for g, g_dict in data.items():
        g=float(g)
        for j, (m, m_dict) in enumerate(g_dict.items()):
            m=float(m)
            thetas = m_dict['thetas']
            dependent_vars = m_dict[key]
            for i in range(len(thetas)):
                if(thetas[i] < threshold):
                    independent_variables_low.append([thetas[i],m,g])
                    dependent_variables_low.append(dependent_vars[i])
                else:
                    independent_variables_high.append([thetas[i],m,g])
                    dependent_variables_high.append(dependent_vars[i])  

    coefficients_low, r2_value_low = multivariablePolynomialFit(orders, independent_variables_low, dependent_variables_low)
    coefficients_high, r2_value_high = multivariablePolynomialFit(orders, independent_variables_high, dependent_variables_high)

    return [coefficients_low, coefficients_high], [r2_value_low, r2_value_high]


def evaluate_function_threshold(coeffs, input_vector, orders, threshold):
        if(input_vector[0] < threshold):
            return multivariablePolynomialFunction(coeffs[0], orders, input_vector)
        else:
            return multivariablePolynomialFunction(coeffs[1], orders, input_vector)

