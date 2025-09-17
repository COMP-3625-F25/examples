import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from optimization_test_functions import himmelblaus_function


result = minimize(himmelblaus_function, x0=[0, 0], method='Nelder-Mead')
print(result)