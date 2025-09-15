import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


def himmelblaus_function(input_vector) -> float:
    x, y = input_vector
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

