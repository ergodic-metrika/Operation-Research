# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:30:38 2022

@author: sigma
"""
#Lecture material from CQF
# Import required libraries
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize_scalar, minimize

# Define objective function
def objective_function(x):
    return x**2 -2*x

results1 = minimize_scalar(objective_function)
results1


# Initial Guess
X0 = 3.0

# Optimize the function
results2 = minimize(fun=objective_function, x0=X0)

# Ouput the result
results2


# Plot the function
x = np.linspace(-3,5,100)
plt.plot(x,objective_function(x))
plt.plot(results2.x,objective_function(results2.x),'ro')


#Constrained optimization
# Specify constraints
cons = ({'type': 'ineq', 'fun' : lambda x: np.array([x[0] - 2])})
results4 = minimize(objective_function, x0=X0, constraints = cons)
results4


# Specify boundary condition
bnds = ((2, None),) # x less than 2 will have a negative values
X0, len(bnds) # your length of boundary should match with the length of variable
results5 = minimize(objective_function, x0=X0, method='SLSQP', bounds=bnds)
results5


#Multiple constraints
new_objective_function = lambda x: x[0]**2 + x[0]*x[1]

cons = ({'type': 'eq', 'fun' : lambda x: x[0]**3 + x[0]*x[1] - 100},
        {'type': 'ineq', 'fun' : lambda x: x[0]**2 + x[1] - 50})

intial_xs = [1,1]
boundary = ((-100,100), (-100,100))
results6 = minimize(new_objective_function, intial_xs, method='SLSQP', bounds=boundary, constraints=cons)
results6