#==========================================================================
#
#   Author      : NMD
#   Github      : https://github.com/manhdung20112000/
#   Email       : manhdung20112000@gmail.com
#   File        : linear_regression.py
#   Created on  : 2021-3-17
#   Description : linear regression example
#
#==========================================================================

# the example above an easy introduction of linear regression
# the input of the algorithm only have 1 dimension (1D)
# the (silly) task is whethere we can guess a person's weight if we know his/he height?

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import ones

# height (cm)
x = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# f(x) = weight = w_1*height + w_0
# loss function: L(w) = 1/2*(sum_{i=1}^N (y - x*w)^2)

# X bar = [1, x_1, x_2]
addition_one = np.ones((x.shape[0], 1))
x_bar = np.concatenate((addition_one, x), axis = 1)

# Solve loss function by derivation = 0
A = np.dot(x_bar.T, x_bar)
b = np.dot(x_bar.T, y)
w = np.dot(np.linalg.pinv(A), b) # pinv is pseudo inverse (gia nghich)
print('w = ', w)

# Draw the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185)
print('x0 = ', x0)
y0 = w_1*x0 + w_0
print('y0 =', y0)

# Visualize data 
plt.plot(x, y, 'ro')
plt.plot(x0, y0)
plt.axis((140, 190, 45, 75))
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show(block=True)

