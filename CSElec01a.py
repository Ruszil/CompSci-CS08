import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)  
X = np.random.rand(100) * 10  
Y = 2.5 * X + np.random.randn(100) * 2  

mean_x = np.mean(X)
mean_y = np.mean(Y)


numerator = np.sum((X - mean_x) * (Y - mean_y))
denominator = np.sum((X - mean_x) ** 2)


slope = numerator / denominator
intercept = mean_y - (slope * mean_x)


print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

regression_line = slope * X + intercept


plt.title('Manual Linear Regression')
plt.plot(X, regression_line, color='red', label='Regression Line')
plt.show()
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()

plt.savefig(sys.stdout.buffer)
sys.stdout.flush()
