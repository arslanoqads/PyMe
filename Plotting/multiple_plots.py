import matplotlib.pyplot as plt
import numpy as np


x = range(5)
y = np.exp(x)
plt.figure(0)
plt.plot(x, y)

z = np.sin(x)
plt.figure(1)
plt.plot(x, z)

w = np.cos(x)
plt.figure(0) # Here's the part I need
plt.plot(x, w)