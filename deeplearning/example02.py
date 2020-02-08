#(2) example02
import numpy as np
import matplotlib.pyplot as plt

#only number not array
def step_function(x):
    if x>0:
        return 1
    else:
        return 0

#availble array
def step_function_array(x):
    y = x > 0
    return y.astype(np.int)

#print graph
x = np.arange(-5.0,5.0,0.1)
y = step_function_array(x)

plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()


