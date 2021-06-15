import matplotlib.pyplot as plt
import numpy as np
import math

def HW(x):
    a=1+x+1/2*x**2+1/6*x**3+1/24*x**4
    b=(1+4/3*x+1/4*x**2+1/24*x**3)/(1-1/4*x)
    c=(1+1/2*x+1/12*x**2)/(1-1/2*x+1/12*x**2)
    d=(1+1/4*x)/(1-3/4*x+1/4*x**2-1/24*x**3)
    e=1/(1-x+1/2*x**2-1/6*x**3+1/24*x**4)
    f=math.e**x
    g=[a,b,c,d,e,f]
    return g

aluf = HW(0.5)
print(aluf)
beta = HW(1)
print(beta)
ganm = HW(-1)
print(ganm)
delt = HW(2)
print(delt)


plt.figure()

x=np.linspace(-1,5,100);
def y(a):
     return np.e**x

a=1+x+1/2*x**2+1/6*x**3+1/24*x**4
b=(1+3/4*x+1/4*x**2+1/24*x**3)/(1-1/4*x)
c=(1+1/2*x+1/12*x**2)/(1-1/2*x+1/12*x**2)
d=(1+1/4*x)/(1-3/4*x+1/4*x**2-1/24*x**3)
e=1/(1-x+1/2*x**2-1/6*x**3+1/24*x**4)


plt.plot(x,a,label="[4/0]")
plt.plot(x,b,label="[3/1]")
plt.plot(x,c,label="[2/2]")
plt.plot(x,d,label="[1/3]")
plt.plot(x,e,label="[0/4]")

plt.plot(x,y(x),label='e^x')
plt.ylim(-5,10)
plt.legend()
plt.grid()
plt.show()


'''
import numpy as np
import math
import sympy
from sympy.plotting import plot
sympy.init_printing(use_unicode=True)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def mcLaughlin_expansion(formula,k_max):
  f = formula
  approximation = np.sum([((-1)**(k))*math.factorial(k)*((x)**k) for k in range(k_max)])
  return approximation

x = sympy.symbols("x")
formula = x
x == 0.2

y = sympy.symbols("y")
k_max = y


for i in range (100):
  k_max == i+1
  fmcLaughlin_expansion(formula,k_max)
  sympy.plot(f1,formula,
                legend=True, show=False)

plt.show()







print("formula: ",formula)
print("mcLaughlin: ",mcLaughlin_expansion(formula,8))
print("mcLaughlin: ",mcLaughlin_expansion(formula,9))
print("mcLaughlin: ",mcLaughlin_expansion(formula,10))
print("mcLaughlin: ",mcLaughlin_expansion(formula,11))
print("mcLaughlin: ",mcLaughlin_expansion(formula,12))
print("mcLaughlin: ",mcLaughlin_expansion(formula,13))
print("mcLaughlin: ",mcLaughlin_expansion(formula,14))
print("mcLaughlin: ",mcLaughlin_expansion(formula,15))
print("mcLaughlin: ",mcLaughlin_expansion(formula,21))
print("mcLaughlin: ",mcLaughlin_expansion(formula,22))
print("mcLaughlin: ",mcLaughlin_expansion(formula,23))
print("mcLaughlin: ",mcLaughlin_expansion(formula,24))
print("mcLaughlin: ",mcLaughlin_expansion(formula,25))
'''

