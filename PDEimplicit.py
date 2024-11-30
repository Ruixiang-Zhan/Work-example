# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:01:42 2023

@author: zhanr
"""

from numpy import *
import scipy.sparse as sp
from scipy.integrate import solve_ivp
import numpy as np

# Setting the grid spacing and radial positions of grids
Nr = 500 # Number of grids
R = 0.05 # Radius of catalyst
dr = R/(Nr-1)

# Setting the boundary and initial conditions
Cs = 1 # Concentration at r = R
Cinitial = zeros(Nr)
Cinitial[0] = Cs
D = 4*10**(-5)
k = 0

rreversed = linspace(0,R,Nr) # In reversed order
#print("Reversed Array =", rreversed)

r = rreversed[::-1]
#print("Array =",r)

# Create a sparse matrix, M, which is the Jacobian matrix
M = zeros((Nr,Nr))

for i in range(1,Nr-1):
    M[i,i-1] = D*(1/r[i]**2)*((r[i]+0.5*dr)**2)*(1/dr**2)
    M[i,i] = -(D/(r[i]**2))*((r[i]-0.5*dr)**2+(r[i]+0.5*dr)**2)/(dr**2)-k
    M[i,i+1] = D*(1/r[i]**2)*((r[i]-0.5*dr)**2)*(1/dr**2)

# Set the zero gradient boundary condition at center
# Method 1
#M[Nr-1,Nr-1-2] = 3*D/(2*dr**2)
#M[Nr-1,Nr-1] = -3*D/(2*dr**2)-k
# Method 2
M[Nr-1,:] = M[Nr-2,:]
#print(M[Nr-1,:])

# Create a vector for constants
b = zeros(Nr)

def dC_dt(t,C):
    dC = M.dot(C)+b
    return dC

# Time constant for diffusion only
tconstant = (R**2)/D
print("Time constant =", tconstant)

# Solve
n =  1 # Number of time constant
nt = 1000 # Number of outputs
outputtimes = np.linspace(0,n*tconstant,nt)
#print("out = ", outputtimes)
ans = solve_ivp(dC_dt,(0,n*tconstant),Cinitial,method="Radau",t_eval=outputtimes, jac_sparsity= M)
#print(ans.t)

# Find eigenvalues of matrix M
from numpy.linalg import eig
J = M[1:,1:]
eigvalue,eigvector = eig(J)
L = min(eigvalue)
print("Largest eigenvalue =", L)

# Plot
import matplotlib.pyplot as plt

nplot = 10 # Number of curves in the plot

for i in range(0,nplot):
    tspace = int((nt-1)/(nplot-2)) # Convert to integer
    a = i*tspace
    # Limit n to be lower than total number of outputs, nt
    if a > nt-1:
        a = nt-1
    if a == 0:
        b = 0
    else:
       b = ((a+1)/nt)*n
    plt.plot(r,ans.y[:,a], label = "Time ={} time constant".format(b))

plt.xlabel("Radial position in m")
plt.ylabel("Concentration in mol/m^3")
plt.title("Concentration profile")
plt.legend(bbox_to_anchor=(1,0.9))
plt.show()

# Plot the space-time diagram
space = r
time = zeros(int(nt))
for i in range(0,nt):
    time[i] = i*(n/(nt-1))

# Create 2D grid
[X, Y] = np.meshgrid(space,time)
fig, ax= plt.subplots()

# Plot contours
z = np.transpose(ans.y)
vmin = min(0,z.min())
vmax = max(1,z.max())
conplot = plt.contourf(space,time,z,cmap='plasma',levels = linspace(vmin,vmax,400))
fig.colorbar(conplot,label = "Concentration in mol/m^3")
plt.xlabel("Radial position in m")
plt.ylabel("Number of time constant")
plt.title("Space-time diagram")
plt.show()
