# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:01:42 2023

@author: zhanr
"""

from numpy import *
import scipy.sparse as sp
from scipy.integrate import solve_ivp
import numpy as np

# This code is used for question 8

n = 20
nr = zeros(n)
L = zeros(n)
for  i in range(0,n):
    # Setting the grid spacing and radial positions of grids
    nr[i] = 30+30*i # Number of grids
    Nr = int(nr[i])
    R = 0.05 # Radius of catalyst
    dr = R/(Nr-1)

    # Setting the boundary and initial conditions
    Cs = 1 # Concentration at r = R
    Cinitial = zeros(Nr)
    Cinitial[0] = Cs
    D = 4*(10**(-5))
    k = 0

    rreversed = linspace(0,R,Nr) # In reversed order
    #print("Reversed Array =", rreversed)

    r = rreversed[::-1]
    #print("Array =",r)

    # Create a sparse matrix, M, which is the Jacobian matrix
    M = zeros((Nr,Nr))

    for j in range(1,Nr-1):
        M[j,j-1] = D*(1/r[j]**2)*((r[j]+0.5*dr)**2)*(1/dr**2)
        M[j,j] = -(D/(r[j]**2))*((r[j]-0.5*dr)**2+(r[j]+0.5*dr)**2)/(dr**2)-k
        M[j,j+1] = D*(1/r[j]**2)*((r[j]-0.5*dr)**2)*(1/dr**2)
        
    # Set the zero gradient boundary condition at center
    # Method 1
    #M[Nr-1,Nr-1-2] = 3*D/(2*dr**2)
    #M[Nr-1,Nr-1] = -3*D/(2*dr**2)-k
    # Method 2
    M[Nr-1,:] = M[Nr-2,:]
    #print(M[Nr-1,:])

    # Create a vector for constants
    b = zeros(Nr)

    # Find eigenvalues of matrix M
    from numpy.linalg import eig
    J = M[1:,1:]
    eigvalue,eigvector = eig(J)
    L[i] = min(eigvalue)

# Plot
import matplotlib.pyplot as plt
plt.plot(nr,L)
plt.xlabel("Number of nodes")
plt.ylabel("Largest magnitude of eigenvalue")
plt.title("Relationship between largest magnitude of eigenvalue and number of nodes")
plt.show()