# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

def chebypoints(n):
    """
    Finds reference points using the cheby-method
    """
    chebypoints = []
    for i in range(n+1):
        points = cos(((n+1-i)*pi)/(n+1))
        chebypoints.append(points)
    return(chebypoints)
        
    


def func(x):
    """
    The function that is to be approximated
    """
    return(1/(1+(25*x**2)))

             
def poly_constructor(n,c,x,basis):
    """
    Constructs the polynomial from given coefficients, and basis
    """
    p=0
    for j in range(n):
        p += (basis[0][j]*c[j])*(x**(basis[1][j]))
    
      
    return p
    
    
def basis_constructor(ref,basis,k,m):
    """
    Constructs the basis
    """
    #print(basis[0][k]*ref[m]**(basis[1][k]))
    return(basis[0][k]*ref[m]**(basis[1][k]))
    

   
def exchange_algorithm(func,n,interval,ref,basis,tol,itera,vector):
    """
    This function produces the minimax approximation of a continuous function using
    the exchange algorithm. 
    """
    #Set up variables:
    x = np.linspace(interval[0],interval[1],1000)
  
    
    #If basis is empty assume dimension (n-1)
    # Monomial basis of dimension (n-1)
    if (basis == None):
        basis = [[],[]]
        for i in range(n):
            basis[0].append(1)
            basis[1].append(i)
          
    #coefficient matrix calculations, and b vector calculations
    c_matrix = np.array([])
    temp = np.array([])
    b = np.array([])
                
    for i in range(n):
        c_matrix = np.append(c_matrix,basis_constructor(ref,basis,i,0))
    c_matrix = np.append(c_matrix,1)
       


    for i in range(n):
        b = np.append(b,func(ref[i]))

        for j in range(n):
            temp = np.append(temp,basis_constructor(ref,basis,j,i+1))
                
                
            
        temp = np.append(temp,(-1)**(i+1))
        c_matrix = np.row_stack((c_matrix,temp))
       
        temp = array([])

    b = np.append(b,func(ref[-1]))

    #solve for Coefficients for polynomial
    c = np.linalg.solve(c_matrix,b)
    
    #construct polynomial in A
    p = poly_constructor(n,c,x,basis)
    
    #construct error function
    e_function = array([])
    for i in range(len(x)):
        e_function = np.append(e_function,func(x[i])-p[i])
    

    #finds the maximum error
    error_index = np.where(abs(e_function)==max(abs(e_function)))
    error_index = error_index[0][0]
    error_max = e_function[error_index]
    

    #Searching for the references
    ref_max = x[error_index]
    ref.append(ref_max)
    ref.sort()   
   
   
    #Adjusts reference       
    if (ref[-1] == ref_max):
        if (error_max*((-1)**(len(ref)))*c[-1] > 0):
            del ref[-2]
        else:
            del ref[0]

    
    elif (ref[0] == ref_max):
        if (error_max*c[-1] > 0):
            del ref[1]
        else:
            del ref[-1]
            
    else:
        index_ref_max = ref.index(ref_max)

        
        if (error_max*(-1)**(index_ref_max-1)*c[-1]>0):
            del ref[index_ref_max-1]
        else:
            del ref[index_ref_max+1]
           
    
    
    #final things:
    error = abs(func(ref_max)-p[error_index])
    itera = itera+1
    vector.append(c[-1])
    tol_error = error-abs(c[-1])
    
    if (itera==20):
        
        print("the algorithm did not achieve an error less than tolerance in 20 iterartions")
        print("Coefficients of polynomial ", c)
        print("The Error: ", error)
        print("Number of iterations: ", itera)
        print("Reference_level vector:", vector)
        print("Actual reference", ref)
        print(error)
        
        plt.plot(x,e_function)
        plt.title("Error function")
        plt.plot(x,c[-1]*np.ones(len(x)))
        plt.plot(x,-c[-1]*np.ones(len(x)))
        plt.show()
                
        plt.plot(x,func(x))
        #plt.plot(x,[0.3,4.2,0.1,3.4,5.7,4.9,5.7])
        plt.title("Final approximation, n: {}".format(n))
        plt.plot(x,p)
        plt.show()
        
    
    elif (tol_error>tol):
        exchange_algorithm(func,n,interval,ref,basis,tol,itera,vector)
        
    
    else:
        print("Coefficients of polynomial ", c)
        print("The Error: ", error)
        print("Number of iterations: ", itera)
        print("Reference_level vector:", vector)
        print("Actual reference", ref)
        
        plt.plot(x,e_function)
        plt.title("Error function")
        plt.plot(x,c[-1]*np.ones(len(x)))
        plt.plot(x,-c[-1]*np.ones(len(x)))
        plt.show()  
        
        plt.plot(x,func(x))
        plt.title("Final approximation, n: {}".format(n))
        plt.plot(x,p)
        plt.show()
        

####################################################
########################## Main ####################
####################################################

# Inputs
#n = 19
vector = []
n=15
interval = [-1.,1.]
tol = 1.e-15
basis = None
itera = 0

ref = chebypoints(n)

# Function call.
exchange_algorithm(func,n,interval,ref,basis,tol,itera,vector)
