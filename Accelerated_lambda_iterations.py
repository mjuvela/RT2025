import sys
from numpy import *

if (len(sys.argv)<2):
    print("Usage:")
    print(" python Accelerated_lambda_iterations.py 0  # for normal, non-accelerated run")
    print(" python Accelerated_lambda_iterations.py 1  # ... for accelerated (~ALI) run\n")
    sys.exit(0)
    
accelerated = sys.argv[1]=="1"

# Demonstrate Accelerated Lambda Iterations (ALI): case of a single-cell model.
# We use the "diagonal operator" =>
#   take changed emission of the cell immediately into account
#   (the same iteration) when updating the temperature

def Emission(T):        # Energy emitted by the cell
    return T**4

def Temperature(E):     # Energy to temperature
    return E**0.25

def ExternalHeating():  # External heating of the cell
    return 1e3

beta  = 0.1    # escape probability
T     = 1.0    # initial temperature guess
Tp    = 0.0    # Tp will be the previous temperature estimate
TOL   = 0.001  # tolerance

print("------------------------------------------------------------")
print("beta = %.2f" % beta)

for iter in range(1, 10001):  # a maximum of 1000 iterations

    Tp    =  T   # Temperature estimate from previous iteration
        
    if (not(accelerated)): #  This branch using normal lambda iterations        
        # Current heating rates
        H   =  ExternalHeating() + (1.0-beta)*Emission(T)
        # New temperature estimate
        T   =  Temperature(H)
        
    else:   #  This branch using "Accelerated Lambda Iterations"
        
        #   COOLING RATE            =  HEATING RATE
        #   ------------               ------------
        #   Emission(T)             =  ExternalHeating() + (1.0-beta)*Emission(T)
        #   <==>
        #   Emission(T)*beta        =  ExternalHeating()
        H  =   ExternalHeating()
        T  =   Temperature(H/beta)
            
    print("Iteration %2d      T = %7.4f" % (iter, T))
    if (abs(T-Tp)<TOL):
        break   # test for convergence
        
print("------------------------------------------------------------")        


