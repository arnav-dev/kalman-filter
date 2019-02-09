import numpy as np

dt = 0.01      # Sample time

def predicteState(X_k_p,U_k):

    # To make the prediction from just passed state about currently happening state.
    #Here X_k_p (passed state vector) is 2 X 1, and U_k (control vector) is 1 X 1.
    
    global dt

    F_k = np.array([[1,dt],[0,1]])        #Prediction matrix 2 X 2
    B_k = np.array([[(dt**2)/2],[dt]])

    X_k = F_k.dot(X_k_p) + B_k.dot(U_k)
    
    return(X_k)

def predicteCovariance(P_k_p):

    # To figure out the trend.
    # Here P_k_p is state covariance matrix of order 2. 
    
    global dt

    Q_k = [[0,dt**2],[dt**2,0]]      #Process noice
    F_k = np.array([[1,dt],[0,1]])        #Prediction matrix 2 X 2

    P_k = (F_k.dot(P_k_p)).dot(F_k)

    return(P_k)



