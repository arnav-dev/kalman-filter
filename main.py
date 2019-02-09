Python 2.7.12 (v2.7.12:d33e0cf91556, Jun 27 2016, 15:19:22) [MSC v.1500 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import numpy as np
def predicteState(X_k_p,U_k):

    # To make the prediction from just passed state about currently happening state.
    #Here X_k_p (passed state vector) is 2 X 1, and U_k (control vector) is 1 X 1.
    
    F_k = np.array([[1,dt],[0,1]])        #Prediction matrix 2 X 2
    B_k = np.array([[(dt**2)/2],[dt]])

    X_k = F_k.dot(X_k_p) + B_k.dot(U_k)
    
    return(X_k)
    
def predicteCovariance(P_k_p):

    # To figure out the trend.
    # Here P_k_p is state covariance matrix of order 2. 

    Q_k = [[0,dt**2],[dt**2,0]]      #Process noice
    F_k = np.array([[1,dt],[0,1]])        #Prediction matrix 2 X 2

    P_k = (F_k.dot(P_k_p)).dot(F_k)

    return(P_k)
    
def updateState(X_k,Z_k,K):

    # To fuse the sensor values with predicted values to get actual state.
    # X_k is the predicted state taken out from state prediction.
    # Z_k is the sensor reading.

    H_k = np.array([[1,0]])      #Observation matrix

    X_f = X_k + K.dot(Z_k-H_k.dot(X_k))

    return(X_f)


def kalmanGain(P_k):

    # To decide on what we have to rely more measurement or prediction
    
    H_k = np.array([[1,0]])      #Observation matrix
    R_k = np.array([[0]])                 #Observation noise

    K = (P_k.dot(H_k.T))/((H_k.dot(P_k)).dot(H_k.T)+R_k)

    return(K)

def updateCovariance(P_k,K):

    #We cannot miss the trend
    
    H_k = np.array([[1,0]])      #Observation matrix
     
    P_f  = P_k - (K.dot(H_k)).dot(P_k)

    return(P_f)
    
dt = 0.01      # Sample time
res = 0.01    #resolution of sensor

X_k_p = np.array([[3],[4]])
U_k   = np.array([[1]])
Z_k   = np.array([[2.5]])
P_k_p = np.array([[1,2],[2,1]])

P_f = updateCovariance(predicteCovariance(P_k_p),kalmanGain(predicteCovariance(P_k_p)))
X_f = updateState(predicteState(X_k_p,U_k),Z_k,kalmanGain(predicteCovariance(P_k_p)))

print('State matrix',X_f)
print('Covariance matrix',P_f)
