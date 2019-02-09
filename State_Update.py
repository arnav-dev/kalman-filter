import numpy as np

res = 0.01    #resolution of sensor

def updateState(X_k,Z_k,K):

    # To fuse the sensor values with predicted values to get actual state.
    # X_k is the predicted state taken out from state prediction.
    # Z_k is the sensor reading.

    H_k = np.array([[1,0]])      #Observation matrix

    X_f = X_k + K.dot(Z_k-H_k.dot(X_k))

    return(X_f)


def kalmanGain(P_k):

    # To decide on what we have to rely more measurement or prediction
    global res
    H_k = np.array([[1,0]])      #Observation matrix
    R_k = np.array([[0]])                 #Observation noise

    K = (P_k.dot(H_k.T))/((H_k.dot(P_k)).dot(H_k.T)+R_k)

    return(K)

def updateCovariance(P_k,K):

    #We cannot miss the trend
    
    H_k = np.array([[1,0]])      #Observation matrix
     
    P_f  = P_k - (K.dot(H_k)).dot(P_k)

    return(P_f)
