
import State_Prediction
import State_Update
import numpy as np


X_k_p = np.array([[3],[4]])
U_k   = np.array([[1]])
Z_k   = np.array([[2.5]])
P_k_p = np.array([[1,2],[2,1]])

P_f = State_Update.updateCovariance(State_Prediction.predicteCovariance(P_k_p),State_Update.kalmanGain(State_Prediction.predicteCovariance(P_k_p)))
X_f = State_Update.updateState(State_Prediction.predicteState(X_k_p,U_k),Z_k,State_Update.kalmanGain(State_Prediction.predicteCovariance(P_k_p)))

print('State matrix',X_f)
print('Covariance matrix',P_f)
raw_input("Press ENTER to exit")
