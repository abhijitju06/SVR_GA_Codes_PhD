"""
MIMO model only
"""
import numpy as np                                      #Numpy -> for array
import pickle                                           #Data pickling

class model:
    def __init__(self):
        self.Load_model()
        return

    def Load_model(self):
        with open('model_with_thread_without_scaling.pickle', 'rb') as fc:                  #Retrieve the saved model
            self.clfs = pickle.load(fc)
        with open('op_dim_with_thread_without_scaling.pickle', 'rb') as fc:                 #Retrieve output dimensions
            self.n = pickle.load(fc)
        return
    
    def Predict(self, X_test):
        Y_pred = np.zeros((X_test.shape[0], self.n))
        for j in range(self.n):
            Y_pred[:,j] = self.clfs[j].predict(X_test)          #Predict each feature(column) of output
        return Y_pred

