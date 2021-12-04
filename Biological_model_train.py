"""
MIMO using multiple SVRs
TRAINING
"""

from sklearn.preprocessing import MinMaxScaler
from math import exp                                    #exponent function  
from sklearn import svm                                 #Support Vector Machine module
import numpy as np                                      #Numpy -> for array
#import xlsxwriter                                       #Excel read/write operations
import time as tm                                       #Time module
import pickle                                           #Data pickling
from threading import Thread				# Multithread implementation

class heli_data:                                        #Class for helicopter data set handling
    def __init__(self):                                   
        return
    
    def Create_batch(self):                             #Creates a data set with random k samples of every 1000 in original data set
        with open('biological_train_data1.pkl', 'rb') as f1:
            tr_data1 = pickle.load(f1)                       #Retrieve original data set
            
        with open('biological_train_data2.pkl', 'rb') as f2:
            tr_data2 = pickle.load(f2)
            
        with open('biological_train_data3.pkl', 'rb') as f3:
            tr_data3 = pickle.load(f3)
            
        with open('biological_train_data4.pkl', 'rb') as f4:
            tr_data4 = pickle.load(f4)
            
        with open('New5000_biological_train_dataset1.pkl', 'rb') as f5:
            tr_data5 = pickle.load(f5)
            
        with open('New5000_biological_train_dataset2.pkl', 'rb') as f6:
            tr_data6 = pickle.load(f6)
            
        with open('New5000_biological_train_dataset3.pkl', 'rb') as f7:
            tr_data7 = pickle.load(f7)
            
        with open('New5000_biological_train_dataset4.pkl', 'rb') as f8:
            tr_data8 = pickle.load(f8)
        
        batch_data = []
        for i in range(tr_data1.shape[0]):
            for row in np.array(tr_data1[i, :]):
                batch_data.append(row)
                
        for i in range(tr_data2.shape[0]):
            for row in np.array(tr_data2[i, :]):
                batch_data.append(row)
                
        for i in range(tr_data3.shape[0]):
            for row in np.array(tr_data3[i, :]):
                batch_data.append(row)
                
        for i in range(tr_data4.shape[0]):
            for row in np.array(tr_data4[i, :]):
                batch_data.append(row)
                
        for i in range(tr_data5.shape[0]):
            for row in np.array(tr_data5[i, :]):
                batch_data.append(row)
                
        for i in range(tr_data6.shape[0]):
            for row in np.array(tr_data6[i, :]):
                batch_data.append(row)
                
        for i in range(tr_data7.shape[0]):
            for row in np.array(tr_data7[i, :]):
                batch_data.append(row)
                
        for i in range(tr_data8.shape[0]):
            for row in np.array(tr_data8[i, :]):
                batch_data.append(row)        
                
                
        batch_data = np.array(batch_data).reshape((int(len(batch_data)/(total_input+R)),(total_input+R)))
        print("(Train_data size: {})".format(batch_data.shape[0]))
        self.inputs = np.array(batch_data[:, :total_input])                    #Slice data for inputs
        self.outputs = np.array(batch_data[:, total_input:(total_input+R)])    #Slice data for outputs
        return
        


class multisvr:
    
    def __init__(self):
        return

    def Fit(self,name,delay, X_train_scaled, Y_train_scaled,clfs,i,k):
        print('\nThread ',name,' started')
        #self.n = Y_train_scaled.shape[1]                           #Output dimensions
        for j in range(i,k):
            #tm.sleep(delay)
            clf = svm.SVR(kernel = 'rbf')#, epsilon=0.01)   
            clf.fit(X_train_scaled, Y_train_scaled[:,j])                  #Fit each feature of output separately
            clfs[j] = clf
            print("\tFitting SVR for output feature -> {}\t{:.2f}s".format(j+1, tm.time()-st))
            sv_len = len(clf.support_vectors_)
            print("The number of support vectors are\t",sv_len)
 
        print('\nThread ',name,' stopped')
        
        

def Save_model1(clfs):
      with open('model_with_thread_without_scaling.pickle', 'wb') as fc:              #Save the model
         pickle.dump(clfs, fc)
      return

def Save_model2(Y_train_scaled):       
      n = Y_train_scaled.shape[1]
      with open('op_dim_with_thread_without_scaling.pickle', 'wb') as fc:             #Save the output dimensions
         pickle.dump(n, fc)
      return
    
Q = 134
R1 = 107
R =107
num_of_past_input = 19
total_input = (Q+R1)*num_of_past_input + Q
number_of_sample = 40000

st = tm.time()                                              #Save starting time
print("Action\t\t\t\t\tTime")
print("------\t\t\t\t\t----")
print("Loading training dataset\t\t\t{:.2f}s".format(tm.time()-st))

tr = heli_data()                                          #Training data of specified size
tr.Create_batch()                                           #Create the batch

print("Finished loading training dataset\t\t\t{:.2f}s".format(tm.time()-st))

X_train = tr.inputs                                         #Assign inputs
Y_train = tr.outputs                                        #Assign outputs

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler.fit(Y_train)
Y_train_scaled = scaler.transform(Y_train)

#X_train_scaled, Y_train_scaled = normalize_train_data(X_train, Y_train)
print("Finished scaling of training dataset\t\t\t{:.2f}s".format(tm.time()-st))

#write_data(X_train_scaled, Y_train_scaled)                  #Write training data into excel file
#print("Finished writing training dataset\t\t\t{:.2f}s".format(tm.time()-st))

r1 = multisvr()                                             #Initialize an object

print("Training process begins\t\t\t{:.2f}s".format(tm.time()-st))
print("\n\t----------------------------------------------\n\tProcess details\n\t----------------------------------------------")


threads = [None] * 5
clfs    = [None] * R

l=0
for i in range(len(threads)-1):
    Thread_name = 'Thread'+str(i+1)
    threads[i] = Thread(target=r1.Fit, args=(Thread_name,2,X_train, Y_train,clfs,l,l+25))
    l = l + 25
    threads[i].start()
    
Thread_name = 'Thread'+str(5)    
threads[4] = Thread(target=r1.Fit, args=(Thread_name,2,X_train, Y_train,clfs,l,l+7))
threads[4].start()    
    

for i in range(len(threads)):
    threads[i].join()


print("\t----------------------------------------------\n")
print("Training process ends\t\t\t{:.2f}s".format(tm.time()-st))

#r1.Save_model(clfs)                                                 #Save model after training
Save_model1(clfs) 


Save_model2(Y_train) 
print("Saved the model\t\t\t\t{:.2f}s".format(tm.time()-st))
