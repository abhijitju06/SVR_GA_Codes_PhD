"""
Testing MIMO multisvr model
"""
from sklearn.preprocessing import MinMaxScaler
from math import exp                                    #exponent function
from math import sqrt  
from sklearn import svm                                 #Support Vector Machine module
import numpy as np                                      #Numpy -> for array
import xlsxwriter                                       #Excel read/write operations
import time as tm                                       #Time module
import pickle                                           #Data pickling
import matplotlib.pyplot as plt
class heli_data:                                        #Class for helicopter data set handling
    def __init__(self):                                   
        return
    
    def Create_batch_test(self):                             #Creates a data set with random k samples of every 1000 in original data set
        with open('biological_test_data1.pkl', 'rb') as f:
            te_data = pickle.load(f)                       #Retrieve original data set
        
        batch_data = []
        for i in range(te_data.shape[0]):
            for row in np.array(te_data[i, :]):
                batch_data.append(row)

        batch_data = np.array(batch_data).reshape((int(len(batch_data)/(total_input+R)),(total_input+R)))
        print("(Test_data size: {})".format(batch_data.shape[0]))
        self.inputs = np.array(batch_data[:, :total_input])                    #Slice data for inputs
        self.outputs = np.array(batch_data[:, total_input:(total_input+R)])                  #Slice data for outputs
        return
    
    def Create_batch_train(self):                             #Creates a data set with random k samples of every 1000 in original data set
        with open('biological_train_data1.pkl', 'rb') as f:
            tr_data = pickle.load(f)                       #Retrieve original data set
        
        batch_data = []
        for i in range(tr_data.shape[0]):
            for row in np.array(tr_data[i, :]):
                batch_data.append(row)
        batch_data = np.array(batch_data).reshape((int(len(batch_data)/(total_input+R)),(total_input+R)))
        print("(Train_data size: {})".format(batch_data.shape[0]))
        self.inputs = np.array(batch_data[:, :total_input])                    #Slice data for inputs
        self.outputs = np.array(batch_data[:, total_input:(total_input+R)])                  #Slice data for outputs
        return
        
def write_data(inputs, outputs):                                    #Write the batch data for testing into excel file
    workbook = xlsxwriter.Workbook('batch_data_test.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'Inputs')
    for row, data in enumerate(inputs):
        for col, ele in enumerate(data):
            worksheet.write_column(row+1, col, [ele])
    worksheet.write(0, (total_input+1), 'Outputs')
    for row, data in enumerate(outputs):
        for col, ele in enumerate(data):
            worksheet.write_column(row+1, col+(total_input+1), [ele])
    workbook.close()
    return

def write_result(X_test, Y_pred, Y_test, err1):                     #Write the output after testing into excel file
    workbook = xlsxwriter.Workbook('output_multisvr_scaled_min_max.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'Input')
    for row, data in enumerate(X_test):
        for col, ele in enumerate(data):
            worksheet.write_column(row+1, col, [ele])
    worksheet.write(0, (total_input+1), 'Predicted')
    for row, data in enumerate(Y_pred):
        for col, ele in enumerate(data):
            worksheet.write_column(row+1, col+(total_input+1), [ele])
    worksheet.write(0, (total_input+Q), 'Expected')
    for row, data in enumerate(Y_test):
        for col, ele in enumerate(data):
            worksheet.write_column(row+1, col+(total_input+Q), [ele])
    worksheet.write(0, (total_input+1+R+1+R+1), 'Error')
    for row, data in enumerate(err1):
        worksheet.write_column(row+1, (total_input+1+R+1+R+1), [data])
    workbook.close()

def plot_result(X, Y_, Y, err1):
    plt.title('SVM Model')
    for i in range(Y.shape[1]):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.title('Feature: {}'.format(i+1))
        plt.plot(X[:,i], Y[:, i], color='green', label='Expected')
        plt.plot(X[:,i], Y_[:, i], color='blue', label='Predicted')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend(loc='best')
    plt.show()
    plt.title('Errors')
    plt.plot(np.arange(1, len(err1)+1), err1, color='red', label='Errors per sample')
    plt.legend(loc='best')
    plt.xlabel('Sample number')
    plt.ylabel('Error')
    plt.show()
    return

class multisvr:
    
    def __init__(self):
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


def normalize_test_data_input(X_test,stdv_of_train_input,mean_of_train_input):
    row = number_of_sample
    
    ### START SCALING OF TRAIN INPUT
    X_test_scaled = np.zeros((row,total_input)) 
    for c in range(0,total_input):
        for r in range(0,row):
            X_test_scaled[r,c] = (X_test[r,c] - mean_of_train_input[0,c])/stdv_of_train_input[0,c]
        
    return  X_test_scaled

def normalize_test_data_output(Y_test,stdv_of_train_output,mean_of_train_output):
    row = number_of_sample
    
    ### START SCALING OF TRAIN INPUT
    Y_test_scaled = np.zeros((row,R)) 
    for c in range(0,R):
        for r in range(0,row):
            Y_test_scaled[r,c] = (Y_test[r,c] - mean_of_train_output[0,c])/stdv_of_train_output[0,c]
        
    return  Y_test_scaled


def find_std_mean_of_train_data(X_train, Y_train):
    stdv_of_train_input = np.zeros((1,total_input))
    mean_of_train_input = np.zeros((1,total_input))
    for col in range(0,total_input):
        stdv_of_train_input[0,col] =  np.std(X_train[:,col])
        mean_of_train_input[0,col] = X_train[:,col].mean()
    

    
    stdv_of_train_output = np.zeros((1,R))
    mean_of_train_output = np.zeros((1,R))
    for col in range(0,R):
        stdv_of_train_output[0,col] =  np.std(Y_train[:,col])
        mean_of_train_output[0,col] = Y_train[:,col].mean()
        

    return stdv_of_train_input, mean_of_train_input, stdv_of_train_output, mean_of_train_output

def denormalize_predicted_output(Y_pred,stdv_of_train_output,mean_of_train_output):
    row = number_of_sample
    
    Y_pred_reverse_normalised= np.zeros((row,R)) 
    for c in range(0,R):
       for r in range(0,row):
           Y_pred_reverse_normalised[r,c] = mean_of_train_output[0,c]+ Y_pred[r,c]*stdv_of_train_output[0,c]

    return Y_pred_reverse_normalised


Q = 134
R = 107
R1 = 107
num_of_past_input = 19
total_input = (Q+R1)*num_of_past_input + Q
number_of_sample = 5000

st = tm.time() 
print("Action\t\t\t\t\tTime")   #Save starting time
print("------\t\t\t\t\t----")
#print("Loading training dataset\t\t\t{:.2f}s".format(tm.time()-st))
#tr = heli_data()                                          #Training data of specified size
#tr.Create_batch_train()                                           #Create the batch
#print("training batch dataset created\t\t\t{:.2f}s".format(tm.time()-st))
#X_train = tr.inputs                                         #Assign inputs
#Y_train = tr.outputs                                         #Assign outputs
#                                    
#stdv_of_train_input, mean_of_train_input, stdv_of_train_output, mean_of_train_output = find_std_mean_of_train_data(X_train, Y_train)
                                                 
print("Loading testing dataset\t\t\t{:.2f}s".format(tm.time()-st))

te = heli_data()                                                #Testing data of specified size
te.Create_batch_test()                                                #Create batch   
print("Finished loading testing dataset\t{:.2f}s".format(tm.time()-st))
#scaler = MinMaxScaler()
X_test = te.inputs    #Assign inputs
#scaler.fit(X_train)
#X_test_scaled = scaler.transform(X_test)
#X_test_scaled = normalize_test_data_input(X_test, stdv_of_train_input, mean_of_train_input)

                                           
Y_test = te.outputs
#scaler.fit(Y_train)
#Y_test_scaled = scaler.transform(Y_test)                                              #Assign outputs
#Y_test_scaled = normalize_test_data_output(Y_test,stdv_of_train_output, mean_of_train_output)
#print("Finished normalization of testing dataset\t{:.2f}s".format(tm.time()-st))

#write_data(X_test_scaled, Y_test_scaled)                                       #Write testing data into excel file

r1 = multisvr()                                                 #Initialize an object
r1.Load_model()                                                 #Load model before testing
print("Loaded the model\t\t\t{:.2f}s".format(tm.time()-st))
print("Testing process begins\t\t\t{:.2f}s".format(tm.time()-st))

Y_pred = r1.Predict(X_test)      
#Y_pred_reverse_normalised = scaler.inverse_transform(Y_pred)
#Y_pred_reverse_normalised = denormalize_predicted_output(Y_pred,stdv_of_train_output,mean_of_train_output)




corr = 0
tot = 0

#for pi, ti in zip(Y_pred, Y_test_scaled): 
for pi, ti in zip(Y_pred, Y_test):     
    err = (sum([(yi-di)**2 for yi,di in zip(pi, ti)])/107)                #MSE
    if(err < 0.05):                                                     #Threshold
        corr +=1
    tot += 1

print("Testing process ends\t\t\t{:.2f}s".format(tm.time()-st))
#err1 =[(sum([(yi-di)**2 for yi,di in zip(y1, d1)])/107) for y1,d1 in zip(Y_pred, Y_test_scaled)]
err1 =[(sum([(yi-di)**2 for yi,di in zip(y1, d1)])/107) for y1,d1 in zip(Y_pred, Y_test)]

print("\t-------------------------")
print("\t|Accuracy: {:.2f}% \t|\t{:.2f}s".format(((corr*100)/tot), tm.time()-st))
print("\t|Min Mean Squared Error(MSE): {:.5f} \t|".format(min(err1)))
print("\t|Max Mean Squared Error(MSE): {:.5f} \t|".format(max(err1)))
print("\t|Avg Mean Squared Error(MSE): {:.5f} \t|".format(sum(err1)/len(err1)))
print("\t-------------------------")
#write_result(X_test_scaled, Y_pred, Y_test_scaled, err1)
#plot_result(X_test_scaled, Y_pred, Y_test_scaled, err1)
#print("Results written to excel\t\t{:.2f}s".format(tm.time()-st))

#write_result(X_test, Y_pred_reverse_normalised, Y_test, err1)
#plot_result(X_test_scaled, Y_pred_reverse_normalised, Y_test, err1)
#print("Results written to excel\t\t{:.2f}s".format(tm.time()-st))
