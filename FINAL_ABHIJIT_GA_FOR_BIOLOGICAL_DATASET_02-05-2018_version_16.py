# -*- coding: utf-8 -*-
"""
Created on Wed May 02 13:07:41 2018

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:55:07 2017

@author: Abhijit
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 12:25:28 2017

@author: user
"""
from math import sin
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
from plant_on_biological_dataset import plant
from Model.biological_model import model
from Model.biological_model import biological_model_only
#from math import sin
#from math import exp   
import pickle
import time as tm 
from sklearn import svm 
import math
import xlsxwriter
import sys


import warnings
warnings.filterwarnings("ignore")
#
# Global variables
# Setup optimal string and GA input variables.
#

class heli_data:                                        #Class for helicopter data set handling
    def __init__(self):                                   
        return
    
    def Create_batch_train(self):                             #Creates a data set with random k samples of every 1000 in original data set
        with open('biological_train_data1.pkl', 'rb') as f1:
            tr_data1 = pickle.load(f1)  # Retrieve original data set

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

        batch_data = np.array(batch_data).reshape((int(len(batch_data) / (total_input + R)), (total_input + R)))
        print("(Train_data size: {})".format(batch_data.shape[0]))
        self.inputs = np.array(batch_data[:, :total_input])  # Slice data for inputs
        self.outputs = np.array(batch_data[:, total_input:(total_input + R)])  # Slice data for outputs
        return
    
    def Create_initial_u_candidate(self):                             #Creates a data set with random k samples of every 1000 in original data set
        with open('U_Candidate_for_Biological_dataset_13-05-2018_4.pickle', 'rb') as f:
            tr_data1 = pickle.load(f)                       #Retrieve original data set
        
        batch_data1 = []
        for i in range(tr_data1.shape[0]):
            for row in np.array(tr_data1[i, :]):
                batch_data1.append(row)
        batch_data1 = np.array(batch_data1).reshape((int(len(batch_data1)/num_of_input_to_SVM),num_of_input_to_SVM))
        #print("(Train_data size: {})".format(batch_data.shape[0]))
        self.inputs = np.array(batch_data1[:, :num_of_input_to_SVM])                    #Slice data for inputs
        return




def write_data(outputs,file_name):                                    #Write the batch data for testing into excel file
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'Outputs')
    for row, data in enumerate(outputs):
        for col, ele in enumerate(data):
            worksheet.write_column(row+1, col, [ele])
            
    workbook.close()
    return





def plant_output_prediction(U_Candidate,X_train, Y_train):

    Q = 134                     # Number of input of the Biological MIMO system
    R = 107                     # Number of output of the Biological MIMO system

    col_m=0
    U_for_plant = np.zeros((1,Q))
    for col_p in range(0,Q):
        U_for_plant[0,col_p]= U_Candidate[0,col_m]
        col_m=col_m+20
    
    # CONVERT U_for_plant INTO ROUND OFF TO 3 DECIMAL 
    U_for_plant_round = np.zeros((1,Q))
    for col_p in range(0,Q):
        U_for_plant_round[0,col_p]= round(U_for_plant[0,col_p],3)

    pl = plant()

    U_for_plant_round = U_for_plant_round[0]
    y_actual = pl.Predict(U_for_plant_round)
    y_actual_transpose = np.zeros((1,R))
    for i in range(0,R):
       y_actual_transpose[0,i] = y_actual[i]
       
    y_actual = y_actual_transpose 
    
    return y_actual


def plant_output_check_for_NaN(U_Candidate,X_train, Y_train):

    Q = 134                     # Number of input of the Biological MIMO system

    col_m=0
    U_for_plant = np.zeros((1,Q))
    for col_p in range(0,Q):
        U_for_plant[0,col_p]= U_Candidate[0,col_m]
        col_m=col_m+20
    
    # CONVERT U_for_plant INTO ROUND OFF TO 3 DECIMAL 
    U_for_plant_round = np.zeros((1,Q))
    for col_p in range(0,Q):
        U_for_plant_round[0,col_p]= round(U_for_plant[0,col_p],3)    
    
    pl = plant()
    U_for_plant_round = U_for_plant_round[0]
    y_actual = pl.Predict(U_for_plant_round)
       
    return y_actual




def SVM_model_predict_output(U_Candidate):

    R = 107                     # Number of output of the Biological MIMO system

    mo1 = model()
    y_pred = np.zeros((1,R)) 
    y_pred = mo1.Predict(U_Candidate,clfs,op_dim)[0]
    y_pred_transpose = np.zeros((1,R))
    for i in range(0,R):
       y_pred_transpose[0,i] = y_pred[i]

    return y_pred_transpose



def biological_model_load():

    mo = biological_model_only()
    return mo.clfs, mo.n




def random_population(iter, u_optimal_2):
    """
    Return a list of POP_SIZE individuals, each randomly generated via iterating
    DNA_SIZE times to generate a string of random characters with random_char().
    """    

    DNA_SIZE = 134
    POP_SIZE    = 20
    Q = 134                     # Number of input of the Biological MIMO system
    R = 107                     # Number of output of the Biological MIMO system
    fixed = 0.01                # fIXED VALUE 0.01 CONSIDERED FOR NEGATIVE VALUE UNDER ROOT AND 0 IN DINOMINATOR

    max_concentration = 1.0
    min_concentration = 0.01   
    pop = np.zeros((POP_SIZE, Q))


    if(iter == 0):
        for i in range(POP_SIZE):
            
            
            for j in range(0,R):
                pop[i][j] = random.uniform(min_concentration, max_concentration)
            
            constant_terms_in_u18_design = ((KS7*pop[i][43])/60 + (KS29*pop[i][43])/60 + KS13/(60*(F_S15*pop[i][43] + 1)*(F_S14*pop[i][58] + 1)) + (KS9*pop[i][47])/(60*(F_S8*pop[i][43] + 1)*(F_S9*pop[i][46] + 1)) - (KS36*(kbg16 - lg16 + kg16*pop[i][48]))/(60*FG6*kg16*pop[i][48]) + (KS8*pop[i][41]*pop[i][42]*pop[i][43]*pop[i][46]*pop[i][49])/(60*(F_S6*pop[i][56] + 1)*(F_S7*pop[i][57] + 1)))
            pop[i][107] = np.real(-(60*((KS31*(kbg16 - lg16 + kg16*pop[i][48]))/(60*FG6*kg16*pop[i][48]) - (KS8*pop[i][41]*pop[i][42]*pop[i][43]*pop[i][46]*pop[i][49])/(60*(F_S6*pop[i][56] + 1)*(F_S7*pop[i][57] + 1))))/KS1)
            pop[i][108] = np.real(((kg28*pop[i][45]*pop[i][64]*pop[i][66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/((kbg28 - lg28)*(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60)))) - 1)/FG13)
            pop[i][109] = np.real(-(60*((KS29*pop[i][43])/60 + (KS19*pop[i][45]*pop[i][64]*pop[i][65])/(60*(F_S21*pop[i][50] + 1)*(F_S20*pop[i][66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))) - 1))))/KS54)
            pop[i][110] = np.real(-((KS49*((KS33*((KS29*pop[i][43])/60 + (KS19*pop[i][45]*pop[i][64]*pop[i][65])/(60*(F_S21*pop[i][50] + 1)*(F_S20*pop[i][66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))) - 1))))/KS54 - (KS32*((kg28*pop[i][45]*pop[i][64]*pop[i][66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/((kbg28 - lg28)*(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60)))) - 1))/(60*FG13) - (KS2*pop[i][44]*pop[i][51])/60 + (KS17*pop[i][42])/(60*(F_S17*pop[i][44] + 1)*(F_S18*pop[i][45] + 1)) + (KS8*pop[i][41]*pop[i][42]*pop[i][43]*pop[i][46]*pop[i][49])/(60*(F_S6*pop[i][56] + 1)*(F_S7*pop[i][57] + 1))))/KS35 - (KS2*pop[i][44]*pop[i][51])/60)/(KS48/60 - (KS34*KS49)/(60*KS35)))
            pop[i][111] = np.real((60*((KS2*pop[i][44]*pop[i][51])/60 + (KS48*((KS49*((KS33*((KS29*pop[i][43])/60 + (KS19*pop[i][45]*pop[i][64]*pop[i][65])/(60*(F_S21*pop[i][50] + 1)*(F_S20*pop[i][66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))) - 1))))/KS54 - (KS32*((kg28*pop[i][45]*pop[i][64]*pop[i][66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/((kbg28 - lg28)*(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60)))) - 1))/(60*FG13) - (KS2*pop[i][44]*pop[i][51])/60 + (KS17*pop[i][42])/(60*(F_S17*pop[i][44] + 1)*(F_S18*pop[i][45] + 1)) + (KS8*pop[i][41]*pop[i][42]*pop[i][43]*pop[i][46]*pop[i][49])/(60*(F_S6*pop[i][56] + 1)*(F_S7*pop[i][57] + 1))))/KS35 - (KS2*pop[i][44]*pop[i][51])/60))/(60*(KS48/60 - (KS34*KS49)/(60*KS35)))))/KS49)
            pop[i][112] = np.real((kbg16 - lg16 + kg16*pop[i][48])/(FG6*kg16*pop[i][48]))
            pop[i][113] = np.real((60*((KS2*pop[i][44]*pop[i][51])/60 - (KS4*pop[i][50])/(60*(F_S3*pop[i][55] + 1)) + (KS18*pop[i][44])/(60*(F_S19*pop[i][45] + 1)) + (KS5*pop[i][44]*pop[i][68])/(60*(F_S4*pop[i][55] + 1)) + (KS14*pop[i][44]*pop[i][48])/(60*(F_S16*pop[i][55] + 1)) + (KS17*pop[i][42])/(60*(F_S17*pop[i][44] + 1)*(F_S18*pop[i][45] + 1)) - (KS37*(kbg16 - lg16 + kg16*pop[i][48]))/(60*FG6*kg16*pop[i][48]) - (KS52*(kbg16 - lg16 + kg16*pop[i][48]))/(60*FG6*kg16*pop[i][48]) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/(60*((60*(KS24/(60*F_S28*pop[i][53] + 60) - (KS28*pop[i][30]*pop[i][69])/60 + KS23/((F_S26*pop[i][21] + 1)*(60*F_S25*pop[i][20] + 60)*(F_S27*pop[i][53] + 1)) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/60))/(KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54]) + 1))))/KS38)
            pop[i][114] = np.real(-((K9*pop[i][78]*pop[i][5])/((F23*pop[i][3] + 1)*(3600*km10 + 3600*pop[i][5])) - (K8*pop[i][77]*pop[i][3]*pop[i][30]*(F20*pop[i][3] + 1))/(3600*km9 + 3600*pop[i][3]*pop[i][30]))/((F22*K9*pop[i][78]*pop[i][5])/((F23*pop[i][3] + 1)*(3600*km10 + 3600*pop[i][5])) + (F21*K8*pop[i][77]*pop[i][3]*pop[i][30]*(F20*pop[i][3] + 1))/(3600*km9 + 3600*pop[i][3]*pop[i][30])))
            pop[i][115] = np.real((60*(KS24/(60*F_S28*pop[i][53] + 60) - (KS28*pop[i][30]*pop[i][69])/60 + KS23/((F_S26*pop[i][21] + 1)*(60*F_S25*pop[i][20] + 60)*(F_S27*pop[i][53] + 1)) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/60))/(F_S5*KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54]))
            pop[i][116] = np.real(-(kbg23 - lg23)/kg23)
            pop[i][117] = np.real((60*((KS47*((KS49*((KS33*((KS29*pop[i][43])/60 + (KS19*pop[i][45]*pop[i][64]*pop[i][65])/(60*(F_S21*pop[i][50] + 1)*(F_S20*pop[i][66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))) - 1))))/KS54 - (KS32*((kg28*pop[i][45]*pop[i][64]*pop[i][66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/((kbg28 - lg28)*(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60)))) - 1))/(60*FG13) - (KS2*pop[i][44]*pop[i][51])/60 + (KS17*pop[i][42])/(60*(F_S17*pop[i][44] + 1)*(F_S18*pop[i][45] + 1)) + (KS8*pop[i][41]*pop[i][42]*pop[i][43]*pop[i][46]*pop[i][49])/(60*(F_S6*pop[i][56] + 1)*(F_S7*pop[i][57] + 1))))/KS35 - (KS2*pop[i][44]*pop[i][51])/60))/(60*(KS48/60 - (KS34*KS49)/(60*KS35))) - KS13/(60*(F_S15*pop[i][43] + 1)*(F_S14*pop[i][58] + 1)) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/(60*((60*(KS24/(60*F_S28*pop[i][53] + 60) - (KS28*pop[i][30]*pop[i][69])/60 + KS23/((F_S26*pop[i][21] + 1)*(60*F_S25*pop[i][20] + 60)*(F_S27*pop[i][53] + 1)) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/60))/(KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54]) + 1))))/KS46)
            pop[i][118] = np.real((60*((KS22*pop[i][64])/60 + (KS25*pop[i][64])/60 - (KS20*pop[i][48]*pop[i][58])/(60*((60*F_S11*((KS2*pop[i][44]*pop[i][51])/60 - (KS4*pop[i][50])/(60*(F_S3*pop[i][55] + 1)) + (KS18*pop[i][44])/(60*(F_S19*pop[i][45] + 1)) + (KS5*pop[i][44]*pop[i][68])/(60*(F_S4*pop[i][55] + 1)) + (KS14*pop[i][44]*pop[i][48])/(60*(F_S16*pop[i][55] + 1)) + (KS17*pop[i][42])/(60*(F_S17*pop[i][44] + 1)*(F_S18*pop[i][45] + 1)) - (KS37*(kbg16 - lg16 + kg16*pop[i][48]))/(60*FG6*kg16*pop[i][48]) - (KS52*(kbg16 - lg16 + kg16*pop[i][48]))/(60*FG6*kg16*pop[i][48]) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/(60*((60*(KS24/(60*F_S28*pop[i][53] + 60) - (KS28*pop[i][30]*pop[i][69])/60 + KS23/((F_S26*pop[i][21] + 1)*(60*F_S25*pop[i][20] + 60)*(F_S27*pop[i][53] + 1)) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/60))/(KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54]) + 1))))/KS38 + 1)*(F_S23*pop[i][63] + 1)) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/(60*(F_S21*pop[i][50] + 1)*(F_S20*pop[i][66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))) - 1))))/KS50)
            pop[i][119] = np.real(((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*pop[i][50] + 60*F_S13*constant_terms_in_u18_design*pop[i][45]))/(15*(F_S13*pop[i][45] + 1)))**(1/2)/(2*F_S1*constant_terms_in_u18_design))
            pop[i][120] = np.real(((60*KS3*pop[i][52]*(F_S13*pop[i][45] + 1))/(KS11*((30*((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*pop[i][50] + 60*F_S13*constant_terms_in_u18_design*pop[i][45]))/(15*(F_S13*pop[i][45] + 1)))**(1/2))/constant_terms_in_u18_design + 60)) - 1)/F_S2)
            pop[i][121] = np.real((KS10/((KS43*((K9*pop[i][78]*pop[i][5])/((F23*pop[i][3] + 1)*(3600*km10 + 3600*pop[i][5])) - (K8*pop[i][77]*pop[i][3]*pop[i][30]*(F20*pop[i][3] + 1))/(3600*km9 + 3600*pop[i][3]*pop[i][30])))/((F22*K9*pop[i][78]*pop[i][5])/((F23*pop[i][3] + 1)*(3600*km10 + 3600*pop[i][5])) + (F21*K8*pop[i][77]*pop[i][3]*pop[i][30]*(F20*pop[i][3] + 1))/(3600*km9 + 3600*pop[i][3]*pop[i][30])) + (KS4*pop[i][50])/(F_S3*pop[i][55] + 1) + (KS11*pop[i][50])/(F_S13*pop[i][45] + 1) - (KS39*(kbg16 - lg16 + kg16*pop[i][48]))/(FG6*kg16*pop[i][48]) + (KS11*pop[i][50]*((30*((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*pop[i][50] + 60*F_S13*constant_terms_in_u18_design*pop[i][45]))/(15*(F_S13*pop[i][45] + 1)))**(1/2))/constant_terms_in_u18_design + 60))/(60*(((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*pop[i][50] + 60*F_S13*constant_terms_in_u18_design*pop[i][45]))/(15*(F_S13*pop[i][45] + 1)))**(1/2)/(2*constant_terms_in_u18_design) + 1)*(F_S13*pop[i][45] + 1)) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S21*pop[i][50] + 1)*(F_S20*pop[i][66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))) - 1))) - 1)/F_S10)
            pop[i][122] = np.real(-(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60)))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))
            pop[i][123] = np.real(-(km0*((K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(km4 + pop[i][1]) - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1)*((3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))))/((F1*K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (F2*K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))) - 1))/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))))/((F1*K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (F2*K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))) - 1))/(3600*(km2 + pop[i][1])) + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(3600*(km4 + pop[i][1])) + (K5*pop[i][74]*pop[i][1])/(3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1)) - (K5*pop[i][74]*pop[i][3])/(3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1)) - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/(3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)) + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(7200*(F52*pop[i][40] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40])))*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40]))))/(K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1)) + 1))/((km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1))))/(K1*pop[i][70]*(((K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(km4 + pop[i][1]) - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1)*((3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))))/((F1*K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (F2*K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))) - 1))/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))))/((F1*K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (F2*K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))) - 1))/(3600*(km2 + pop[i][1])) + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(3600*(km4 + pop[i][1])) + (K5*pop[i][74]*pop[i][1])/(3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1)) - (K5*pop[i][74]*pop[i][3])/(3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1)) - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/(3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)) + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(7200*(F52*pop[i][40] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40])))*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40]))))/(K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1)) + 1))/((km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)))/(K1*pop[i][70]) + 1)))
            pop[i][124] = np.real((km40*km42*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40])))
            pop[i][125] = np.real((3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))))/((F1*K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (F2*K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))) - 1))/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))))/((F1*K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (F2*K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))) - 1))/(3600*(km2 + pop[i][1])) + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(3600*(km4 + pop[i][1])) + (K5*pop[i][74]*pop[i][1])/(3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1)) - (K5*pop[i][74]*pop[i][3])/(3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1)) - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/(3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)) + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(7200*(F52*pop[i][40] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40])))*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40]))))/(F3*K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1)))
            

            DIV_1 = (3600*(km1 + pop[i][0]))
            DIV_2 = (3600*(km2 + pop[i][1]))
            DIV_5 = (3600*(km4 + pop[i][1]))
            DIV_6 = (3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1))
            DIV_7 = (3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1))
            DIV_8 = (3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1))
            DIV_9 = (12960000*(km43 + pop[i][25])*(km44 + pop[i][25]))
            DIV_11 = (F3*K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))
            DIV_12 = (3600*(km39 + pop[i][22]*pop[i][33])*(F51*pop[i][15] + 1))
            DIV_16 = ((F1*K2*pop[i][71]*pop[i][0])/DIV_1 - (F2*K2*pop[i][71]*pop[i][1])/DIV_2)
            DIV_17 = ((F1*K2*pop[i][71]*pop[i][0])/DIV_1 - (F2*K2*pop[i][71]*pop[i][1])/DIV_2)
            DIV_14 = (F34*K22*pop[i][91]*pop[i][15]*pop[i][30])
            DIV_18 = (3600*(km28 + pop[i][12]*pop[i][30]))
            DIV_19 = (3600*(km30 + pop[i][14]*pop[i][15])*(F39*pop[i][14] + 1)*(F38*pop[i][30] + 1))
            DIV_15 = (2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))

            if(DIV_1 == 0):
                DIV_1 = fixed

            if(DIV_2 == 0):
                DIV_2 = fixed
 
            if(DIV_5 == 0):
                DIV_5 = fixed

            if(DIV_6 == 0):
                DIV_6 = fixed

            if(DIV_7 == 0):
                DIV_7 = fixed

            if(DIV_8 == 0):
                DIV_8 = fixed

            if(DIV_9 == 0):
                DIV_9 = fixed    

            if(DIV_11 == 0):
                DIV_11 = fixed

            if(DIV_12 == 0):
                DIV_12 = fixed

            if(DIV_16 == 0):
                DIV_16 = fixed

            if(DIV_17 == 0):
                DIV_17 = fixed

            if(DIV_14 == 0):
                DIV_14 = fixed

            if(DIV_18 == 0):
                DIV_18 = fixed

            if(DIV_19 == 0):
                DIV_19 = fixed

            if(DIV_15 == 0):
                DIV_15 = fixed

            DIV_3 = ((F1*K2*pop[i][71]*pop[i][0])/DIV_1 - (F2*K2*pop[i][71]*pop[i][1])/DIV_2)

            if(DIV_3 == 0):
                DIV_3 = fixed

            DIV_4 = ((F1*K2*pop[i][71]*pop[i][0])/DIV_1 - (F2*K2*pop[i][71]*pop[i][1])/DIV_2)

            if(DIV_4 == 0):
                DIV_4 = fixed

            ROOT_1 = ((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/DIV_9)

            if(ROOT_1 < 0):
                ROOT_1 = 0.01

            DIV_10 = (7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*(ROOT_1)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/DIV_15)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))

            if(DIV_10 == 0):
                DIV_10 = fixed

            DIV_13 = (3600*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/DIV_1 - (K2*pop[i][71]*pop[i][1])/DIV_2))/DIV_16 - 1))/DIV_1 - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/DIV_1 - (K2*pop[i][71]*pop[i][1])/DIV_2))/DIV_17 - 1))/DIV_2 + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/DIV_5 + (K5*pop[i][74]*pop[i][1])/DIV_6 - (K5*pop[i][74]*pop[i][3])/DIV_7 - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/DIV_8 + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*(ROOT_1)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/DIV_10))/DIV_11 + 1))

            if(DIV_13 == 0):
                DIV_13 = fixed
    
            pop[i][126] = np.real((3600*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/DIV_1 - (K2*pop[i][71]*pop[i][1])/DIV_2))/DIV_3 - 1))/DIV_1 - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/DIV_1 - (K2*pop[i][71]*pop[i][1])/DIV_2))/DIV_4 - 1))/DIV_2 + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/DIV_5 + (K5*pop[i][74]*pop[i][1])/DIV_6 - (K5*pop[i][74]*pop[i][3])/DIV_7 - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/DIV_8 + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*(ROOT_1)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/DIV_10))/DIV_11 + 1)*((K30*pop[i][99]*pop[i][22]*pop[i][33])/DIV_12 - (K22*pop[i][91]*pop[i][15]*pop[i][30])/DIV_13 + (K21*pop[i][90]*pop[i][12]*pop[i][30]*(F33*pop[i][14] + 1))/DIV_18 - (K23*pop[i][92]*pop[i][14]*pop[i][15]*(F36*pop[i][14] + 1)*(F37*pop[i][15] + 1))/DIV_19))/DIV_14)
    
            pop[i][127] = np.real(-((K27*pop[i][96]*pop[i][20])/(3600*(km36 + pop[i][20])) - 3*((K27*pop[i][96]*pop[i][19]*pop[i][31])/(3600*(km35 + pop[i][19]*pop[i][31]))) + (K18*pop[i][87]*pop[i][12]*pop[i][33]*pop[i][34])/(3600*(km25 + pop[i][12]*pop[i][33]*pop[i][34])) - (K20*pop[i][89]*pop[i][14]*pop[i][31]*pop[i][32]*pop[i][38])/(3600*(km27 + pop[i][14]*pop[i][31]*pop[i][32]*pop[i][38])) + (K19*pop[i][88]*pop[i][30]*pop[i][33]*pop[i][36]*pop[i][37])/(3600*(km26 + pop[i][30]*pop[i][33]*pop[i][36]*pop[i][37])) - (K26*pop[i][95]*pop[i][18]*pop[i][33]*pop[i][39])/(3600*(km34 + pop[i][18]*pop[i][33]*pop[i][39])*(F47*pop[i][19] + 1)*(F46*pop[i][30] + 1)) + (K25*pop[i][94]*pop[i][17]*pop[i][33]*(F42*pop[i][17] + 1)*(F43*pop[i][31] + 1))/(3600*(km33 + pop[i][17]*pop[i][33])*(F44*pop[i][30] + 1)*(F45*pop[i][32] + 1)))/((F53*K18*pop[i][87]*pop[i][12]*pop[i][33]*pop[i][34])/(3600*(km25 + pop[i][12]*pop[i][33]*pop[i][34])) - (F55*K26*pop[i][95]*pop[i][18]*pop[i][33]*pop[i][39])/(3600*(km34 + pop[i][18]*pop[i][33]*pop[i][39])*(F47*pop[i][19] + 1)*(F46*pop[i][30] + 1)) + (F54*K25*pop[i][94]*pop[i][17]*pop[i][33]*(F42*pop[i][17] + 1)*(F43*pop[i][31] + 1))/(3600*(km33 + pop[i][17]*pop[i][33])*(F44*pop[i][30] + 1)*(F45*pop[i][32] + 1))))
            pop[i][128] = np.real(-((K24*pop[i][93]*pop[i][17])/(3600*km32 + 3600*pop[i][17]) - (K24*pop[i][93]*pop[i][16])/(3600*km31 + 3600*pop[i][16]) + (K23*pop[i][92]*pop[i][14]*pop[i][15]*(F36*pop[i][14] + 1)*(F37*pop[i][15] + 1))/((3600*km30 + 3600*pop[i][14]*pop[i][15])*(F39*pop[i][14] + 1)*(F38*pop[i][30] + 1)))/((F40*K24*pop[i][93]*pop[i][16])/(3600*km31 + 3600*pop[i][16]) - (F41*K24*pop[i][93]*pop[i][17])/(3600*km32 + 3600*pop[i][17])))
            pop[i][129] = np.real(((3600*km37 + 3600*pop[i][20]*pop[i][37])*((K27*pop[i][96]*pop[i][20])/(3600*km36 + 3600*pop[i][20]) - (K29*pop[i][98]*pop[i][21])/(3600*km38 + 3600*pop[i][21]) - 3*((K27*pop[i][96]*pop[i][19]*pop[i][31])/(3600*km35 + 3600*pop[i][19]*pop[i][31])) + (4*K28*pop[i][97]*pop[i][20]*pop[i][37])/(3600*km37 + 3600*pop[i][20]*pop[i][37]) - (2*K20*pop[i][89]*pop[i][14]*pop[i][31]*pop[i][32]*pop[i][38])/(3600*km27 + 3600*pop[i][14]*pop[i][31]*pop[i][32]*pop[i][38]) + (2*K19*pop[i][88]*pop[i][30]*pop[i][33]*pop[i][36]*pop[i][37])/(3600*km26 + 3600*pop[i][30]*pop[i][33]*pop[i][36]*pop[i][37])))/(4*F48*K28*pop[i][97]*pop[i][20]*pop[i][37]))
            pop[i][130] = np.real(((KS27*pop[i][45])/(60*(F_S30*pop[i][68] + 1)) - (KS26*pop[i][30])/(60*(F_S29*pop[i][69] + 1)) + (KS5*pop[i][44]*pop[i][68])/(60*(F_S4*pop[i][55] + 1)) + (K10*pop[i][79]*pop[i][4])/(3600*km11 + 3600*pop[i][4]) + (K7*pop[i][76]*pop[i][4])/((F19*pop[i][5] + 1)*(3600*km8 + 3600*pop[i][4])) - (K10*pop[i][79]*pop[i][6]*pop[i][7])/(3600*km12 + 3600*pop[i][6]*pop[i][7]) - (K6*pop[i][75]*pop[i][3]*pop[i][30]*(F15*pop[i][5] + 1))/((3600*km7 + 3600*pop[i][3]*pop[i][30])*(F17*pop[i][16] + 1)*(F16*pop[i][30] + 1)))/(KS16/60 + (F18*K7*pop[i][76]*pop[i][4])/((F19*pop[i][5] + 1)*(3600*km8 + 3600*pop[i][4])) + (F14_modified*K6*pop[i][75]*pop[i][3]*pop[i][30]*(F15*pop[i][5] + 1))/((3600*km7 + 3600*pop[i][3]*pop[i][30])*(F17*pop[i][16] + 1)*(F16*pop[i][30] + 1)))) 
            
            
            A = (3600*(km1 + pop[i][0]))
            B = (3600*(km2 + pop[i][1]))
            C = (3600*(km1 + pop[i][0]))
            D = (3600*(km2 + pop[i][1]))
            E = (3600*(km1 + pop[i][0]))
            F = (3600*(km1 + pop[i][0]))
            G = (3600*(km2 + pop[i][1]))
            H = (3600*(km1 + pop[i][0]))

            if(A == 0):
                A = fixed

            if(B == 0):
                B = fixed

            if(C == 0):
                C = fixed

            if(D == 0):
                D = fixed

            if(E == 0):
                E = fixed

            if(F == 0):
                F = fixed

            if(G == 0):
                G = fixed

            if(H == 0):
                H = fixed
            
            I = (3600*(km2 + pop[i][1]))

            if(I == 0):
                I = fixed

            J = (3600*(km2 + pop[i][1]))
            K = (3600*(km4 + pop[i][1]))
            L = (3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1))
            M = (3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1))
            N = (3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1))
            P = ((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I)
            Q = (12960000*(km43 + pop[i][25])*(km44 + pop[i][25]))
            RR = (2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))

            if(J == 0):
                J = fixed

            if(K == 0):
                K = fixed

            if(L == 0):
                L = fixed

            if(M == 0):
                M = fixed

            if(N == 0):
                N = fixed

            if(P == 0):
                P = fixed

            if(Q == 0):
                Q = fixed

            if(RR == 0):
                RR = fixed

            S = ((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I)
            T = ((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I)
            U = (3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1))
            V = (3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1))
            W = (3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1))
            X = (12960000*(km43 + pop[i][25])*(km44 + pop[i][25]))

            if(S == 0):
                S = fixed

            if(T == 0):
                T = fixed

            if(U == 0):
                U = fixed

            if(V == 0):
                V = fixed

            if(W == 0):
                W = fixed

            if(X == 0):
                X = fixed           

            Y = (7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/X)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/RR)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))
            Z = (3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1))
            AA = (3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1))
            BB = (12960000*(km43 + pop[i][25])*(km44 + pop[i][25]))
            CC = (7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/Q)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/RR)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))

            if(Y == 0):
                Y = fixed

            if(Z == 0):
                Z = fixed

            if(AA == 0):
                AA = fixed

            if(BB == 0):
                BB = fixed

            if(CC == 0):
                CC = fixed

            DD = (3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1))
            EE = (3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1))

            if(DD == 0):
                DD = fixed

            if(EE == 0):
                EE = fixed

            FF = (3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1))
            GG = (2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))
            HH = (12960000*(km43 + pop[i][25])*(km44 + pop[i][25]))

            if(FF == 0):
                FF = fixed

            if(GG == 0):
                GG = fixed

            if(HH == 0):
                HH = fixed

            II = (3600*(km30 + pop[i][14]*pop[i][15])*(F39*pop[i][14] + 1)*(F38*pop[i][30] + 1))
            JJ = (F3*K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))
            KK = (3600*(km28 + pop[i][12]*pop[i][30]))
            LL = (F3*K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))

            if(II == 0):
                II = fixed

            if(JJ == 0):
                JJ = fixed

            if(KK == 0):
                KK = fixed

            if(LL == 0):
                LL = fixed

            MM = ((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I)
            NN = (7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/RR)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41])
            OO = ((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I)
            PP = (3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1))
            QQ = (F34*K22*pop[i][91]*pop[i][15]*pop[i][30])
            SS = (3600*(km21 + pop[i][10]))
            TT = (3600*(km22 + pop[i][11]))
            UU = ((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I)
            VV = (3600*(km39 + pop[i][22]*pop[i][33])*(F51*pop[i][15] + 1))
            XXX = (4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])
            YYY = (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))
            ZZZ = (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))
            WWW = (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))
            UUU = (7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/GG)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))

            if(MM == 0):
                MM = fixed

            if(NN == 0):
                NN = fixed

            if(OO == 0):
                OO = fixed

            if(PP == 0):
                PP = fixed

            if(QQ == 0):
                QQ = fixed

            if(SS == 0):
                SS = fixed

            if(TT == 0):
                TT = fixed

            if(UU == 0):
                UU = fixed

            if(VV == 0):
                VV = fixed

            if(XXX == 0):
                XXX = fixed

            if(YYY == 0):
                YYY = fixed

            if(ZZZ == 0):
                ZZZ = fixed

            if(WWW == 0):
                WWW = fixed

            if(UUU == 0):
                UUU = fixed

            CD = ((F1*K2*pop[i][71]*pop[i][0])/C - (F2*K2*pop[i][71]*pop[i][1])/D)

            if(CD == 0):
                CD = fixed   

            VVV = (3600*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I) - 1))/H - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/MM - 1))/I + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(3600*(km4 + pop[i][1])) + (K5*pop[i][74]*pop[i][1])/AA - (K5*pop[i][74]*pop[i][3])/Z - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/DD + YYY/UUU))/JJ + 1))

            if(VVV == 0):
                VVV = fixed


            XXXX = (7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/RR)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))

            if(XXXX == 0):
                XXXX = fixed


            YYYY = (3600*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/UU - 1))/H - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/OO - 1))/I + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(3600*(km4 + pop[i][1])) + (K5*pop[i][74]*pop[i][1])/PP - (K5*pop[i][74]*pop[i][3])/EE - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/(3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)) + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*(XXX/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/(7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/RR)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))))/LL + 1))

            if(YYYY == 0):
                YYYY = fixed

            BBBB = (3600*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/UU - 1))/H - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/UU - 1))/I + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/K + (K5*pop[i][74]*pop[i][1])/PP - (K5*pop[i][74]*pop[i][3])/EE - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/FF + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/XXXX))/LL + 1))

            if(BBBB == 0):
                BBBB = fixed


            AAAA = (K22*pop[i][91]*pop[i][15]*pop[i][30])

            if(AAAA == 0):
                AAAA = fixed

            WWWW = (3600*(km23 + pop[i][11]*pop[i][31])*(F29*pop[i][14] + 1)*(F28*pop[i][30] + 1)*(F32*pop[i][36] + 1)*((3600*F31*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/S - 1))/H - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/T - 1))/I + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/K + (K5*pop[i][74]*pop[i][1])/U - (K5*pop[i][74]*pop[i][3])/V - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/W + WWW/Y))/LL + 1)*((K30*pop[i][99]*pop[i][22]*pop[i][33])/VV - AAAA/VVV + (K21*pop[i][90]*pop[i][12]*pop[i][30]*(F33*pop[i][14] + 1))/KK - (K23*pop[i][92]*pop[i][14]*pop[i][15]*(F36*pop[i][14] + 1)*(F37*pop[i][15] + 1))/II))/QQ + 1)*((K15*pop[i][84]*pop[i][10])/SS - (K15*pop[i][84]*pop[i][11])/TT + (K22*pop[i][91]*pop[i][15]*pop[i][30]*((3600*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/UU - 1))/H - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/UU - 1))/I + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/K + (K5*pop[i][74]*pop[i][1])/PP - (K5*pop[i][74]*pop[i][3])/EE - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/FF + YYY/NN))/LL + 1)*((K30*pop[i][99]*pop[i][22]*pop[i][33])/VV - AAAA/YYYY + (K21*pop[i][90]*pop[i][12]*pop[i][30]*(F33*pop[i][14] + 1))/KK - (K23*pop[i][92]*pop[i][14]*pop[i][15]*(F36*pop[i][14] + 1)*(F37*pop[i][15] + 1))/II))/AAAA + 1))/BBBB))

            if(WWWW == 0):
                WWWW = fixed

            pop[i][131] = np.real(((K16*pop[i][85]*pop[i][11]*pop[i][31]*(F27*pop[i][4] + 1)*((3600*F26*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/A - (K2*pop[i][71]*pop[i][1])/B))/CD - 1))/E - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/F - (K2*pop[i][71]*pop[i][1])/G))/P - 1))/J + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/K + (K5*pop[i][74]*pop[i][1])/L - (K5*pop[i][74]*pop[i][3])/M - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/N + ZZZ/CC))/LL + 1))/WWWW - 1)/F30)
            
                   
            pop[i][132] = np.real(-((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I)/UU)
            pop[i][133] = np.real(-((K13*pop[i][82]*pop[i][9])/(3600*(km18 + pop[i][9])) + (K14*pop[i][83]*pop[i][9])/(1800*(km19 + pop[i][9])) - (K14*pop[i][83]*pop[i][10])/(1800*(km20 + pop[i][10])) - (K15*pop[i][84]*pop[i][10])/(3600*(km21 + pop[i][10])) + (K15*pop[i][84]*pop[i][11])/(3600*(km22 + pop[i][11])) - (K13*pop[i][82]*pop[i][8]*pop[i][31])/(3600*(km17 + pop[i][8]*pop[i][31])))/((F24*K14*pop[i][83]*pop[i][9])/(1800*(km19 + pop[i][9])) - (F25*K14*pop[i][83]*pop[i][10])/(1800*(km20 + pop[i][10]))))
  
         
    else:
        for i in range(POP_SIZE):
                       
            for j in range(0,R):
                pop[i][j] = u_optimal_2[j] + random.uniform(-0.08, 0.08)   # u_optimal_2[0,j] is changed to u_optimal_2[j]
            
	        constant_terms_in_u18_design = ((KS7*pop[i][43])/60 + (KS29*pop[i][43])/60 + KS13/(60*(F_S15*pop[i][43] + 1)*(F_S14*pop[i][58] + 1)) + (KS9*pop[i][47])/(60*(F_S8*pop[i][43] + 1)*(F_S9*pop[i][46] + 1)) - (KS36*(kbg16 - lg16 + kg16*pop[i][48]))/(60*FG6*kg16*pop[i][48]) + (KS8*pop[i][41]*pop[i][42]*pop[i][43]*pop[i][46]*pop[i][49])/(60*(F_S6*pop[i][56] + 1)*(F_S7*pop[i][57] + 1)))
            pop[i][107] = np.real(-(60*((KS31*(kbg16 - lg16 + kg16*pop[i][48]))/(60*FG6*kg16*pop[i][48]) - (KS8*pop[i][41]*pop[i][42]*pop[i][43]*pop[i][46]*pop[i][49])/(60*(F_S6*pop[i][56] + 1)*(F_S7*pop[i][57] + 1))))/KS1)
            pop[i][108] = np.real(((kg28*pop[i][45]*pop[i][64]*pop[i][66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/((kbg28 - lg28)*(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60)))) - 1)/FG13)
            pop[i][109] = np.real(-(60*((KS29*pop[i][43])/60 + (KS19*pop[i][45]*pop[i][64]*pop[i][65])/(60*(F_S21*pop[i][50] + 1)*(F_S20*pop[i][66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))) - 1))))/KS54)
            pop[i][110] = np.real(-((KS49*((KS33*((KS29*pop[i][43])/60 + (KS19*pop[i][45]*pop[i][64]*pop[i][65])/(60*(F_S21*pop[i][50] + 1)*(F_S20*pop[i][66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))) - 1))))/KS54 - (KS32*((kg28*pop[i][45]*pop[i][64]*pop[i][66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/((kbg28 - lg28)*(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60)))) - 1))/(60*FG13) - (KS2*pop[i][44]*pop[i][51])/60 + (KS17*pop[i][42])/(60*(F_S17*pop[i][44] + 1)*(F_S18*pop[i][45] + 1)) + (KS8*pop[i][41]*pop[i][42]*pop[i][43]*pop[i][46]*pop[i][49])/(60*(F_S6*pop[i][56] + 1)*(F_S7*pop[i][57] + 1))))/KS35 - (KS2*pop[i][44]*pop[i][51])/60)/(KS48/60 - (KS34*KS49)/(60*KS35)))
            pop[i][111] = np.real((60*((KS2*pop[i][44]*pop[i][51])/60 + (KS48*((KS49*((KS33*((KS29*pop[i][43])/60 + (KS19*pop[i][45]*pop[i][64]*pop[i][65])/(60*(F_S21*pop[i][50] + 1)*(F_S20*pop[i][66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))) - 1))))/KS54 - (KS32*((kg28*pop[i][45]*pop[i][64]*pop[i][66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/((kbg28 - lg28)*(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60)))) - 1))/(60*FG13) - (KS2*pop[i][44]*pop[i][51])/60 + (KS17*pop[i][42])/(60*(F_S17*pop[i][44] + 1)*(F_S18*pop[i][45] + 1)) + (KS8*pop[i][41]*pop[i][42]*pop[i][43]*pop[i][46]*pop[i][49])/(60*(F_S6*pop[i][56] + 1)*(F_S7*pop[i][57] + 1))))/KS35 - (KS2*pop[i][44]*pop[i][51])/60))/(60*(KS48/60 - (KS34*KS49)/(60*KS35)))))/KS49)
            pop[i][112] = np.real((kbg16 - lg16 + kg16*pop[i][48])/(FG6*kg16*pop[i][48]))
            pop[i][113] = np.real((60*((KS2*pop[i][44]*pop[i][51])/60 - (KS4*pop[i][50])/(60*(F_S3*pop[i][55] + 1)) + (KS18*pop[i][44])/(60*(F_S19*pop[i][45] + 1)) + (KS5*pop[i][44]*pop[i][68])/(60*(F_S4*pop[i][55] + 1)) + (KS14*pop[i][44]*pop[i][48])/(60*(F_S16*pop[i][55] + 1)) + (KS17*pop[i][42])/(60*(F_S17*pop[i][44] + 1)*(F_S18*pop[i][45] + 1)) - (KS37*(kbg16 - lg16 + kg16*pop[i][48]))/(60*FG6*kg16*pop[i][48]) - (KS52*(kbg16 - lg16 + kg16*pop[i][48]))/(60*FG6*kg16*pop[i][48]) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/(60*((60*(KS24/(60*F_S28*pop[i][53] + 60) - (KS28*pop[i][30]*pop[i][69])/60 + KS23/((F_S26*pop[i][21] + 1)*(60*F_S25*pop[i][20] + 60)*(F_S27*pop[i][53] + 1)) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/60))/(KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54]) + 1))))/KS38)
            pop[i][114] = np.real(-((K9*pop[i][78]*pop[i][5])/((F23*pop[i][3] + 1)*(3600*km10 + 3600*pop[i][5])) - (K8*pop[i][77]*pop[i][3]*pop[i][30]*(F20*pop[i][3] + 1))/(3600*km9 + 3600*pop[i][3]*pop[i][30]))/((F22*K9*pop[i][78]*pop[i][5])/((F23*pop[i][3] + 1)*(3600*km10 + 3600*pop[i][5])) + (F21*K8*pop[i][77]*pop[i][3]*pop[i][30]*(F20*pop[i][3] + 1))/(3600*km9 + 3600*pop[i][3]*pop[i][30])))
            pop[i][115] = np.real((60*(KS24/(60*F_S28*pop[i][53] + 60) - (KS28*pop[i][30]*pop[i][69])/60 + KS23/((F_S26*pop[i][21] + 1)*(60*F_S25*pop[i][20] + 60)*(F_S27*pop[i][53] + 1)) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/60))/(F_S5*KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54]))
            pop[i][116] = np.real(-(kbg23 - lg23)/kg23)
            pop[i][117] = np.real((60*((KS47*((KS49*((KS33*((KS29*pop[i][43])/60 + (KS19*pop[i][45]*pop[i][64]*pop[i][65])/(60*(F_S21*pop[i][50] + 1)*(F_S20*pop[i][66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))) - 1))))/KS54 - (KS32*((kg28*pop[i][45]*pop[i][64]*pop[i][66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/((kbg28 - lg28)*(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60)))) - 1))/(60*FG13) - (KS2*pop[i][44]*pop[i][51])/60 + (KS17*pop[i][42])/(60*(F_S17*pop[i][44] + 1)*(F_S18*pop[i][45] + 1)) + (KS8*pop[i][41]*pop[i][42]*pop[i][43]*pop[i][46]*pop[i][49])/(60*(F_S6*pop[i][56] + 1)*(F_S7*pop[i][57] + 1))))/KS35 - (KS2*pop[i][44]*pop[i][51])/60))/(60*(KS48/60 - (KS34*KS49)/(60*KS35))) - KS13/(60*(F_S15*pop[i][43] + 1)*(F_S14*pop[i][58] + 1)) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/(60*((60*(KS24/(60*F_S28*pop[i][53] + 60) - (KS28*pop[i][30]*pop[i][69])/60 + KS23/((F_S26*pop[i][21] + 1)*(60*F_S25*pop[i][20] + 60)*(F_S27*pop[i][53] + 1)) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/60))/(KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54]) + 1))))/KS46)
            pop[i][118] = np.real((60*((KS22*pop[i][64])/60 + (KS25*pop[i][64])/60 - (KS20*pop[i][48]*pop[i][58])/(60*((60*F_S11*((KS2*pop[i][44]*pop[i][51])/60 - (KS4*pop[i][50])/(60*(F_S3*pop[i][55] + 1)) + (KS18*pop[i][44])/(60*(F_S19*pop[i][45] + 1)) + (KS5*pop[i][44]*pop[i][68])/(60*(F_S4*pop[i][55] + 1)) + (KS14*pop[i][44]*pop[i][48])/(60*(F_S16*pop[i][55] + 1)) + (KS17*pop[i][42])/(60*(F_S17*pop[i][44] + 1)*(F_S18*pop[i][45] + 1)) - (KS37*(kbg16 - lg16 + kg16*pop[i][48]))/(60*FG6*kg16*pop[i][48]) - (KS52*(kbg16 - lg16 + kg16*pop[i][48]))/(60*FG6*kg16*pop[i][48]) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/(60*((60*(KS24/(60*F_S28*pop[i][53] + 60) - (KS28*pop[i][30]*pop[i][69])/60 + KS23/((F_S26*pop[i][21] + 1)*(60*F_S25*pop[i][20] + 60)*(F_S27*pop[i][53] + 1)) + (KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54])/60))/(KS6*pop[i][44]*pop[i][45]*pop[i][53]*pop[i][54]) + 1))))/KS38 + 1)*(F_S23*pop[i][63] + 1)) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/(60*(F_S21*pop[i][50] + 1)*(F_S20*pop[i][66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))) - 1))))/KS50)
            pop[i][119] = np.real(((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*pop[i][50] + 60*F_S13*constant_terms_in_u18_design*pop[i][45]))/(15*(F_S13*pop[i][45] + 1)))**(1/2)/(2*F_S1*constant_terms_in_u18_design))
            pop[i][120] = np.real(((60*KS3*pop[i][52]*(F_S13*pop[i][45] + 1))/(KS11*((30*((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*pop[i][50] + 60*F_S13*constant_terms_in_u18_design*pop[i][45]))/(15*(F_S13*pop[i][45] + 1)))**(1/2))/constant_terms_in_u18_design + 60)) - 1)/F_S2)
            pop[i][121] = np.real((KS10/((KS43*((K9*pop[i][78]*pop[i][5])/((F23*pop[i][3] + 1)*(3600*km10 + 3600*pop[i][5])) - (K8*pop[i][77]*pop[i][3]*pop[i][30]*(F20*pop[i][3] + 1))/(3600*km9 + 3600*pop[i][3]*pop[i][30])))/((F22*K9*pop[i][78]*pop[i][5])/((F23*pop[i][3] + 1)*(3600*km10 + 3600*pop[i][5])) + (F21*K8*pop[i][77]*pop[i][3]*pop[i][30]*(F20*pop[i][3] + 1))/(3600*km9 + 3600*pop[i][3]*pop[i][30])) + (KS4*pop[i][50])/(F_S3*pop[i][55] + 1) + (KS11*pop[i][50])/(F_S13*pop[i][45] + 1) - (KS39*(kbg16 - lg16 + kg16*pop[i][48]))/(FG6*kg16*pop[i][48]) + (KS11*pop[i][50]*((30*((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*pop[i][50] + 60*F_S13*constant_terms_in_u18_design*pop[i][45]))/(15*(F_S13*pop[i][45] + 1)))**(1/2))/constant_terms_in_u18_design + 60))/(60*(((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*pop[i][50] + 60*F_S13*constant_terms_in_u18_design*pop[i][45]))/(15*(F_S13*pop[i][45] + 1)))**(1/2)/(2*constant_terms_in_u18_design) + 1)*(F_S13*pop[i][45] + 1)) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S21*pop[i][50] + 1)*(F_S20*pop[i][66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))) - 1))) - 1)/F_S10)
            pop[i][122] = np.real(-(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*pop[i][63] + 60) - (KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60)))/(kg30*pop[i][43] + kg9*pop[i][43]*pop[i][66] + kg27*pop[i][43]*pop[i][66] + kg37*pop[i][41]*pop[i][66] + kg29*pop[i][42]*pop[i][64]*pop[i][66] + kg21*pop[i][41]*pop[i][42]*pop[i][64]*pop[i][66] + (kg20*pop[i][43]*pop[i][45]*pop[i][66])/(FG9*pop[i][58] + 1) + (F_S22*KS19*pop[i][45]*pop[i][64]*pop[i][65])/((F_S20*pop[i][66] + 1)*(60*F_S21*pop[i][50] + 60))))
            pop[i][123] = np.real(-(km0*((K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(km4 + pop[i][1]) - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1)*((3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))))/((F1*K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (F2*K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))) - 1))/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))))/((F1*K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (F2*K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))) - 1))/(3600*(km2 + pop[i][1])) + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(3600*(km4 + pop[i][1])) + (K5*pop[i][74]*pop[i][1])/(3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1)) - (K5*pop[i][74]*pop[i][3])/(3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1)) - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/(3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)) + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(7200*(F52*pop[i][40] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40])))*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40]))))/(K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1)) + 1))/((km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1))))/(K1*pop[i][70]*(((K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(km4 + pop[i][1]) - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1)*((3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))))/((F1*K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (F2*K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))) - 1))/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))))/((F1*K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (F2*K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))) - 1))/(3600*(km2 + pop[i][1])) + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(3600*(km4 + pop[i][1])) + (K5*pop[i][74]*pop[i][1])/(3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1)) - (K5*pop[i][74]*pop[i][3])/(3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1)) - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/(3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)) + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(7200*(F52*pop[i][40] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40])))*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40]))))/(K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1)) + 1))/((km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)))/(K1*pop[i][70]) + 1)))
            pop[i][124] = np.real((km40*km42*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40])))
            pop[i][125] = np.real((3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))))/((F1*K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (F2*K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))) - 1))/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))))/((F1*K2*pop[i][71]*pop[i][0])/(3600*(km1 + pop[i][0])) - (F2*K2*pop[i][71]*pop[i][1])/(3600*(km2 + pop[i][1]))) - 1))/(3600*(km2 + pop[i][1])) + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(3600*(km4 + pop[i][1])) + (K5*pop[i][74]*pop[i][1])/(3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1)) - (K5*pop[i][74]*pop[i][3])/(3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1)) - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/(3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)) + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(7200*(F52*pop[i][40] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][40] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][40] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][40] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][40] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][40])/(12960000*(km43 + pop[i][25])*(km44 + pop[i][25])))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][40]))/(2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40])))*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][40]))))/(F3*K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1)))
            
            DIV_1 = (3600*(km1 + pop[i][0]))
            DIV_2 = (3600*(km2 + pop[i][1]))
            DIV_5 = (3600*(km4 + pop[i][1]))
            DIV_6 = (3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1))
            DIV_7 = (3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1))
            DIV_8 = (3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1))
            DIV_9 = (12960000*(km43 + pop[i][25])*(km44 + pop[i][25]))
            DIV_11 = (F3*K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))
            DIV_12 = (3600*(km39 + pop[i][22]*pop[i][33])*(F51*pop[i][15] + 1))
            DIV_16 = ((F1*K2*pop[i][71]*pop[i][0])/DIV_1 - (F2*K2*pop[i][71]*pop[i][1])/DIV_2)
            DIV_17 = ((F1*K2*pop[i][71]*pop[i][0])/DIV_1 - (F2*K2*pop[i][71]*pop[i][1])/DIV_2)
            DIV_14 = (F34*K22*pop[i][91]*pop[i][15]*pop[i][30])
            DIV_18 = (3600*(km28 + pop[i][12]*pop[i][30]))
            DIV_19 = (3600*(km30 + pop[i][14]*pop[i][15])*(F39*pop[i][14] + 1)*(F38*pop[i][30] + 1))
            DIV_15 = (2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))

            if(DIV_1 == 0):
                DIV_1 = fixed

            if(DIV_2 == 0):
                DIV_2 = fixed

            DIV_16 = ((F1*K2*pop[i][71]*pop[i][0])/DIV_1 - (F2*K2*pop[i][71]*pop[i][1])/DIV_2)
            DIV_17 = ((F1*K2*pop[i][71]*pop[i][0])/DIV_1 - (F2*K2*pop[i][71]*pop[i][1])/DIV_2)
 
            if(DIV_5 == 0):
                DIV_5 = fixed

            if(DIV_6 == 0):
                DIV_6 = fixed

            if(DIV_7 == 0):
                DIV_7 = fixed

            if(DIV_8 == 0):
                DIV_8 = fixed

            if(DIV_9 == 0):
                DIV_9 = fixed    

            if(DIV_11 == 0):
                DIV_11 = fixed

            if(DIV_12 == 0):
                DIV_12 = fixed

            if(DIV_16 == 0):
                DIV_16 = fixed

            if(DIV_17 == 0):
                DIV_17 = fixed

            if(DIV_14 == 0):
                DIV_14 = fixed

            if(DIV_18 == 0):
                DIV_18 = fixed

            if(DIV_19 == 0):
                DIV_19 = fixed

            if(DIV_15 == 0):
                DIV_15 = fixed

            DIV_3 = ((F1*K2*pop[i][71]*pop[i][0])/DIV_1 - (F2*K2*pop[i][71]*pop[i][1])/DIV_2)

            if(DIV_3 == 0):
                DIV_3 = fixed

            DIV_4 = ((F1*K2*pop[i][71]*pop[i][0])/DIV_1 - (F2*K2*pop[i][71]*pop[i][1])/DIV_2)

            if(DIV_4 == 0):
                DIV_4 = fixed

            ROOT_1 = ((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/DIV_9)

            if(ROOT_1 < 0):
                ROOT_1 = fixed

            DIV_10 = (7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*(ROOT_1)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/DIV_15)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))

            if(DIV_10 == 0):
                DIV_10 = fixed

            DIV_13 = (3600*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/DIV_1 - (K2*pop[i][71]*pop[i][1])/DIV_2))/DIV_16 - 1))/DIV_1 - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/DIV_1 - (K2*pop[i][71]*pop[i][1])/DIV_2))/DIV_17 - 1))/DIV_2 + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/DIV_5 + (K5*pop[i][74]*pop[i][1])/DIV_6 - (K5*pop[i][74]*pop[i][3])/DIV_7 - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/DIV_8 + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*(ROOT_1)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/DIV_10))/DIV_11 + 1))

            if(DIV_13 == 0):
                DIV_13 = fixed
    
            pop[i][126] = np.real((3600*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/DIV_1 - (K2*pop[i][71]*pop[i][1])/DIV_2))/DIV_3 - 1))/DIV_1 - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/DIV_1 - (K2*pop[i][71]*pop[i][1])/DIV_2))/DIV_4 - 1))/DIV_2 + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/DIV_5 + (K5*pop[i][74]*pop[i][1])/DIV_6 - (K5*pop[i][74]*pop[i][3])/DIV_7 - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/DIV_8 + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*(ROOT_1)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/DIV_10))/DIV_11 + 1)*((K30*pop[i][99]*pop[i][22]*pop[i][33])/DIV_12 - (K22*pop[i][91]*pop[i][15]*pop[i][30])/DIV_13 + (K21*pop[i][90]*pop[i][12]*pop[i][30]*(F33*pop[i][14] + 1))/DIV_18 - (K23*pop[i][92]*pop[i][14]*pop[i][15]*(F36*pop[i][14] + 1)*(F37*pop[i][15] + 1))/DIV_19))/DIV_14)
              
            
            pop[i][127] = np.real(-((K27*pop[i][96]*pop[i][20])/(3600*(km36 + pop[i][20])) - 3*((K27*pop[i][96]*pop[i][19]*pop[i][31])/(3600*(km35 + pop[i][19]*pop[i][31]))) + (K18*pop[i][87]*pop[i][12]*pop[i][33]*pop[i][34])/(3600*(km25 + pop[i][12]*pop[i][33]*pop[i][34])) - (K20*pop[i][89]*pop[i][14]*pop[i][31]*pop[i][32]*pop[i][38])/(3600*(km27 + pop[i][14]*pop[i][31]*pop[i][32]*pop[i][38])) + (K19*pop[i][88]*pop[i][30]*pop[i][33]*pop[i][36]*pop[i][37])/(3600*(km26 + pop[i][30]*pop[i][33]*pop[i][36]*pop[i][37])) - (K26*pop[i][95]*pop[i][18]*pop[i][33]*pop[i][39])/(3600*(km34 + pop[i][18]*pop[i][33]*pop[i][39])*(F47*pop[i][19] + 1)*(F46*pop[i][30] + 1)) + (K25*pop[i][94]*pop[i][17]*pop[i][33]*(F42*pop[i][17] + 1)*(F43*pop[i][31] + 1))/(3600*(km33 + pop[i][17]*pop[i][33])*(F44*pop[i][30] + 1)*(F45*pop[i][32] + 1)))/((F53*K18*pop[i][87]*pop[i][12]*pop[i][33]*pop[i][34])/(3600*(km25 + pop[i][12]*pop[i][33]*pop[i][34])) - (F55*K26*pop[i][95]*pop[i][18]*pop[i][33]*pop[i][39])/(3600*(km34 + pop[i][18]*pop[i][33]*pop[i][39])*(F47*pop[i][19] + 1)*(F46*pop[i][30] + 1)) + (F54*K25*pop[i][94]*pop[i][17]*pop[i][33]*(F42*pop[i][17] + 1)*(F43*pop[i][31] + 1))/(3600*(km33 + pop[i][17]*pop[i][33])*(F44*pop[i][30] + 1)*(F45*pop[i][32] + 1))))
            pop[i][128] = np.real(-((K24*pop[i][93]*pop[i][17])/(3600*km32 + 3600*pop[i][17]) - (K24*pop[i][93]*pop[i][16])/(3600*km31 + 3600*pop[i][16]) + (K23*pop[i][92]*pop[i][14]*pop[i][15]*(F36*pop[i][14] + 1)*(F37*pop[i][15] + 1))/((3600*km30 + 3600*pop[i][14]*pop[i][15])*(F39*pop[i][14] + 1)*(F38*pop[i][30] + 1)))/((F40*K24*pop[i][93]*pop[i][16])/(3600*km31 + 3600*pop[i][16]) - (F41*K24*pop[i][93]*pop[i][17])/(3600*km32 + 3600*pop[i][17])))
            pop[i][129] = np.real(((3600*km37 + 3600*pop[i][20]*pop[i][37])*((K27*pop[i][96]*pop[i][20])/(3600*km36 + 3600*pop[i][20]) - (K29*pop[i][98]*pop[i][21])/(3600*km38 + 3600*pop[i][21]) - 3*((K27*pop[i][96]*pop[i][19]*pop[i][31])/(3600*km35 + 3600*pop[i][19]*pop[i][31])) + (4*K28*pop[i][97]*pop[i][20]*pop[i][37])/(3600*km37 + 3600*pop[i][20]*pop[i][37]) - (2*K20*pop[i][89]*pop[i][14]*pop[i][31]*pop[i][32]*pop[i][38])/(3600*km27 + 3600*pop[i][14]*pop[i][31]*pop[i][32]*pop[i][38]) + (2*K19*pop[i][88]*pop[i][30]*pop[i][33]*pop[i][36]*pop[i][37])/(3600*km26 + 3600*pop[i][30]*pop[i][33]*pop[i][36]*pop[i][37])))/(4*F48*K28*pop[i][97]*pop[i][20]*pop[i][37]))
            pop[i][130] = np.real(((KS27*pop[i][45])/(60*(F_S30*pop[i][68] + 1)) - (KS26*pop[i][30])/(60*(F_S29*pop[i][69] + 1)) + (KS5*pop[i][44]*pop[i][68])/(60*(F_S4*pop[i][55] + 1)) + (K10*pop[i][79]*pop[i][4])/(3600*km11 + 3600*pop[i][4]) + (K7*pop[i][76]*pop[i][4])/((F19*pop[i][5] + 1)*(3600*km8 + 3600*pop[i][4])) - (K10*pop[i][79]*pop[i][6]*pop[i][7])/(3600*km12 + 3600*pop[i][6]*pop[i][7]) - (K6*pop[i][75]*pop[i][3]*pop[i][30]*(F15*pop[i][5] + 1))/((3600*km7 + 3600*pop[i][3]*pop[i][30])*(F17*pop[i][16] + 1)*(F16*pop[i][30] + 1)))/(KS16/60 + (F18*K7*pop[i][76]*pop[i][4])/((F19*pop[i][5] + 1)*(3600*km8 + 3600*pop[i][4])) + (F14_modified*K6*pop[i][75]*pop[i][3]*pop[i][30]*(F15*pop[i][5] + 1))/((3600*km7 + 3600*pop[i][3]*pop[i][30])*(F17*pop[i][16] + 1)*(F16*pop[i][30] + 1)))) 
            
            
            A = (3600*(km1 + pop[i][0]))
            B = (3600*(km2 + pop[i][1]))
            C = (3600*(km1 + pop[i][0]))
            D = (3600*(km2 + pop[i][1]))
            E = (3600*(km1 + pop[i][0]))
            F = (3600*(km1 + pop[i][0]))
            G = (3600*(km2 + pop[i][1]))
            H = (3600*(km1 + pop[i][0]))

            if(A == 0):
                A = fixed

            if(B == 0):
                B = fixed

            if(C == 0):
                C = fixed

            if(D == 0):
                D = fixed

            if(E == 0):
                E = fixed

            if(F == 0):
                F = fixed

            if(G == 0):
                G = fixed

            if(H == 0):
                H = fixed
            
            I = (3600*(km2 + pop[i][1]))

            if(I == 0):
                I = fixed

            J = (3600*(km2 + pop[i][1]))
            K = (3600*(km4 + pop[i][1]))
            L = (3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1))
            M = (3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1))
            N = (3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1))
            P = ((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I)
            Q = (12960000*(km43 + pop[i][25])*(km44 + pop[i][25]))
            RR = (2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))

            if(J == 0):
                J = fixed

            if(K == 0):
                K = fixed

            if(L == 0):
                L = fixed

            if(M == 0):
                M = fixed

            if(N == 0):
                N = fixed

            if(P == 0):
                P = fixed

            if(Q == 0):
                Q = fixed

            if(RR == 0):
                RR = fixed

            S = ((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I)
            T = ((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I)
            U = (3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1))
            V = (3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1))
            W = (3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1))
            X = (12960000*(km43 + pop[i][25])*(km44 + pop[i][25]))

            if(S == 0):
                S = fixed

            if(T == 0):
                T = fixed

            if(U == 0):
                U = fixed

            if(V == 0):
                V = fixed

            if(W == 0):
                W = fixed

            if(X == 0):
                X = fixed           

            Y = (7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/X)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/RR)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))
            Z = (3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1))
            AA = (3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1))
            BB = (12960000*(km43 + pop[i][25])*(km44 + pop[i][25]))
            CC = (7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/Q)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/RR)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))

            if(Y == 0):
                Y = fixed

            if(Z == 0):
                Z = fixed

            if(AA == 0):
                AA = fixed

            if(BB == 0):
                BB = fixed

            if(CC == 0):
                CC = fixed

            DD = (3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1))
            EE = (3600*(km6 + pop[i][3])*(F11*pop[i][4] + 1)*(F12*pop[i][24] + 1)*(F13*pop[i][29] + 1))

            if(DD == 0):
                DD = fixed

            if(EE == 0):
                EE = fixed

            FF = (3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1))
            GG = (2*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))
            HH = (12960000*(km43 + pop[i][25])*(km44 + pop[i][25]))

            if(FF == 0):
                FF = fixed

            if(GG == 0):
                GG = fixed

            if(HH == 0):
                HH = fixed

            II = (3600*(km30 + pop[i][14]*pop[i][15])*(F39*pop[i][14] + 1)*(F38*pop[i][30] + 1))
            JJ = (F3*K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))
            KK = (3600*(km28 + pop[i][12]*pop[i][30]))
            LL = (F3*K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))

            if(II == 0):
                II = fixed

            if(JJ == 0):
                JJ = fixed

            if(KK == 0):
                KK = fixed

            if(LL == 0):
                LL = fixed

            MM = ((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I)
            NN = (7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH))**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/RR)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41])
            OO = ((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I)
            PP = (3600*(km5 + pop[i][1])*(F8*pop[i][4] + 1)*(F9*pop[i][24] + 1)*(F10*pop[i][29] + 1))
            QQ = (F34*K22*pop[i][91]*pop[i][15]*pop[i][30])
            SS = (3600*(km21 + pop[i][10]))
            TT = (3600*(km22 + pop[i][11]))
            UU = ((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I)
            VV = (3600*(km39 + pop[i][22]*pop[i][33])*(F51*pop[i][15] + 1))
            XXX = (4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])
            YYY = (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))
            ZZZ = (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))
            WWW = (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))
            UUU = (7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/GG)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))

            if(MM == 0):
                MM = fixed

            if(NN == 0):
                NN = fixed

            if(OO == 0):
                OO = fixed

            if(PP == 0):
                PP = fixed

            if(QQ == 0):
                QQ = fixed

            if(SS == 0):
                SS = fixed

            if(TT == 0):
                TT = fixed

            if(UU == 0):
                UU = fixed

            if(VV == 0):
                VV = fixed

            if(XXX == 0):
                XXX = fixed

            if(YYY == 0):
                YYY = fixed

            if(ZZZ == 0):
                ZZZ = fixed

            if(WWW == 0):
                WWW = fixed

            if(UUU == 0):
                UUU = fixed

            CD = ((F1*K2*pop[i][71]*pop[i][0])/C - (F2*K2*pop[i][71]*pop[i][1])/D)

            if(CD == 0):
                CD = fixed   

            VVV = (3600*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/((F1*K2*pop[i][71]*pop[i][0])/H - (F2*K2*pop[i][71]*pop[i][1])/I) - 1))/H - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/MM - 1))/I + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(3600*(km4 + pop[i][1])) + (K5*pop[i][74]*pop[i][1])/AA - (K5*pop[i][74]*pop[i][3])/Z - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/DD + YYY/UUU))/JJ + 1))

            if(VVV == 0):
                VVV = fixed


            XXXX = (7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/RR)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))

            if(XXXX == 0):
                XXXX = fixed


            YYYY = (3600*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/UU - 1))/H - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/OO - 1))/I + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/(3600*(km4 + pop[i][1])) + (K5*pop[i][74]*pop[i][1])/PP - (K5*pop[i][74]*pop[i][3])/EE - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/(3600*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)) + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*(XXX/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/(7200*(F52*pop[i][41] + 1)*(km40 + (km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/RR)*(2*K31*pop[i][100]*km42**2*pop[i][1]**2 + K33*pop[i][102]*km40**2*pop[i][24]**2 + F52*K33*pop[i][102]*km40**2*pop[i][24]**2*pop[i][41]))))/LL + 1))

            if(YYYY == 0):
                YYYY = fixed

            BBBB = (3600*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/UU - 1))/H - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/UU - 1))/I + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/K + (K5*pop[i][74]*pop[i][1])/PP - (K5*pop[i][74]*pop[i][3])/EE - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/FF + (K31*pop[i][100]*km40*km42*pop[i][1]*(2*K31*pop[i][100]*km42*pop[i][1] - 3600*((4*K31**2*pop[i][100]**2*km42**2*pop[i][1]**2*pop[i][25]**2 + K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*km44*pop[i][1]**2 + K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2 + 4*K31**2*pop[i][100]**2*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K31**2*pop[i][100]**2*km42**2*km44*pop[i][1]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25] + K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2 - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2 + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2 - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 8*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25] - 8*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25] + 4*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25] - 4*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25] + F52**2*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + F52**2*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*km44*pop[i][24]**2*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 2*F52*K33**2*pop[i][102]**2*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2 + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*pop[i][1]**2*pop[i][25]**2*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*pop[i][24]**2*pop[i][25]**2*pop[i][41] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25] + 4*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25] + 4*F52**2*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41]**2 - 4*F52**2*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41]**2 + 8*F52*K31*K34*pop[i][100]*pop[i][103]*km42**2*km44*pop[i][1]**2*pop[i][25]*pop[i][41] - 8*F52*K31*K35*pop[i][100]*pop[i][104]*km42**2*km43*pop[i][1]**2*pop[i][25]*pop[i][41] + 8*F52*K33*K34*pop[i][102]*pop[i][103]*km40**2*km44*pop[i][24]**2*pop[i][25]*pop[i][41] - 8*F52*K33*K35*pop[i][102]*pop[i][104]*km40**2*km43*pop[i][24]**2*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*pop[i][1]*pop[i][24]*pop[i][25]**2*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*km44*pop[i][1]*pop[i][24]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km43*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41] + 4*F52*K31*K33*pop[i][100]*pop[i][102]*km40*km42*km44*pop[i][1]*pop[i][24]*pop[i][25]*pop[i][41])/HH)**(1/2) + K33*pop[i][102]*km40*pop[i][24] + F52*K33*pop[i][102]*km40*pop[i][24]*pop[i][41]))/XXXX))/LL + 1))

            if(BBBB == 0):
                BBBB = fixed


            AAAA = (K22*pop[i][91]*pop[i][15]*pop[i][30])

            if(AAAA == 0):
                AAAA = fixed

            WWWW = (3600*(km23 + pop[i][11]*pop[i][31])*(F29*pop[i][14] + 1)*(F28*pop[i][30] + 1)*(F32*pop[i][36] + 1)*((3600*F31*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/S - 1))/H - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/T - 1))/I + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/K + (K5*pop[i][74]*pop[i][1])/U - (K5*pop[i][74]*pop[i][3])/V - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/W + WWW/Y))/LL + 1)*((K30*pop[i][99]*pop[i][22]*pop[i][33])/VV - AAAA/VVV + (K21*pop[i][90]*pop[i][12]*pop[i][30]*(F33*pop[i][14] + 1))/KK - (K23*pop[i][92]*pop[i][14]*pop[i][15]*(F36*pop[i][14] + 1)*(F37*pop[i][15] + 1))/II))/QQ + 1)*((K15*pop[i][84]*pop[i][10])/SS - (K15*pop[i][84]*pop[i][11])/TT + (K22*pop[i][91]*pop[i][15]*pop[i][30]*((3600*(km29 + pop[i][15]*pop[i][30])*((3600*F35*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/UU - 1))/H - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I))/UU - 1))/I + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/K + (K5*pop[i][74]*pop[i][1])/PP - (K5*pop[i][74]*pop[i][3])/EE - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/FF + YYY/NN))/LL + 1)*((K30*pop[i][99]*pop[i][22]*pop[i][33])/VV - AAAA/YYYY + (K21*pop[i][90]*pop[i][12]*pop[i][30]*(F33*pop[i][14] + 1))/KK - (K23*pop[i][92]*pop[i][14]*pop[i][15]*(F36*pop[i][14] + 1)*(F37*pop[i][15] + 1))/II))/AAAA + 1))/BBBB))

            if(WWWW == 0):
                WWWW = fixed

            pop[i][131] = np.real(((K16*pop[i][85]*pop[i][11]*pop[i][31]*(F27*pop[i][4] + 1)*((3600*F26*(km3 + pop[i][2]*pop[i][30])*(F5*pop[i][1] + 1)*(F6*pop[i][14] + 1)*((K2*pop[i][71]*pop[i][0]*((F1*((K2*pop[i][71]*pop[i][0])/A - (K2*pop[i][71]*pop[i][1])/B))/CD - 1))/E - (K2*pop[i][71]*pop[i][1]*((F2*((K2*pop[i][71]*pop[i][0])/F - (K2*pop[i][71]*pop[i][1])/G))/P - 1))/J + (K4*pop[i][73]*pop[i][1]*(F7*pop[i][1] + 1))/K + (K5*pop[i][74]*pop[i][1])/L - (K5*pop[i][74]*pop[i][3])/M - (K3*pop[i][72]*pop[i][2]*pop[i][30]*(F4*pop[i][31] + 1))/N + ZZZ/CC))/LL + 1))/WWWW - 1)/F30)
            
                   
            pop[i][132] = np.real(-((K2*pop[i][71]*pop[i][0])/H - (K2*pop[i][71]*pop[i][1])/I)/UU)
            pop[i][133] = np.real(-((K13*pop[i][82]*pop[i][9])/(3600*(km18 + pop[i][9])) + (K14*pop[i][83]*pop[i][9])/(1800*(km19 + pop[i][9])) - (K14*pop[i][83]*pop[i][10])/(1800*(km20 + pop[i][10])) - (K15*pop[i][84]*pop[i][10])/(3600*(km21 + pop[i][10])) + (K15*pop[i][84]*pop[i][11])/(3600*(km22 + pop[i][11])) - (K13*pop[i][82]*pop[i][8]*pop[i][31])/(3600*(km17 + pop[i][8]*pop[i][31])))/((F24*K14*pop[i][83]*pop[i][9])/(1800*(km19 + pop[i][9])) - (F25*K14*pop[i][83]*pop[i][10])/(1800*(km20 + pop[i][10]))))
                      
    DNA_SIZE = 134
    for i in range(0,POP_SIZE):
        
        for j in range(0,DNA_SIZE):            
        
            if(pop[i][j] < 0):
                pop[i][j] = 0.001
            
            if(pop[i][j] > 1):
                pop[i][j] = 0.999



    return pop



def fitness(dna,U_Candidate_1,y_pred_1,y_actual_1,y_ref_for_fixed_first_27_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train):
  """
  For each gene in the DNA, this function calculates the difference between
  it and the character in the same position in the OPTIMAL string. These values
  are summed and then returned.
  """
  
  Q = 134                     # Number of input of the Biological MIMO system
  R = 107                     # Number of output of the Biological MIMO system
  num_of_past_input = 19
  num_of_input_to_SVM = (Q+R)*num_of_past_input + Q
  total_input = (Q+R)*num_of_past_input + Q

  h = 1
  Nyq = 1   # prediction horizon
  Nur = 1   # control horizon
  nr = 19     # Number of past inputs
  nz = 19     # Number of past inputs
  mq = 19    # Number of past outputs
  error = 0.001
  mz = 19   # Number of past outputs
  
  y_ref_1 = np.zeros((1,R))  
  fitness = 0
  
  U_Candidate_backup = np.zeros((1,num_of_input_to_SVM))
  y_pred_backup = np.zeros((1,R))
  y_actual_backup = np.zeros((1,R))


  for col in range(0,num_of_input_to_SVM):
     U_Candidate_backup[0,col] = U_Candidate_1[0,col]
     
  for col in range(0,R):
     y_pred_backup[0,col] = y_pred_1[0,col]
     
  for col in range(0,R):
     y_actual_backup[0,col] = y_actual_1[0,col]
     
     
     
  for ww in range(0,Q):
      pos2 = ww*nz+ww
      U_Candidate_backup[0,pos2+1:pos2+nz] = U_Candidate_backup[0,pos2:pos2+nz-1]        
      U_Candidate_backup[0,pos2] = dna[ww]


  for yy in range(0,R):
      s2 = yy*mz
      s3 = Q*nz
      U_Candidate_backup[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate_backup[0,s3+Q+s2:s3+Q+s2+mz-2]        
      U_Candidate_backup[0,s3+Q+s2] = y_actual_backup[0,yy]      
      
  y_act = plant_output_check_for_NaN(U_Candidate_backup,X_train, Y_train) 

  #print('\n\n plant_output_check_for_NaN done')

  
  if((math.isnan(y_act[0]) == False) and (math.isnan(y_act[1]) == False) and (math.isnan(y_act[2]) == False) and (math.isnan(y_act[3]) == False) and (math.isnan(y_act[4]) == False) and (math.isnan(y_act[5]) == False) and (math.isnan(y_act[6]) == False) and (math.isnan(y_act[7]) == False) and (math.isnan(y_act[8]) == False) and (math.isnan(y_act[9]) == False) and (math.isnan(y_act[10]) == False) and (math.isnan(y_act[11]) == False) and (math.isnan(y_act[12]) == False) and (math.isnan(y_act[13]) == False) and (math.isnan(y_act[14]) == False) and (math.isnan(y_act[15]) == False) and (math.isnan(y_act[16]) == False) and (math.isnan(y_act[17]) == False) and (math.isnan(y_act[18]) == False) and (math.isnan(y_act[19]) == False) and (math.isnan(y_act[20]) == False) and (math.isnan(y_act[21]) == False) and (math.isnan(y_act[22]) == False) and (math.isnan(y_act[23]) == False) and (math.isnan(y_act[24]) == False) and (math.isnan(y_act[25]) == False) and (math.isnan(y_act[26]) == False) and (math.isnan(y_act[27]) == False) and (math.isnan(y_act[28]) == False) and (math.isnan(y_act[29]) == False) and (math.isnan(y_act[30]) == False) and (math.isnan(y_act[31]) == False) and (math.isnan(y_act[32]) == False) and (math.isnan(y_act[33]) == False) and (math.isnan(y_act[34]) == False) and (math.isnan(y_act[35]) == False) and (math.isnan(y_act[36]) == False) and (math.isnan(y_act[37]) == False) and (math.isnan(y_act[38]) == False) and (math.isnan(y_act[39]) == False) and (math.isnan(y_act[40]) == False) and (math.isnan(y_act[41]) == False) and (math.isnan(y_act[42]) == False) and (math.isnan(y_act[43]) == False) and (math.isnan(y_act[44]) == False) and (math.isnan(y_act[45]) == False) and (math.isnan(y_act[46]) == False) and (math.isnan(y_act[47]) == False) and (math.isnan(y_act[48]) == False) and (math.isnan(y_act[49]) == False) and (math.isnan(y_act[50]) == False) and (math.isnan(y_act[51]) == False) and (math.isnan(y_act[52]) == False) and (math.isnan(y_act[53]) == False) and (math.isnan(y_act[54]) == False) and (math.isnan(y_act[55]) == False) and (math.isnan(y_act[56]) == False) and (math.isnan(y_act[57]) == False) and (math.isnan(y_act[58]) == False) and (math.isnan(y_act[59]) == False) and (math.isnan(y_act[60]) == False) and (math.isnan(y_act[61]) == False) and (math.isnan(y_act[62]) == False) and (math.isnan(y_act[63]) == False) and (math.isnan(y_act[64]) == False) and (math.isnan(y_act[65]) == False) and (math.isnan(y_act[66]) == False) and (math.isnan(y_act[67]) == False) and (math.isnan(y_act[68]) == False) and (math.isnan(y_act[69]) == False) and (math.isnan(y_act[70]) == False) and (math.isnan(y_act[71]) == False) and (math.isnan(y_act[72]) == False) and (math.isnan(y_act[73]) == False) and (math.isnan(y_act[74]) == False) and (math.isnan(y_act[75]) == False) and (math.isnan(y_act[76]) == False) and (math.isnan(y_act[77]) == False) and (math.isnan(y_act[78]) == False) and (math.isnan(y_act[79]) == False) and (math.isnan(y_act[80]) == False) and (math.isnan(y_act[81]) == False) and (math.isnan(y_act[82]) == False) and (math.isnan(y_act[83]) == False) and (math.isnan(y_act[84]) == False) and (math.isnan(y_act[85]) == False) and (math.isnan(y_act[86]) == False) and (math.isnan(y_act[87]) == False) and (math.isnan(y_act[88]) == False) and (math.isnan(y_act[89]) == False) and (math.isnan(y_act[90]) == False) and (math.isnan(y_act[91]) == False) and (math.isnan(y_act[92]) == False) and (math.isnan(y_act[93]) == False) and (math.isnan(y_act[94]) == False) and (math.isnan(y_act[95]) == False) and (math.isnan(y_act[96]) == False) and (math.isnan(y_act[97]) == False) and (math.isnan(y_act[98]) == False) and (math.isnan(y_act[99]) == False) and (math.isnan(y_act[100]) == False) and (math.isnan(y_act[101]) == False) and (math.isnan(y_act[102]) == False) and (math.isnan(y_act[103]) == False) and (math.isnan(y_act[104]) == False) and (math.isnan(y_act[105]) == False) and (math.isnan(y_act[106]) == False)):  ### eta thk kora holo
      y_pred_backup[0,:] = SVM_model_predict_output(U_Candidate_backup)
      #print('\n\n SVM_model_predict_output done')
      y_actual_backup = plant_output_prediction(U_Candidate_backup,X_train, Y_train)
      #print('\n\n plant_output_prediction done')
      y_ref_1[0,:] =[y_pred_backup[0,0], y_pred_backup[0,1], y_pred_backup[0,2], y_pred_backup[0,3], y_pred_backup[0,4], y_pred_backup[0,5], y_pred_backup[0,6], y_pred_backup[0,7], y_pred_backup[0,8], y_pred_backup[0,9], y_pred_backup[0,10], y_pred_backup[0,11], y_pred_backup[0,12], y_pred_backup[0,13], y_pred_backup[0,14], y_pred_backup[0,15], y_pred_backup[0,16], y_pred_backup[0,17], y_pred_backup[0,18], y_pred_backup[0,19], y_pred_backup[0,20], y_pred_backup[0,21], y_pred_backup[0,22], y_pred_backup[0,23], y_pred_backup[0,24], y_pred_backup[0,25], y_ref_for_fixed_first_27_31[0,26], y_pred_backup[0,27], y_pred_backup[0,28], y_pred_backup[0,29], y_ref_for_fixed_first_27_31[0,30], y_pred_backup[0,31], y_pred_backup[0,32], y_pred_backup[0,33], y_pred_backup[0,34], y_pred_backup[0,35], y_pred_backup[0,36], y_pred_backup[0,37], y_pred_backup[0,38], y_pred_backup[0,39], y_pred_backup[0,40], y_pred_backup[0,41], y_pred_backup[0,42], y_pred_backup[0,43], y_pred_backup[0,44], y_pred_backup[0,45], y_pred_backup[0,46], y_pred_backup[0,47], y_pred_backup[0,48], y_pred_backup[0,49], y_pred_backup[0,50], y_pred_backup[0,51], y_pred_backup[0,52], y_pred_backup[0,53], y_pred_backup[0,54], y_pred_backup[0,55], y_pred_backup[0,56], y_pred_backup[0,57], y_pred_backup[0,58], y_pred_backup[0,59], y_pred_backup[0,60], y_pred_backup[0,61], y_pred_backup[0,62], y_pred_backup[0,63], y_pred_backup[0,64], y_pred_backup[0,65], y_pred_backup[0,66], y_pred_backup[0,67], y_pred_backup[0,68], y_pred_backup[0,69], y_pred_backup[0,70], y_pred_backup[0,71], y_pred_backup[0,72], y_pred_backup[0,73], y_pred_backup[0,74], y_pred_backup[0,75], y_pred_backup[0,76], y_pred_backup[0,77], y_pred_backup[0,78], y_pred_backup[0,79], y_pred_backup[0,80], y_pred_backup[0,81], y_pred_backup[0,82], y_pred_backup[0,83], y_pred_backup[0,84], y_pred_backup[0,85], y_pred_backup[0,86], y_pred_backup[0,87], y_pred_backup[0,88], y_pred_backup[0,89], y_pred_backup[0,90], y_pred_backup[0,91], y_pred_backup[0,92], y_pred_backup[0,93], y_pred_backup[0,94], y_pred_backup[0,95], y_pred_backup[0,96], y_pred_backup[0,97], y_pred_backup[0,98], y_pred_backup[0,99], y_pred_backup[0,100], y_pred_backup[0,101], y_pred_backup[0,102], y_pred_backup[0,103], y_pred_backup[0,104], y_pred_backup[0,105], y_pred_backup[0,106]]   #### eta thk kora holo
 
      ## calculate first term of the objective function
      sum_error = 0
      for q in range(0,R):
          for k in range(0,Nyq):
              t = (y_pred_backup[k,q] - y_ref_1[k,q])**2   ### eta thk kora holo
              sum_error = sum_error + t
      
      ## calculate second term of the objective function
      lamb = [0.005 for ll in range(Q)]
      sum2 = 0
      for r in range(0,Q):
          pp = r*nz + r
          for k in range(0,Nur):
              t = lamb[r] *  ( U_Candidate_backup[k,pp] -  U_Candidate_backup[k,pp+1])**2
              sum2 = sum2 + t
        
      ## Final objective value
      OB = np.real(sum_error + sum2)
      #print('\n\n Final objective value calculation done')
      #OB_all[iii] = OB
      iii += 1
  
      fitness = np.real(1/(OB + 0.01))   

  else:
      fitness = -9999.9999    
  
  

  return fitness, U_Candidate_1, y_pred_1, y_actual_1, y_pred_backup, iii

def fitness2(dna,U_Candidate_1,y_pred_1,y_actual_1,y_ref_for_fixed_first_27_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train):
  """
  For each gene in the DNA, this function calculates the difference between
  it and the character in the same position in the OPTIMAL string. These values
  are summed and then returned.
  """
  
  Q = 134                     # Number of input of the Biological MIMO system
  R = 107                     # Number of output of the Biological MIMO system
  num_of_past_input = 19
  num_of_input_to_SVM = (Q+R)*num_of_past_input + Q
  total_input = (Q+R)*num_of_past_input + Q

  h = 1
  Nyq = 1   # prediction horizon
  Nur = 1   # control horizon
  nr = 19     # Number of past inputs
  nz = 19     # Number of past inputs
  mq = 19    # Number of past outputs
  error = 0.001
  mz = 19   # Number of past outputs
  
  y_ref_1 = np.zeros((1,R))  
  fitness = 0
  
  U_Candidate_backup = np.zeros((1,num_of_input_to_SVM))
  y_pred_backup = np.zeros((1,R))
  y_actual_backup = np.zeros((1,R))


  for col in range(0,num_of_input_to_SVM):
     U_Candidate_backup[0,col] = U_Candidate_1[0,col]
     
  for col in range(0,R):
     y_pred_backup[0,col] = y_pred_1[0,col]
     
  for col in range(0,R):
     y_actual_backup[0,col] = y_actual_1[0,col]
     
     
     
  for ww in range(0,Q):
      pos2 = ww*nz+ww
      U_Candidate_backup[0,pos2+1:pos2+nz] = U_Candidate_backup[0,pos2:pos2+nz-1]        
      U_Candidate_backup[0,pos2] = dna[ww]


  for yy in range(0,R):
      s2 = yy*mz
      s3 = Q*nz
      U_Candidate_backup[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate_backup[0,s3+Q+s2:s3+Q+s2+mz-2]        
      U_Candidate_backup[0,s3+Q+s2] = y_actual_backup[0,yy]      
      
  y_act = plant_output_check_for_NaN(U_Candidate_backup,X_train, Y_train) 

  #print('\n\n plant_output_check_for_NaN done')

  
  if((math.isnan(y_act[0]) == False) and (math.isnan(y_act[1]) == False) and (math.isnan(y_act[2]) == False) and (math.isnan(y_act[3]) == False) and (math.isnan(y_act[4]) == False) and (math.isnan(y_act[5]) == False) and (math.isnan(y_act[6]) == False) and (math.isnan(y_act[7]) == False) and (math.isnan(y_act[8]) == False) and (math.isnan(y_act[9]) == False) and (math.isnan(y_act[10]) == False) and (math.isnan(y_act[11]) == False) and (math.isnan(y_act[12]) == False) and (math.isnan(y_act[13]) == False) and (math.isnan(y_act[14]) == False) and (math.isnan(y_act[15]) == False) and (math.isnan(y_act[16]) == False) and (math.isnan(y_act[17]) == False) and (math.isnan(y_act[18]) == False) and (math.isnan(y_act[19]) == False) and (math.isnan(y_act[20]) == False) and (math.isnan(y_act[21]) == False) and (math.isnan(y_act[22]) == False) and (math.isnan(y_act[23]) == False) and (math.isnan(y_act[24]) == False) and (math.isnan(y_act[25]) == False) and (math.isnan(y_act[26]) == False) and (math.isnan(y_act[27]) == False) and (math.isnan(y_act[28]) == False) and (math.isnan(y_act[29]) == False) and (math.isnan(y_act[30]) == False) and (math.isnan(y_act[31]) == False) and (math.isnan(y_act[32]) == False) and (math.isnan(y_act[33]) == False) and (math.isnan(y_act[34]) == False) and (math.isnan(y_act[35]) == False) and (math.isnan(y_act[36]) == False) and (math.isnan(y_act[37]) == False) and (math.isnan(y_act[38]) == False) and (math.isnan(y_act[39]) == False) and (math.isnan(y_act[40]) == False) and (math.isnan(y_act[41]) == False) and (math.isnan(y_act[42]) == False) and (math.isnan(y_act[43]) == False) and (math.isnan(y_act[44]) == False) and (math.isnan(y_act[45]) == False) and (math.isnan(y_act[46]) == False) and (math.isnan(y_act[47]) == False) and (math.isnan(y_act[48]) == False) and (math.isnan(y_act[49]) == False) and (math.isnan(y_act[50]) == False) and (math.isnan(y_act[51]) == False) and (math.isnan(y_act[52]) == False) and (math.isnan(y_act[53]) == False) and (math.isnan(y_act[54]) == False) and (math.isnan(y_act[55]) == False) and (math.isnan(y_act[56]) == False) and (math.isnan(y_act[57]) == False) and (math.isnan(y_act[58]) == False) and (math.isnan(y_act[59]) == False) and (math.isnan(y_act[60]) == False) and (math.isnan(y_act[61]) == False) and (math.isnan(y_act[62]) == False) and (math.isnan(y_act[63]) == False) and (math.isnan(y_act[64]) == False) and (math.isnan(y_act[65]) == False) and (math.isnan(y_act[66]) == False) and (math.isnan(y_act[67]) == False) and (math.isnan(y_act[68]) == False) and (math.isnan(y_act[69]) == False) and (math.isnan(y_act[70]) == False) and (math.isnan(y_act[71]) == False) and (math.isnan(y_act[72]) == False) and (math.isnan(y_act[73]) == False) and (math.isnan(y_act[74]) == False) and (math.isnan(y_act[75]) == False) and (math.isnan(y_act[76]) == False) and (math.isnan(y_act[77]) == False) and (math.isnan(y_act[78]) == False) and (math.isnan(y_act[79]) == False) and (math.isnan(y_act[80]) == False) and (math.isnan(y_act[81]) == False) and (math.isnan(y_act[82]) == False) and (math.isnan(y_act[83]) == False) and (math.isnan(y_act[84]) == False) and (math.isnan(y_act[85]) == False) and (math.isnan(y_act[86]) == False) and (math.isnan(y_act[87]) == False) and (math.isnan(y_act[88]) == False) and (math.isnan(y_act[89]) == False) and (math.isnan(y_act[90]) == False) and (math.isnan(y_act[91]) == False) and (math.isnan(y_act[92]) == False) and (math.isnan(y_act[93]) == False) and (math.isnan(y_act[94]) == False) and (math.isnan(y_act[95]) == False) and (math.isnan(y_act[96]) == False) and (math.isnan(y_act[97]) == False) and (math.isnan(y_act[98]) == False) and (math.isnan(y_act[99]) == False) and (math.isnan(y_act[100]) == False) and (math.isnan(y_act[101]) == False) and (math.isnan(y_act[102]) == False) and (math.isnan(y_act[103]) == False) and (math.isnan(y_act[104]) == False) and (math.isnan(y_act[105]) == False) and (math.isnan(y_act[106]) == False)):  ### eta thk kora holo
      y_pred_backup[0,:] = SVM_model_predict_output(U_Candidate_backup)
      #print('\n\n SVM_model_predict_output done')
      y_actual_backup = plant_output_prediction(U_Candidate_backup,X_train, Y_train)
      #print('\n\n plant_output_prediction done')
      y_ref_1[0,:] =[y_pred_backup[0,0], y_pred_backup[0,1], y_pred_backup[0,2], y_pred_backup[0,3], y_pred_backup[0,4], y_pred_backup[0,5], y_pred_backup[0,6], y_pred_backup[0,7], y_pred_backup[0,8], y_pred_backup[0,9], y_pred_backup[0,10], y_pred_backup[0,11], y_pred_backup[0,12], y_pred_backup[0,13], y_pred_backup[0,14], y_pred_backup[0,15], y_pred_backup[0,16], y_pred_backup[0,17], y_pred_backup[0,18], y_pred_backup[0,19], y_pred_backup[0,20], y_pred_backup[0,21], y_pred_backup[0,22], y_pred_backup[0,23], y_pred_backup[0,24], y_pred_backup[0,25], y_pred_backup[0,26], y_pred_backup[0,27], y_pred_backup[0,28], y_pred_backup[0,29], y_pred_backup[0,30], y_pred_backup[0,31], y_pred_backup[0,32], y_pred_backup[0,33], y_pred_backup[0,34], y_pred_backup[0,35], y_pred_backup[0,36], y_pred_backup[0,37], y_pred_backup[0,38], y_pred_backup[0,39], y_pred_backup[0,40], y_pred_backup[0,41], y_pred_backup[0,42], y_pred_backup[0,43], y_pred_backup[0,44], y_pred_backup[0,45], y_pred_backup[0,46], y_pred_backup[0,47], y_pred_backup[0,48], y_pred_backup[0,49], y_pred_backup[0,50], y_pred_backup[0,51], y_pred_backup[0,52], y_pred_backup[0,53], y_pred_backup[0,54], y_pred_backup[0,55], y_pred_backup[0,56], y_pred_backup[0,57], y_pred_backup[0,58], y_pred_backup[0,59], y_pred_backup[0,60], y_pred_backup[0,61], y_pred_backup[0,62], y_pred_backup[0,63], y_pred_backup[0,64], y_pred_backup[0,65], y_pred_backup[0,66], y_pred_backup[0,67], y_pred_backup[0,68], y_pred_backup[0,69], y_pred_backup[0,70], y_pred_backup[0,71], y_pred_backup[0,72], y_pred_backup[0,73], y_pred_backup[0,74], y_pred_backup[0,75], y_pred_backup[0,76], y_pred_backup[0,77], y_pred_backup[0,78], y_pred_backup[0,79], y_pred_backup[0,80], y_pred_backup[0,81], y_pred_backup[0,82], y_pred_backup[0,83], y_pred_backup[0,84], y_ref_for_fixed_first_27_31[0,85], y_pred_backup[0,86], y_pred_backup[0,87], y_pred_backup[0,88], y_pred_backup[0,89], y_pred_backup[0,90], y_pred_backup[0,91], y_pred_backup[0,92], y_pred_backup[0,93], y_pred_backup[0,94], y_pred_backup[0,95], y_pred_backup[0,96], y_pred_backup[0,97], y_pred_backup[0,98], y_pred_backup[0,99], y_pred_backup[0,100], y_pred_backup[0,101], y_pred_backup[0,102], y_pred_backup[0,103], y_pred_backup[0,104], y_pred_backup[0,105], y_pred_backup[0,106]]   #### eta thk kora holo
 
      ## calculate first term of the objective function
      sum_error = 0
      for q in range(0,R):
          for k in range(0,Nyq):
              t = (y_pred_backup[k,q] - y_ref_1[k,q])**2   ### eta thk kora holo
              sum_error = sum_error + t
      
      ## calculate second term of the objective function
      lamb = [0.005 for ll in range(Q)]
      sum2 = 0
      for r in range(0,Q):
          pp = r*nz + r
          for k in range(0,Nur):
              t = lamb[r] *  ( U_Candidate_backup[k,pp] -  U_Candidate_backup[k,pp+1])**2
              sum2 = sum2 + t
        
      ## Final objective value
      OB = np.real(sum_error + sum2)
      #print('\n\n Final objective value calculation done')
      #OB_all[iii] = OB
      iii += 1
  
      fitness = np.real(1/(OB + 0.01))   

  else:
      fitness = -9999.9999    
  
  

  return fitness, U_Candidate_1, y_pred_1, y_actual_1, y_pred_backup, iii



def fitness3(dna,U_Candidate_1,y_pred_1,y_actual_1,y_ref_for_fixed_first_27_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train):
  """
  For each gene in the DNA, this function calculates the difference between
  it and the character in the same position in the OPTIMAL string. These values
  are summed and then returned.
  """
  
  Q = 134                     # Number of input of the Biological MIMO system
  R = 107                     # Number of output of the Biological MIMO system
  num_of_past_input = 19
  num_of_input_to_SVM = (Q+R)*num_of_past_input + Q
  total_input = (Q+R)*num_of_past_input + Q

  h = 1
  Nyq = 1   # prediction horizon
  Nur = 1   # control horizon
  nr = 19     # Number of past inputs
  nz = 19     # Number of past inputs
  mq = 19    # Number of past outputs
  error = 0.001
  mz = 19   # Number of past outputs
  
  y_ref_1 = np.zeros((1,R))  
  fitness = 0
  
  U_Candidate_backup = np.zeros((1,num_of_input_to_SVM))
  y_pred_backup = np.zeros((1,R))
  y_actual_backup = np.zeros((1,R))


  for col in range(0,num_of_input_to_SVM):
     U_Candidate_backup[0,col] = U_Candidate_1[0,col]
     
  for col in range(0,R):
     y_pred_backup[0,col] = y_pred_1[0,col]
     
  for col in range(0,R):
     y_actual_backup[0,col] = y_actual_1[0,col]
     
     
     
  for ww in range(0,Q):
      pos2 = ww*nz+ww
      U_Candidate_backup[0,pos2+1:pos2+nz] = U_Candidate_backup[0,pos2:pos2+nz-1]        
      U_Candidate_backup[0,pos2] = dna[ww]


  for yy in range(0,R):
      s2 = yy*mz
      s3 = Q*nz
      U_Candidate_backup[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate_backup[0,s3+Q+s2:s3+Q+s2+mz-2]        
      U_Candidate_backup[0,s3+Q+s2] = y_actual_backup[0,yy]      
      
  y_act = plant_output_check_for_NaN(U_Candidate_backup,X_train, Y_train) 

  #print('\n\n plant_output_check_for_NaN done')

  
  if((math.isnan(y_act[0]) == False) and (math.isnan(y_act[1]) == False) and (math.isnan(y_act[2]) == False) and (math.isnan(y_act[3]) == False) and (math.isnan(y_act[4]) == False) and (math.isnan(y_act[5]) == False) and (math.isnan(y_act[6]) == False) and (math.isnan(y_act[7]) == False) and (math.isnan(y_act[8]) == False) and (math.isnan(y_act[9]) == False) and (math.isnan(y_act[10]) == False) and (math.isnan(y_act[11]) == False) and (math.isnan(y_act[12]) == False) and (math.isnan(y_act[13]) == False) and (math.isnan(y_act[14]) == False) and (math.isnan(y_act[15]) == False) and (math.isnan(y_act[16]) == False) and (math.isnan(y_act[17]) == False) and (math.isnan(y_act[18]) == False) and (math.isnan(y_act[19]) == False) and (math.isnan(y_act[20]) == False) and (math.isnan(y_act[21]) == False) and (math.isnan(y_act[22]) == False) and (math.isnan(y_act[23]) == False) and (math.isnan(y_act[24]) == False) and (math.isnan(y_act[25]) == False) and (math.isnan(y_act[26]) == False) and (math.isnan(y_act[27]) == False) and (math.isnan(y_act[28]) == False) and (math.isnan(y_act[29]) == False) and (math.isnan(y_act[30]) == False) and (math.isnan(y_act[31]) == False) and (math.isnan(y_act[32]) == False) and (math.isnan(y_act[33]) == False) and (math.isnan(y_act[34]) == False) and (math.isnan(y_act[35]) == False) and (math.isnan(y_act[36]) == False) and (math.isnan(y_act[37]) == False) and (math.isnan(y_act[38]) == False) and (math.isnan(y_act[39]) == False) and (math.isnan(y_act[40]) == False) and (math.isnan(y_act[41]) == False) and (math.isnan(y_act[42]) == False) and (math.isnan(y_act[43]) == False) and (math.isnan(y_act[44]) == False) and (math.isnan(y_act[45]) == False) and (math.isnan(y_act[46]) == False) and (math.isnan(y_act[47]) == False) and (math.isnan(y_act[48]) == False) and (math.isnan(y_act[49]) == False) and (math.isnan(y_act[50]) == False) and (math.isnan(y_act[51]) == False) and (math.isnan(y_act[52]) == False) and (math.isnan(y_act[53]) == False) and (math.isnan(y_act[54]) == False) and (math.isnan(y_act[55]) == False) and (math.isnan(y_act[56]) == False) and (math.isnan(y_act[57]) == False) and (math.isnan(y_act[58]) == False) and (math.isnan(y_act[59]) == False) and (math.isnan(y_act[60]) == False) and (math.isnan(y_act[61]) == False) and (math.isnan(y_act[62]) == False) and (math.isnan(y_act[63]) == False) and (math.isnan(y_act[64]) == False) and (math.isnan(y_act[65]) == False) and (math.isnan(y_act[66]) == False) and (math.isnan(y_act[67]) == False) and (math.isnan(y_act[68]) == False) and (math.isnan(y_act[69]) == False) and (math.isnan(y_act[70]) == False) and (math.isnan(y_act[71]) == False) and (math.isnan(y_act[72]) == False) and (math.isnan(y_act[73]) == False) and (math.isnan(y_act[74]) == False) and (math.isnan(y_act[75]) == False) and (math.isnan(y_act[76]) == False) and (math.isnan(y_act[77]) == False) and (math.isnan(y_act[78]) == False) and (math.isnan(y_act[79]) == False) and (math.isnan(y_act[80]) == False) and (math.isnan(y_act[81]) == False) and (math.isnan(y_act[82]) == False) and (math.isnan(y_act[83]) == False) and (math.isnan(y_act[84]) == False) and (math.isnan(y_act[85]) == False) and (math.isnan(y_act[86]) == False) and (math.isnan(y_act[87]) == False) and (math.isnan(y_act[88]) == False) and (math.isnan(y_act[89]) == False) and (math.isnan(y_act[90]) == False) and (math.isnan(y_act[91]) == False) and (math.isnan(y_act[92]) == False) and (math.isnan(y_act[93]) == False) and (math.isnan(y_act[94]) == False) and (math.isnan(y_act[95]) == False) and (math.isnan(y_act[96]) == False) and (math.isnan(y_act[97]) == False) and (math.isnan(y_act[98]) == False) and (math.isnan(y_act[99]) == False) and (math.isnan(y_act[100]) == False) and (math.isnan(y_act[101]) == False) and (math.isnan(y_act[102]) == False) and (math.isnan(y_act[103]) == False) and (math.isnan(y_act[104]) == False) and (math.isnan(y_act[105]) == False) and (math.isnan(y_act[106]) == False)):  ### eta thk kora holo
      y_pred_backup[0,:] = SVM_model_predict_output(U_Candidate_backup)
      #print('\n\n SVM_model_predict_output done')
      y_actual_backup = plant_output_prediction(U_Candidate_backup,X_train, Y_train)
      #print('\n\n plant_output_prediction done')
      y_ref_1[0,:] =[y_pred_backup[0,0], y_pred_backup[0,1], y_pred_backup[0,2], y_pred_backup[0,3], y_pred_backup[0,4], y_pred_backup[0,5], y_pred_backup[0,6], y_pred_backup[0,7], y_pred_backup[0,8], y_pred_backup[0,9], y_pred_backup[0,10], y_pred_backup[0,11], y_pred_backup[0,12], y_pred_backup[0,13], y_pred_backup[0,14], y_pred_backup[0,15], y_pred_backup[0,16], y_pred_backup[0,17], y_pred_backup[0,18], y_pred_backup[0,19], y_pred_backup[0,20], y_pred_backup[0,21], y_pred_backup[0,22], y_pred_backup[0,23], y_pred_backup[0,24], y_pred_backup[0,25], y_pred_backup[0,26], y_pred_backup[0,27], y_pred_backup[0,28], y_pred_backup[0,29], y_pred_backup[0,30], y_pred_backup[0,31], y_pred_backup[0,32], y_pred_backup[0,33], y_pred_backup[0,34], y_pred_backup[0,35], y_pred_backup[0,36], y_pred_backup[0,37], y_pred_backup[0,38], y_pred_backup[0,39], y_pred_backup[0,40], y_pred_backup[0,41], y_pred_backup[0,42], y_pred_backup[0,43], y_pred_backup[0,44], y_pred_backup[0,45], y_pred_backup[0,46], y_pred_backup[0,47], y_pred_backup[0,48], y_pred_backup[0,49], y_pred_backup[0,50], y_pred_backup[0,51], y_pred_backup[0,52], y_pred_backup[0,53], y_pred_backup[0,54], y_pred_backup[0,55], y_pred_backup[0,56], y_pred_backup[0,57], y_pred_backup[0,58], y_pred_backup[0,59], y_pred_backup[0,60], y_pred_backup[0,61], y_pred_backup[0,62], y_pred_backup[0,63], y_pred_backup[0,64], y_pred_backup[0,65], y_pred_backup[0,66], y_pred_backup[0,67], y_pred_backup[0,68], y_pred_backup[0,69], y_pred_backup[0,70], y_pred_backup[0,71], y_pred_backup[0,72], y_pred_backup[0,73], y_pred_backup[0,74], y_pred_backup[0,75], y_pred_backup[0,76], y_pred_backup[0,77], y_pred_backup[0,78], y_pred_backup[0,79], y_pred_backup[0,80], y_pred_backup[0,81], y_pred_backup[0,82], y_pred_backup[0,83], y_pred_backup[0,84], y_pred_backup[0,85], y_pred_backup[0,86], y_pred_backup[0,87], y_pred_backup[0,88], y_pred_backup[0,89], y_pred_backup[0,90], y_pred_backup[0,91], y_pred_backup[0,92], y_pred_backup[0,93], y_pred_backup[0,94], y_pred_backup[0,95], y_pred_backup[0,96], y_pred_backup[0,97], y_pred_backup[0,98], y_pred_backup[0,99], y_ref_for_fixed_first_27_31[0,100], y_pred_backup[0,101], y_pred_backup[0,102], y_pred_backup[0,103], y_pred_backup[0,104], y_pred_backup[0,105], y_pred_backup[0,106]]   #### eta thk kora holo
 
      ## calculate first term of the objective function
      sum_error = 0
      for q in range(0,R):
          for k in range(0,Nyq):
              t = (y_pred_backup[k,q] - y_ref_1[k,q])**2   ### eta thk kora holo
              sum_error = sum_error + t
      
      ## calculate second term of the objective function
      lamb = [0.005 for ll in range(Q)]
      sum2 = 0
      for r in range(0,Q):
          pp = r*nz + r
          for k in range(0,Nur):
              t = lamb[r] *  ( U_Candidate_backup[k,pp] -  U_Candidate_backup[k,pp+1])**2
              sum2 = sum2 + t
        
      ## Final objective value
      OB = np.real(sum_error + sum2)
      #print('\n\n Final objective value calculation done')
      #OB_all[iii] = OB
      iii += 1
  
      fitness = np.real(1/(OB + 0.01))   

  else:
      fitness = -9999.9999    
  
  

  return fitness, U_Candidate_1, y_pred_1, y_actual_1, y_pred_backup, iii



def fitness4(dna,U_Candidate_1,y_pred_1,y_actual_1,y_ref_for_fixed_first_27_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train):
  """
  For each gene in the DNA, this function calculates the difference between
  it and the character in the same position in the OPTIMAL string. These values
  are summed and then returned.
  """
  
  Q = 134                     # Number of input of the Biological MIMO system
  R = 107                     # Number of output of the Biological MIMO system
  num_of_past_input = 19
  num_of_input_to_SVM = (Q+R)*num_of_past_input + Q
  total_input = (Q+R)*num_of_past_input + Q

  h = 1
  Nyq = 1   # prediction horizon
  Nur = 1   # control horizon
  nr = 19     # Number of past inputs
  nz = 19     # Number of past inputs
  mq = 19    # Number of past outputs
  error = 0.001
  mz = 19   # Number of past outputs
  
  y_ref_1 = np.zeros((1,R))  
  fitness = 0
  
  U_Candidate_backup = np.zeros((1,num_of_input_to_SVM))
  y_pred_backup = np.zeros((1,R))
  y_actual_backup = np.zeros((1,R))


  for col in range(0,num_of_input_to_SVM):
     U_Candidate_backup[0,col] = U_Candidate_1[0,col]
     
  for col in range(0,R):
     y_pred_backup[0,col] = y_pred_1[0,col]
     
  for col in range(0,R):
     y_actual_backup[0,col] = y_actual_1[0,col]
     
     
     
  for ww in range(0,Q):
      pos2 = ww*nz+ww
      U_Candidate_backup[0,pos2+1:pos2+nz] = U_Candidate_backup[0,pos2:pos2+nz-1]        
      U_Candidate_backup[0,pos2] = dna[ww]


  for yy in range(0,R):
      s2 = yy*mz
      s3 = Q*nz
      U_Candidate_backup[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate_backup[0,s3+Q+s2:s3+Q+s2+mz-2]        
      U_Candidate_backup[0,s3+Q+s2] = y_actual_backup[0,yy]      
      
  y_act = plant_output_check_for_NaN(U_Candidate_backup,X_train, Y_train) 

  #print('\n\n plant_output_check_for_NaN done')

  
  if((math.isnan(y_act[0]) == False) and (math.isnan(y_act[1]) == False) and (math.isnan(y_act[2]) == False) and (math.isnan(y_act[3]) == False) and (math.isnan(y_act[4]) == False) and (math.isnan(y_act[5]) == False) and (math.isnan(y_act[6]) == False) and (math.isnan(y_act[7]) == False) and (math.isnan(y_act[8]) == False) and (math.isnan(y_act[9]) == False) and (math.isnan(y_act[10]) == False) and (math.isnan(y_act[11]) == False) and (math.isnan(y_act[12]) == False) and (math.isnan(y_act[13]) == False) and (math.isnan(y_act[14]) == False) and (math.isnan(y_act[15]) == False) and (math.isnan(y_act[16]) == False) and (math.isnan(y_act[17]) == False) and (math.isnan(y_act[18]) == False) and (math.isnan(y_act[19]) == False) and (math.isnan(y_act[20]) == False) and (math.isnan(y_act[21]) == False) and (math.isnan(y_act[22]) == False) and (math.isnan(y_act[23]) == False) and (math.isnan(y_act[24]) == False) and (math.isnan(y_act[25]) == False) and (math.isnan(y_act[26]) == False) and (math.isnan(y_act[27]) == False) and (math.isnan(y_act[28]) == False) and (math.isnan(y_act[29]) == False) and (math.isnan(y_act[30]) == False) and (math.isnan(y_act[31]) == False) and (math.isnan(y_act[32]) == False) and (math.isnan(y_act[33]) == False) and (math.isnan(y_act[34]) == False) and (math.isnan(y_act[35]) == False) and (math.isnan(y_act[36]) == False) and (math.isnan(y_act[37]) == False) and (math.isnan(y_act[38]) == False) and (math.isnan(y_act[39]) == False) and (math.isnan(y_act[40]) == False) and (math.isnan(y_act[41]) == False) and (math.isnan(y_act[42]) == False) and (math.isnan(y_act[43]) == False) and (math.isnan(y_act[44]) == False) and (math.isnan(y_act[45]) == False) and (math.isnan(y_act[46]) == False) and (math.isnan(y_act[47]) == False) and (math.isnan(y_act[48]) == False) and (math.isnan(y_act[49]) == False) and (math.isnan(y_act[50]) == False) and (math.isnan(y_act[51]) == False) and (math.isnan(y_act[52]) == False) and (math.isnan(y_act[53]) == False) and (math.isnan(y_act[54]) == False) and (math.isnan(y_act[55]) == False) and (math.isnan(y_act[56]) == False) and (math.isnan(y_act[57]) == False) and (math.isnan(y_act[58]) == False) and (math.isnan(y_act[59]) == False) and (math.isnan(y_act[60]) == False) and (math.isnan(y_act[61]) == False) and (math.isnan(y_act[62]) == False) and (math.isnan(y_act[63]) == False) and (math.isnan(y_act[64]) == False) and (math.isnan(y_act[65]) == False) and (math.isnan(y_act[66]) == False) and (math.isnan(y_act[67]) == False) and (math.isnan(y_act[68]) == False) and (math.isnan(y_act[69]) == False) and (math.isnan(y_act[70]) == False) and (math.isnan(y_act[71]) == False) and (math.isnan(y_act[72]) == False) and (math.isnan(y_act[73]) == False) and (math.isnan(y_act[74]) == False) and (math.isnan(y_act[75]) == False) and (math.isnan(y_act[76]) == False) and (math.isnan(y_act[77]) == False) and (math.isnan(y_act[78]) == False) and (math.isnan(y_act[79]) == False) and (math.isnan(y_act[80]) == False) and (math.isnan(y_act[81]) == False) and (math.isnan(y_act[82]) == False) and (math.isnan(y_act[83]) == False) and (math.isnan(y_act[84]) == False) and (math.isnan(y_act[85]) == False) and (math.isnan(y_act[86]) == False) and (math.isnan(y_act[87]) == False) and (math.isnan(y_act[88]) == False) and (math.isnan(y_act[89]) == False) and (math.isnan(y_act[90]) == False) and (math.isnan(y_act[91]) == False) and (math.isnan(y_act[92]) == False) and (math.isnan(y_act[93]) == False) and (math.isnan(y_act[94]) == False) and (math.isnan(y_act[95]) == False) and (math.isnan(y_act[96]) == False) and (math.isnan(y_act[97]) == False) and (math.isnan(y_act[98]) == False) and (math.isnan(y_act[99]) == False) and (math.isnan(y_act[100]) == False) and (math.isnan(y_act[101]) == False) and (math.isnan(y_act[102]) == False) and (math.isnan(y_act[103]) == False) and (math.isnan(y_act[104]) == False) and (math.isnan(y_act[105]) == False) and (math.isnan(y_act[106]) == False)):  ### eta thk kora holo
      y_pred_backup[0,:] = SVM_model_predict_output(U_Candidate_backup)
      #print('\n\n SVM_model_predict_output done')
      y_actual_backup = plant_output_prediction(U_Candidate_backup,X_train, Y_train)
      #print('\n\n plant_output_prediction done')
      y_ref_1[0,:] =[y_pred_backup[0,0], y_pred_backup[0,1], y_pred_backup[0,2], y_pred_backup[0,3], y_pred_backup[0,4], y_pred_backup[0,5], y_pred_backup[0,6], y_pred_backup[0,7], y_pred_backup[0,8], y_pred_backup[0,9], y_pred_backup[0,10], y_pred_backup[0,11], y_pred_backup[0,12], y_pred_backup[0,13], y_pred_backup[0,14], y_pred_backup[0,15], y_pred_backup[0,16], y_pred_backup[0,17], y_pred_backup[0,18], y_pred_backup[0,19], y_pred_backup[0,20], y_pred_backup[0,21], y_pred_backup[0,22], y_pred_backup[0,23], y_pred_backup[0,24], y_pred_backup[0,25], y_pred_backup[0,26], y_pred_backup[0,27], y_pred_backup[0,28], y_pred_backup[0,29], y_pred_backup[0,30], y_pred_backup[0,31], y_pred_backup[0,32], y_pred_backup[0,33], y_pred_backup[0,34], y_pred_backup[0,35], y_pred_backup[0,36], y_pred_backup[0,37], y_pred_backup[0,38], y_pred_backup[0,39], y_pred_backup[0,40], y_pred_backup[0,41], y_pred_backup[0,42], y_pred_backup[0,43], y_pred_backup[0,44], y_pred_backup[0,45], y_pred_backup[0,46], y_pred_backup[0,47], y_pred_backup[0,48], y_pred_backup[0,49], y_pred_backup[0,50], y_pred_backup[0,51], y_pred_backup[0,52], y_pred_backup[0,53], y_pred_backup[0,54], y_pred_backup[0,55], y_pred_backup[0,56], y_pred_backup[0,57], y_pred_backup[0,58], y_pred_backup[0,59], y_pred_backup[0,60], y_pred_backup[0,61], y_pred_backup[0,62], y_pred_backup[0,63], y_pred_backup[0,64], y_pred_backup[0,65], y_pred_backup[0,66], y_pred_backup[0,67], y_pred_backup[0,68], y_pred_backup[0,69], y_pred_backup[0,70], y_pred_backup[0,71], y_pred_backup[0,72], y_pred_backup[0,73], y_pred_backup[0,74], y_pred_backup[0,75], y_pred_backup[0,76], y_pred_backup[0,77], y_pred_backup[0,78], y_pred_backup[0,79], y_pred_backup[0,80], y_pred_backup[0,81], y_pred_backup[0,82], y_pred_backup[0,83], y_pred_backup[0,84], y_pred_backup[0,85], y_pred_backup[0,86], y_pred_backup[0,87], y_pred_backup[0,88], y_pred_backup[0,89], y_pred_backup[0,90], y_pred_backup[0,91], y_pred_backup[0,92], y_pred_backup[0,93], y_pred_backup[0,94], y_pred_backup[0,95], y_pred_backup[0,96], y_pred_backup[0,97], y_pred_backup[0,98], y_pred_backup[0,99], y_pred_backup[0,100], y_pred_backup[0,101], y_pred_backup[0,102], y_pred_backup[0,103], y_pred_backup[0,104], y_ref_for_fixed_first_27_31[0,105], y_pred_backup[0,106]]   #### eta thk kora holo
 
      ## calculate first term of the objective function
      sum_error = 0
      for q in range(0,R):
          for k in range(0,Nyq):
              t = (y_pred_backup[k,q] - y_ref_1[k,q])**2   ### eta thk kora holo
              sum_error = sum_error + t
      
      ## calculate second term of the objective function
      lamb = [0.005 for ll in range(Q)]
      sum2 = 0
      for r in range(0,Q):
          pp = r*nz + r
          for k in range(0,Nur):
              t = lamb[r] *  ( U_Candidate_backup[k,pp] -  U_Candidate_backup[k,pp+1])**2
              sum2 = sum2 + t
        
      ## Final objective value
      OB = np.real(sum_error + sum2)
      #print('\n\n Final objective value calculation done')
      #OB_all[iii] = OB
      iii += 1
  
      fitness = np.real(1/(OB + 0.01))   

  else:
      fitness = -9999.9999    
  
  

  return fitness, U_Candidate_1, y_pred_1, y_actual_1, y_pred_backup, iii




def fitness5(dna,U_Candidate_1,y_pred_1,y_actual_1,y_ref_for_fixed_first_27_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train):
  """
  For each gene in the DNA, this function calculates the difference between
  it and the character in the same position in the OPTIMAL string. These values
  are summed and then returned.
  """
  
  Q = 134                     # Number of input of the Biological MIMO system
  R = 107                     # Number of output of the Biological MIMO system
  num_of_past_input = 19
  num_of_input_to_SVM = (Q+R)*num_of_past_input + Q
  total_input = (Q+R)*num_of_past_input + Q

  h = 1
  Nyq = 1   # prediction horizon
  Nur = 1   # control horizon
  nr = 19     # Number of past inputs
  nz = 19     # Number of past inputs
  mq = 19    # Number of past outputs
  error = 0.001
  mz = 19   # Number of past outputs
  
  y_ref_1 = np.zeros((1,R))  
  fitness = 0
  
  U_Candidate_backup = np.zeros((1,num_of_input_to_SVM))
  y_pred_backup = np.zeros((1,R))
  y_actual_backup = np.zeros((1,R))


  for col in range(0,num_of_input_to_SVM):
     U_Candidate_backup[0,col] = U_Candidate_1[0,col]
     
  for col in range(0,R):
     y_pred_backup[0,col] = y_pred_1[0,col]
     
  for col in range(0,R):
     y_actual_backup[0,col] = y_actual_1[0,col]
     
     
     
  for ww in range(0,Q):
      pos2 = ww*nz+ww
      U_Candidate_backup[0,pos2+1:pos2+nz] = U_Candidate_backup[0,pos2:pos2+nz-1]        
      U_Candidate_backup[0,pos2] = dna[ww]


  for yy in range(0,R):
      s2 = yy*mz
      s3 = Q*nz
      U_Candidate_backup[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate_backup[0,s3+Q+s2:s3+Q+s2+mz-2]        
      U_Candidate_backup[0,s3+Q+s2] = y_actual_backup[0,yy]      
      
  y_act = plant_output_check_for_NaN(U_Candidate_backup,X_train, Y_train) 

  #print('\n\n plant_output_check_for_NaN done')

  
  if((math.isnan(y_act[0]) == False) and (math.isnan(y_act[1]) == False) and (math.isnan(y_act[2]) == False) and (math.isnan(y_act[3]) == False) and (math.isnan(y_act[4]) == False) and (math.isnan(y_act[5]) == False) and (math.isnan(y_act[6]) == False) and (math.isnan(y_act[7]) == False) and (math.isnan(y_act[8]) == False) and (math.isnan(y_act[9]) == False) and (math.isnan(y_act[10]) == False) and (math.isnan(y_act[11]) == False) and (math.isnan(y_act[12]) == False) and (math.isnan(y_act[13]) == False) and (math.isnan(y_act[14]) == False) and (math.isnan(y_act[15]) == False) and (math.isnan(y_act[16]) == False) and (math.isnan(y_act[17]) == False) and (math.isnan(y_act[18]) == False) and (math.isnan(y_act[19]) == False) and (math.isnan(y_act[20]) == False) and (math.isnan(y_act[21]) == False) and (math.isnan(y_act[22]) == False) and (math.isnan(y_act[23]) == False) and (math.isnan(y_act[24]) == False) and (math.isnan(y_act[25]) == False) and (math.isnan(y_act[26]) == False) and (math.isnan(y_act[27]) == False) and (math.isnan(y_act[28]) == False) and (math.isnan(y_act[29]) == False) and (math.isnan(y_act[30]) == False) and (math.isnan(y_act[31]) == False) and (math.isnan(y_act[32]) == False) and (math.isnan(y_act[33]) == False) and (math.isnan(y_act[34]) == False) and (math.isnan(y_act[35]) == False) and (math.isnan(y_act[36]) == False) and (math.isnan(y_act[37]) == False) and (math.isnan(y_act[38]) == False) and (math.isnan(y_act[39]) == False) and (math.isnan(y_act[40]) == False) and (math.isnan(y_act[41]) == False) and (math.isnan(y_act[42]) == False) and (math.isnan(y_act[43]) == False) and (math.isnan(y_act[44]) == False) and (math.isnan(y_act[45]) == False) and (math.isnan(y_act[46]) == False) and (math.isnan(y_act[47]) == False) and (math.isnan(y_act[48]) == False) and (math.isnan(y_act[49]) == False) and (math.isnan(y_act[50]) == False) and (math.isnan(y_act[51]) == False) and (math.isnan(y_act[52]) == False) and (math.isnan(y_act[53]) == False) and (math.isnan(y_act[54]) == False) and (math.isnan(y_act[55]) == False) and (math.isnan(y_act[56]) == False) and (math.isnan(y_act[57]) == False) and (math.isnan(y_act[58]) == False) and (math.isnan(y_act[59]) == False) and (math.isnan(y_act[60]) == False) and (math.isnan(y_act[61]) == False) and (math.isnan(y_act[62]) == False) and (math.isnan(y_act[63]) == False) and (math.isnan(y_act[64]) == False) and (math.isnan(y_act[65]) == False) and (math.isnan(y_act[66]) == False) and (math.isnan(y_act[67]) == False) and (math.isnan(y_act[68]) == False) and (math.isnan(y_act[69]) == False) and (math.isnan(y_act[70]) == False) and (math.isnan(y_act[71]) == False) and (math.isnan(y_act[72]) == False) and (math.isnan(y_act[73]) == False) and (math.isnan(y_act[74]) == False) and (math.isnan(y_act[75]) == False) and (math.isnan(y_act[76]) == False) and (math.isnan(y_act[77]) == False) and (math.isnan(y_act[78]) == False) and (math.isnan(y_act[79]) == False) and (math.isnan(y_act[80]) == False) and (math.isnan(y_act[81]) == False) and (math.isnan(y_act[82]) == False) and (math.isnan(y_act[83]) == False) and (math.isnan(y_act[84]) == False) and (math.isnan(y_act[85]) == False) and (math.isnan(y_act[86]) == False) and (math.isnan(y_act[87]) == False) and (math.isnan(y_act[88]) == False) and (math.isnan(y_act[89]) == False) and (math.isnan(y_act[90]) == False) and (math.isnan(y_act[91]) == False) and (math.isnan(y_act[92]) == False) and (math.isnan(y_act[93]) == False) and (math.isnan(y_act[94]) == False) and (math.isnan(y_act[95]) == False) and (math.isnan(y_act[96]) == False) and (math.isnan(y_act[97]) == False) and (math.isnan(y_act[98]) == False) and (math.isnan(y_act[99]) == False) and (math.isnan(y_act[100]) == False) and (math.isnan(y_act[101]) == False) and (math.isnan(y_act[102]) == False) and (math.isnan(y_act[103]) == False) and (math.isnan(y_act[104]) == False) and (math.isnan(y_act[105]) == False) and (math.isnan(y_act[106]) == False)):  ### eta thk kora holo
      y_pred_backup[0,:] = SVM_model_predict_output(U_Candidate_backup)
      #print('\n\n SVM_model_predict_output done')
      y_actual_backup = plant_output_prediction(U_Candidate_backup,X_train, Y_train)
      #print('\n\n plant_output_prediction done')
      y_ref_1[0,:] =[y_pred_backup[0,0], y_pred_backup[0,1], y_pred_backup[0,2], y_pred_backup[0,3], y_pred_backup[0,4], y_pred_backup[0,5], y_pred_backup[0,6], y_pred_backup[0,7], y_pred_backup[0,8], y_pred_backup[0,9], y_pred_backup[0,10], y_pred_backup[0,11], y_pred_backup[0,12], y_pred_backup[0,13], y_pred_backup[0,14], y_pred_backup[0,15], y_pred_backup[0,16], y_pred_backup[0,17], y_pred_backup[0,18], y_pred_backup[0,19], y_pred_backup[0,20], y_pred_backup[0,21], y_pred_backup[0,22], y_pred_backup[0,23], y_pred_backup[0,24], y_pred_backup[0,25], y_pred_backup[0,26], y_pred_backup[0,27], y_pred_backup[0,28], y_pred_backup[0,29], y_pred_backup[0,30], y_pred_backup[0,31], y_pred_backup[0,32], y_pred_backup[0,33], y_pred_backup[0,34], y_pred_backup[0,35], y_pred_backup[0,36], y_pred_backup[0,37], y_pred_backup[0,38], y_pred_backup[0,39], y_pred_backup[0,40], y_pred_backup[0,41], y_pred_backup[0,42], y_pred_backup[0,43], y_pred_backup[0,44], y_pred_backup[0,45], y_pred_backup[0,46], y_pred_backup[0,47], y_pred_backup[0,48], y_pred_backup[0,49], y_pred_backup[0,50], y_pred_backup[0,51], y_pred_backup[0,52], y_pred_backup[0,53], y_pred_backup[0,54], y_pred_backup[0,55], y_pred_backup[0,56], y_pred_backup[0,57], y_pred_backup[0,58], y_pred_backup[0,59], y_pred_backup[0,60], y_pred_backup[0,61], y_pred_backup[0,62], y_pred_backup[0,63], y_pred_backup[0,64], y_pred_backup[0,65], y_pred_backup[0,66], y_pred_backup[0,67], y_pred_backup[0,68], y_pred_backup[0,69], y_pred_backup[0,70], y_pred_backup[0,71], y_pred_backup[0,72], y_pred_backup[0,73], y_pred_backup[0,74], y_pred_backup[0,75], y_pred_backup[0,76], y_pred_backup[0,77], y_pred_backup[0,78], y_pred_backup[0,79], y_pred_backup[0,80], y_pred_backup[0,81], y_pred_backup[0,82], y_pred_backup[0,83], y_pred_backup[0,84], y_pred_backup[0,85], y_pred_backup[0,86], y_pred_backup[0,87], y_pred_backup[0,88], y_pred_backup[0,89], y_pred_backup[0,90], y_pred_backup[0,91], y_pred_backup[0,92], y_pred_backup[0,93], y_pred_backup[0,94], y_pred_backup[0,95], y_pred_backup[0,96], y_pred_backup[0,97], y_pred_backup[0,98], y_pred_backup[0,99], y_pred_backup[0,100], y_pred_backup[0,101], y_pred_backup[0,102], y_ref_for_fixed_first_27_31[0,103], y_pred_backup[0,104], y_pred_backup[0,105], y_pred_backup[0,106]]   #### eta thk kora holo
 
      ## calculate first term of the objective function
      sum_error = 0
      for q in range(0,R):
          for k in range(0,Nyq):
              t = (y_pred_backup[k,q] - y_ref_1[k,q])**2   ### eta thk kora holo
              sum_error = sum_error + t
      
      ## calculate second term of the objective function
      lamb = [0.005 for ll in range(Q)]
      sum2 = 0
      for r in range(0,Q):
          pp = r*nz + r
          for k in range(0,Nur):
              t = lamb[r] *  ( U_Candidate_backup[k,pp] -  U_Candidate_backup[k,pp+1])**2
              sum2 = sum2 + t
        
      ## Final objective value
      OB = np.real(sum_error + sum2)
      #print('\n\n Final objective value calculation done')
      #OB_all[iii] = OB
      iii += 1
  
      fitness = np.real(1/(OB + 0.01))   

  else:
      fitness = -9999.9999    
  
  

  return fitness, U_Candidate_1, y_pred_1, y_actual_1, y_pred_backup, iii



def fitness6(dna,U_Candidate_1,y_pred_1,y_actual_1,y_ref_for_fixed_first_27_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train):
  """
  For each gene in the DNA, this function calculates the difference between
  it and the character in the same position in the OPTIMAL string. These values
  are summed and then returned.
  """
  
  Q = 134                     # Number of input of the Biological MIMO system
  R = 107                     # Number of output of the Biological MIMO system
  num_of_past_input = 19
  num_of_input_to_SVM = (Q+R)*num_of_past_input + Q
  total_input = (Q+R)*num_of_past_input + Q

  h = 1
  Nyq = 1   # prediction horizon
  Nur = 1   # control horizon
  nr = 19     # Number of past inputs
  nz = 19     # Number of past inputs
  mq = 19    # Number of past outputs
  error = 0.001
  mz = 19   # Number of past outputs
  
  y_ref_1 = np.zeros((1,R))  
  fitness = 0
  
  U_Candidate_backup = np.zeros((1,num_of_input_to_SVM))
  y_pred_backup = np.zeros((1,R))
  y_actual_backup = np.zeros((1,R))


  for col in range(0,num_of_input_to_SVM):
     U_Candidate_backup[0,col] = U_Candidate_1[0,col]
     
  for col in range(0,R):
     y_pred_backup[0,col] = y_pred_1[0,col]
     
  for col in range(0,R):
     y_actual_backup[0,col] = y_actual_1[0,col]
     
     
     
  for ww in range(0,Q):
      pos2 = ww*nz+ww
      U_Candidate_backup[0,pos2+1:pos2+nz] = U_Candidate_backup[0,pos2:pos2+nz-1]        
      U_Candidate_backup[0,pos2] = dna[ww]


  for yy in range(0,R):
      s2 = yy*mz
      s3 = Q*nz
      U_Candidate_backup[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate_backup[0,s3+Q+s2:s3+Q+s2+mz-2]        
      U_Candidate_backup[0,s3+Q+s2] = y_actual_backup[0,yy]      
      
  y_act = plant_output_check_for_NaN(U_Candidate_backup,X_train, Y_train) 

  #print('\n\n plant_output_check_for_NaN done')

  
  if((math.isnan(y_act[0]) == False) and (math.isnan(y_act[1]) == False) and (math.isnan(y_act[2]) == False) and (math.isnan(y_act[3]) == False) and (math.isnan(y_act[4]) == False) and (math.isnan(y_act[5]) == False) and (math.isnan(y_act[6]) == False) and (math.isnan(y_act[7]) == False) and (math.isnan(y_act[8]) == False) and (math.isnan(y_act[9]) == False) and (math.isnan(y_act[10]) == False) and (math.isnan(y_act[11]) == False) and (math.isnan(y_act[12]) == False) and (math.isnan(y_act[13]) == False) and (math.isnan(y_act[14]) == False) and (math.isnan(y_act[15]) == False) and (math.isnan(y_act[16]) == False) and (math.isnan(y_act[17]) == False) and (math.isnan(y_act[18]) == False) and (math.isnan(y_act[19]) == False) and (math.isnan(y_act[20]) == False) and (math.isnan(y_act[21]) == False) and (math.isnan(y_act[22]) == False) and (math.isnan(y_act[23]) == False) and (math.isnan(y_act[24]) == False) and (math.isnan(y_act[25]) == False) and (math.isnan(y_act[26]) == False) and (math.isnan(y_act[27]) == False) and (math.isnan(y_act[28]) == False) and (math.isnan(y_act[29]) == False) and (math.isnan(y_act[30]) == False) and (math.isnan(y_act[31]) == False) and (math.isnan(y_act[32]) == False) and (math.isnan(y_act[33]) == False) and (math.isnan(y_act[34]) == False) and (math.isnan(y_act[35]) == False) and (math.isnan(y_act[36]) == False) and (math.isnan(y_act[37]) == False) and (math.isnan(y_act[38]) == False) and (math.isnan(y_act[39]) == False) and (math.isnan(y_act[40]) == False) and (math.isnan(y_act[41]) == False) and (math.isnan(y_act[42]) == False) and (math.isnan(y_act[43]) == False) and (math.isnan(y_act[44]) == False) and (math.isnan(y_act[45]) == False) and (math.isnan(y_act[46]) == False) and (math.isnan(y_act[47]) == False) and (math.isnan(y_act[48]) == False) and (math.isnan(y_act[49]) == False) and (math.isnan(y_act[50]) == False) and (math.isnan(y_act[51]) == False) and (math.isnan(y_act[52]) == False) and (math.isnan(y_act[53]) == False) and (math.isnan(y_act[54]) == False) and (math.isnan(y_act[55]) == False) and (math.isnan(y_act[56]) == False) and (math.isnan(y_act[57]) == False) and (math.isnan(y_act[58]) == False) and (math.isnan(y_act[59]) == False) and (math.isnan(y_act[60]) == False) and (math.isnan(y_act[61]) == False) and (math.isnan(y_act[62]) == False) and (math.isnan(y_act[63]) == False) and (math.isnan(y_act[64]) == False) and (math.isnan(y_act[65]) == False) and (math.isnan(y_act[66]) == False) and (math.isnan(y_act[67]) == False) and (math.isnan(y_act[68]) == False) and (math.isnan(y_act[69]) == False) and (math.isnan(y_act[70]) == False) and (math.isnan(y_act[71]) == False) and (math.isnan(y_act[72]) == False) and (math.isnan(y_act[73]) == False) and (math.isnan(y_act[74]) == False) and (math.isnan(y_act[75]) == False) and (math.isnan(y_act[76]) == False) and (math.isnan(y_act[77]) == False) and (math.isnan(y_act[78]) == False) and (math.isnan(y_act[79]) == False) and (math.isnan(y_act[80]) == False) and (math.isnan(y_act[81]) == False) and (math.isnan(y_act[82]) == False) and (math.isnan(y_act[83]) == False) and (math.isnan(y_act[84]) == False) and (math.isnan(y_act[85]) == False) and (math.isnan(y_act[86]) == False) and (math.isnan(y_act[87]) == False) and (math.isnan(y_act[88]) == False) and (math.isnan(y_act[89]) == False) and (math.isnan(y_act[90]) == False) and (math.isnan(y_act[91]) == False) and (math.isnan(y_act[92]) == False) and (math.isnan(y_act[93]) == False) and (math.isnan(y_act[94]) == False) and (math.isnan(y_act[95]) == False) and (math.isnan(y_act[96]) == False) and (math.isnan(y_act[97]) == False) and (math.isnan(y_act[98]) == False) and (math.isnan(y_act[99]) == False) and (math.isnan(y_act[100]) == False) and (math.isnan(y_act[101]) == False) and (math.isnan(y_act[102]) == False) and (math.isnan(y_act[103]) == False) and (math.isnan(y_act[104]) == False) and (math.isnan(y_act[105]) == False) and (math.isnan(y_act[106]) == False)):  ### eta thk kora holo
      y_pred_backup[0,:] = SVM_model_predict_output(U_Candidate_backup)
      #print('\n\n SVM_model_predict_output done')
      y_actual_backup = plant_output_prediction(U_Candidate_backup,X_train, Y_train)
      #print('\n\n plant_output_prediction done')
      y_ref_1[0,:] =[y_pred_backup[0,0], y_pred_backup[0,1], y_pred_backup[0,2], y_pred_backup[0,3], y_pred_backup[0,4], y_pred_backup[0,5], y_pred_backup[0,6], y_pred_backup[0,7], y_pred_backup[0,8], y_pred_backup[0,9], y_pred_backup[0,10], y_pred_backup[0,11], y_pred_backup[0,12], y_pred_backup[0,13], y_pred_backup[0,14], y_pred_backup[0,15], y_pred_backup[0,16], y_pred_backup[0,17], y_pred_backup[0,18], y_pred_backup[0,19], y_pred_backup[0,20], y_pred_backup[0,21], y_pred_backup[0,22], y_pred_backup[0,23], y_pred_backup[0,24], y_pred_backup[0,25], y_pred_backup[0,26], y_pred_backup[0,27], y_pred_backup[0,28], y_pred_backup[0,29], y_pred_backup[0,30], y_pred_backup[0,31], y_pred_backup[0,32], y_pred_backup[0,33], y_pred_backup[0,34], y_pred_backup[0,35], y_pred_backup[0,36], y_pred_backup[0,37], y_pred_backup[0,38], y_pred_backup[0,39], y_pred_backup[0,40], y_pred_backup[0,41], y_pred_backup[0,42], y_pred_backup[0,43], y_pred_backup[0,44], y_pred_backup[0,45], y_pred_backup[0,46], y_pred_backup[0,47], y_pred_backup[0,48], y_pred_backup[0,49], y_pred_backup[0,50], y_pred_backup[0,51], y_pred_backup[0,52], y_pred_backup[0,53], y_pred_backup[0,54], y_pred_backup[0,55], y_pred_backup[0,56], y_pred_backup[0,57], y_pred_backup[0,58], y_pred_backup[0,59], y_pred_backup[0,60], y_pred_backup[0,61], y_pred_backup[0,62], y_pred_backup[0,63], y_pred_backup[0,64], y_pred_backup[0,65], y_pred_backup[0,66], y_pred_backup[0,67], y_pred_backup[0,68], y_pred_backup[0,69], y_pred_backup[0,70], y_pred_backup[0,71], y_pred_backup[0,72], y_pred_backup[0,73],  y_ref_for_fixed_first_27_31[0,74], y_pred_backup[0,75], y_pred_backup[0,76], y_pred_backup[0,77], y_pred_backup[0,78], y_pred_backup[0,79], y_pred_backup[0,80], y_pred_backup[0,81], y_pred_backup[0,82], y_pred_backup[0,83], y_pred_backup[0,84], y_pred_backup[0,85], y_pred_backup[0,86], y_pred_backup[0,87], y_pred_backup[0,88], y_pred_backup[0,89], y_pred_backup[0,90], y_pred_backup[0,91], y_pred_backup[0,92], y_pred_backup[0,93], y_pred_backup[0,94], y_pred_backup[0,95], y_pred_backup[0,96], y_pred_backup[0,97], y_pred_backup[0,98], y_pred_backup[0,99], y_pred_backup[0,100], y_pred_backup[0,101], y_pred_backup[0,102], y_pred_backup[0,103], y_pred_backup[0,104], y_pred_backup[0,105], y_pred_backup[0,106]]   #### eta thk kora holo
 
      ## calculate first term of the objective function
      sum_error = 0
      for q in range(0,R):
          for k in range(0,Nyq):
              t = (y_pred_backup[k,q] - y_ref_1[k,q])**2   ### eta thk kora holo
              sum_error = sum_error + t
      
      ## calculate second term of the objective function
      lamb = [0.005 for ll in range(Q)]
      sum2 = 0
      for r in range(0,Q):
          pp = r*nz + r
          for k in range(0,Nur):
              t = lamb[r] *  ( U_Candidate_backup[k,pp] -  U_Candidate_backup[k,pp+1])**2
              sum2 = sum2 + t
        
      ## Final objective value
      OB = np.real(sum_error + sum2)
      #print('\n\n Final objective value calculation done')
      #OB_all[iii] = OB
      iii += 1
  
      fitness = np.real(1/(OB + 0.01))   

  else:
      fitness = -9999.9999    
  
  

  return fitness, U_Candidate_1, y_pred_1, y_actual_1, y_pred_backup, iii




def weighted_choice(items):
  """
  Chooses a random element from items, where items is a list of tuples in
  the form (item, weight). weight determines the probability of choosing its
  respective item. Note: this function is borrowed from ActiveState Recipes.
  """
  weight_total = sum((item[1] for item in items))
  n = random.uniform(0, weight_total)
  for item, weight in items:
    if n < weight:
      return item
    n = n - weight
  return item


def crossover(dna1, dna2):
  

  """
  Slices both dna1 and dna2 into two parts at a random index within their
  length and merges them. Both keep their initial sublist up to the crossover
  index, but their ends are swapped.
  """
  fixed = 0.01
  POP_SIZE    = 20
  DNA_SIZE    = 134
  no_of_control_input = 27
  dna1_mod = np.zeros(DNA_SIZE)
  dna2_mod = np.zeros(DNA_SIZE)
  pos = 53#int(random.random()*DNA_SIZE) ## assume crossover by mid position
  
  dna1_mod[0:pos] = dna1[0:pos]
  dna2_mod[0:pos] = dna2[0:pos]
  for i in range(pos,DNA_SIZE-no_of_control_input):
      dna1_mod[i] = dna2[i]
      dna2_mod[i] = dna1[i]
      


  constant_terms_in_u18_design = ((KS7*dna1_mod[43])/60 + (KS29*dna1_mod[43])/60 + KS13/(60*(F_S15*dna1_mod[43] + 1)*(F_S14*dna1_mod[58] + 1)) + (KS9*dna1_mod[47])/(60*(F_S8*dna1_mod[43] + 1)*(F_S9*dna1_mod[46] + 1)) - (KS36*(kbg16 - lg16 + kg16*dna1_mod[48]))/(60*FG6*kg16*dna1_mod[48]) + (KS8*dna1_mod[41]*dna1_mod[42]*dna1_mod[43]*dna1_mod[46]*dna1_mod[49])/(60*(F_S6*dna1_mod[56] + 1)*(F_S7*dna1_mod[57] + 1)))
  dna1_mod[107] = np.real(-(60*((KS31*(kbg16 - lg16 + kg16*dna1_mod[48]))/(60*FG6*kg16*dna1_mod[48]) - (KS8*dna1_mod[41]*dna1_mod[42]*dna1_mod[43]*dna1_mod[46]*dna1_mod[49])/(60*(F_S6*dna1_mod[56] + 1)*(F_S7*dna1_mod[57] + 1))))/KS1)
  dna1_mod[108] = np.real(((kg28*dna1_mod[45]*dna1_mod[64]*dna1_mod[66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna1_mod[63] + 60) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))))/((kbg28 - lg28)*(kg30*dna1_mod[43] + kg9*dna1_mod[43]*dna1_mod[66] + kg27*dna1_mod[43]*dna1_mod[66] + kg37*dna1_mod[41]*dna1_mod[66] + kg29*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + kg21*dna1_mod[41]*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + (kg20*dna1_mod[43]*dna1_mod[45]*dna1_mod[66])/(FG9*dna1_mod[58] + 1) + (F_S22*KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60)))) - 1)/FG13)
  dna1_mod[109] = np.real(-(60*((KS29*dna1_mod[43])/60 + (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/(60*(F_S21*dna1_mod[50] + 1)*(F_S20*dna1_mod[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna1_mod[63] + 60) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))))/(kg30*dna1_mod[43] + kg9*dna1_mod[43]*dna1_mod[66] + kg27*dna1_mod[43]*dna1_mod[66] + kg37*dna1_mod[41]*dna1_mod[66] + kg29*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + kg21*dna1_mod[41]*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + (kg20*dna1_mod[43]*dna1_mod[45]*dna1_mod[66])/(FG9*dna1_mod[58] + 1) + (F_S22*KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))) - 1))))/KS54)
  dna1_mod[110] = np.real(-((KS49*((KS33*((KS29*dna1_mod[43])/60 + (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/(60*(F_S21*dna1_mod[50] + 1)*(F_S20*dna1_mod[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna1_mod[63] + 60) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))))/(kg30*dna1_mod[43] + kg9*dna1_mod[43]*dna1_mod[66] + kg27*dna1_mod[43]*dna1_mod[66] + kg37*dna1_mod[41]*dna1_mod[66] + kg29*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + kg21*dna1_mod[41]*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + (kg20*dna1_mod[43]*dna1_mod[45]*dna1_mod[66])/(FG9*dna1_mod[58] + 1) + (F_S22*KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))) - 1))))/KS54 - (KS32*((kg28*dna1_mod[45]*dna1_mod[64]*dna1_mod[66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna1_mod[63] + 60) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))))/((kbg28 - lg28)*(kg30*dna1_mod[43] + kg9*dna1_mod[43]*dna1_mod[66] + kg27*dna1_mod[43]*dna1_mod[66] + kg37*dna1_mod[41]*dna1_mod[66] + kg29*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + kg21*dna1_mod[41]*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + (kg20*dna1_mod[43]*dna1_mod[45]*dna1_mod[66])/(FG9*dna1_mod[58] + 1) + (F_S22*KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60)))) - 1))/(60*FG13) - (KS2*dna1_mod[44]*dna1_mod[51])/60 + (KS17*dna1_mod[42])/(60*(F_S17*dna1_mod[44] + 1)*(F_S18*dna1_mod[45] + 1)) + (KS8*dna1_mod[41]*dna1_mod[42]*dna1_mod[43]*dna1_mod[46]*dna1_mod[49])/(60*(F_S6*dna1_mod[56] + 1)*(F_S7*dna1_mod[57] + 1))))/KS35 - (KS2*dna1_mod[44]*dna1_mod[51])/60)/(KS48/60 - (KS34*KS49)/(60*KS35)))
  dna1_mod[111] = np.real((60*((KS2*dna1_mod[44]*dna1_mod[51])/60 + (KS48*((KS49*((KS33*((KS29*dna1_mod[43])/60 + (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/(60*(F_S21*dna1_mod[50] + 1)*(F_S20*dna1_mod[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna1_mod[63] + 60) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))))/(kg30*dna1_mod[43] + kg9*dna1_mod[43]*dna1_mod[66] + kg27*dna1_mod[43]*dna1_mod[66] + kg37*dna1_mod[41]*dna1_mod[66] + kg29*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + kg21*dna1_mod[41]*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + (kg20*dna1_mod[43]*dna1_mod[45]*dna1_mod[66])/(FG9*dna1_mod[58] + 1) + (F_S22*KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))) - 1))))/KS54 - (KS32*((kg28*dna1_mod[45]*dna1_mod[64]*dna1_mod[66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna1_mod[63] + 60) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))))/((kbg28 - lg28)*(kg30*dna1_mod[43] + kg9*dna1_mod[43]*dna1_mod[66] + kg27*dna1_mod[43]*dna1_mod[66] + kg37*dna1_mod[41]*dna1_mod[66] + kg29*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + kg21*dna1_mod[41]*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + (kg20*dna1_mod[43]*dna1_mod[45]*dna1_mod[66])/(FG9*dna1_mod[58] + 1) + (F_S22*KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60)))) - 1))/(60*FG13) - (KS2*dna1_mod[44]*dna1_mod[51])/60 + (KS17*dna1_mod[42])/(60*(F_S17*dna1_mod[44] + 1)*(F_S18*dna1_mod[45] + 1)) + (KS8*dna1_mod[41]*dna1_mod[42]*dna1_mod[43]*dna1_mod[46]*dna1_mod[49])/(60*(F_S6*dna1_mod[56] + 1)*(F_S7*dna1_mod[57] + 1))))/KS35 - (KS2*dna1_mod[44]*dna1_mod[51])/60))/(60*(KS48/60 - (KS34*KS49)/(60*KS35)))))/KS49)
  dna1_mod[112] = np.real((kbg16 - lg16 + kg16*dna1_mod[48])/(FG6*kg16*dna1_mod[48]))
  dna1_mod[113] = np.real((60*((KS2*dna1_mod[44]*dna1_mod[51])/60 - (KS4*dna1_mod[50])/(60*(F_S3*dna1_mod[55] + 1)) + (KS18*dna1_mod[44])/(60*(F_S19*dna1_mod[45] + 1)) + (KS5*dna1_mod[44]*dna1_mod[68])/(60*(F_S4*dna1_mod[55] + 1)) + (KS14*dna1_mod[44]*dna1_mod[48])/(60*(F_S16*dna1_mod[55] + 1)) + (KS17*dna1_mod[42])/(60*(F_S17*dna1_mod[44] + 1)*(F_S18*dna1_mod[45] + 1)) - (KS37*(kbg16 - lg16 + kg16*dna1_mod[48]))/(60*FG6*kg16*dna1_mod[48]) - (KS52*(kbg16 - lg16 + kg16*dna1_mod[48]))/(60*FG6*kg16*dna1_mod[48]) + (KS6*dna1_mod[44]*dna1_mod[45]*dna1_mod[53]*dna1_mod[54])/(60*((60*(KS24/(60*F_S28*dna1_mod[53] + 60) - (KS28*dna1_mod[30]*dna1_mod[69])/60 + KS23/((F_S26*dna1_mod[21] + 1)*(60*F_S25*dna1_mod[20] + 60)*(F_S27*dna1_mod[53] + 1)) + (KS6*dna1_mod[44]*dna1_mod[45]*dna1_mod[53]*dna1_mod[54])/60))/(KS6*dna1_mod[44]*dna1_mod[45]*dna1_mod[53]*dna1_mod[54]) + 1))))/KS38)
  dna1_mod[114] = np.real(-((K9*dna1_mod[78]*dna1_mod[5])/((F23*dna1_mod[3] + 1)*(3600*km10 + 3600*dna1_mod[5])) - (K8*dna1_mod[77]*dna1_mod[3]*dna1_mod[30]*(F20*dna1_mod[3] + 1))/(3600*km9 + 3600*dna1_mod[3]*dna1_mod[30]))/((F22*K9*dna1_mod[78]*dna1_mod[5])/((F23*dna1_mod[3] + 1)*(3600*km10 + 3600*dna1_mod[5])) + (F21*K8*dna1_mod[77]*dna1_mod[3]*dna1_mod[30]*(F20*dna1_mod[3] + 1))/(3600*km9 + 3600*dna1_mod[3]*dna1_mod[30])))
  dna1_mod[115] = np.real((60*(KS24/(60*F_S28*dna1_mod[53] + 60) - (KS28*dna1_mod[30]*dna1_mod[69])/60 + KS23/((F_S26*dna1_mod[21] + 1)*(60*F_S25*dna1_mod[20] + 60)*(F_S27*dna1_mod[53] + 1)) + (KS6*dna1_mod[44]*dna1_mod[45]*dna1_mod[53]*dna1_mod[54])/60))/(F_S5*KS6*dna1_mod[44]*dna1_mod[45]*dna1_mod[53]*dna1_mod[54]))
  dna1_mod[116] = np.real(-(kbg23 - lg23)/kg23)
  dna1_mod[117] = np.real((60*((KS47*((KS49*((KS33*((KS29*dna1_mod[43])/60 + (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/(60*(F_S21*dna1_mod[50] + 1)*(F_S20*dna1_mod[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna1_mod[63] + 60) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))))/(kg30*dna1_mod[43] + kg9*dna1_mod[43]*dna1_mod[66] + kg27*dna1_mod[43]*dna1_mod[66] + kg37*dna1_mod[41]*dna1_mod[66] + kg29*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + kg21*dna1_mod[41]*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + (kg20*dna1_mod[43]*dna1_mod[45]*dna1_mod[66])/(FG9*dna1_mod[58] + 1) + (F_S22*KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))) - 1))))/KS54 - (KS32*((kg28*dna1_mod[45]*dna1_mod[64]*dna1_mod[66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna1_mod[63] + 60) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))))/((kbg28 - lg28)*(kg30*dna1_mod[43] + kg9*dna1_mod[43]*dna1_mod[66] + kg27*dna1_mod[43]*dna1_mod[66] + kg37*dna1_mod[41]*dna1_mod[66] + kg29*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + kg21*dna1_mod[41]*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + (kg20*dna1_mod[43]*dna1_mod[45]*dna1_mod[66])/(FG9*dna1_mod[58] + 1) + (F_S22*KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60)))) - 1))/(60*FG13) - (KS2*dna1_mod[44]*dna1_mod[51])/60 + (KS17*dna1_mod[42])/(60*(F_S17*dna1_mod[44] + 1)*(F_S18*dna1_mod[45] + 1)) + (KS8*dna1_mod[41]*dna1_mod[42]*dna1_mod[43]*dna1_mod[46]*dna1_mod[49])/(60*(F_S6*dna1_mod[56] + 1)*(F_S7*dna1_mod[57] + 1))))/KS35 - (KS2*dna1_mod[44]*dna1_mod[51])/60))/(60*(KS48/60 - (KS34*KS49)/(60*KS35))) - KS13/(60*(F_S15*dna1_mod[43] + 1)*(F_S14*dna1_mod[58] + 1)) + (KS6*dna1_mod[44]*dna1_mod[45]*dna1_mod[53]*dna1_mod[54])/(60*((60*(KS24/(60*F_S28*dna1_mod[53] + 60) - (KS28*dna1_mod[30]*dna1_mod[69])/60 + KS23/((F_S26*dna1_mod[21] + 1)*(60*F_S25*dna1_mod[20] + 60)*(F_S27*dna1_mod[53] + 1)) + (KS6*dna1_mod[44]*dna1_mod[45]*dna1_mod[53]*dna1_mod[54])/60))/(KS6*dna1_mod[44]*dna1_mod[45]*dna1_mod[53]*dna1_mod[54]) + 1))))/KS46)
  dna1_mod[118] = np.real((60*((KS22*dna1_mod[64])/60 + (KS25*dna1_mod[64])/60 - (KS20*dna1_mod[48]*dna1_mod[58])/(60*((60*F_S11*((KS2*dna1_mod[44]*dna1_mod[51])/60 - (KS4*dna1_mod[50])/(60*(F_S3*dna1_mod[55] + 1)) + (KS18*dna1_mod[44])/(60*(F_S19*dna1_mod[45] + 1)) + (KS5*dna1_mod[44]*dna1_mod[68])/(60*(F_S4*dna1_mod[55] + 1)) + (KS14*dna1_mod[44]*dna1_mod[48])/(60*(F_S16*dna1_mod[55] + 1)) + (KS17*dna1_mod[42])/(60*(F_S17*dna1_mod[44] + 1)*(F_S18*dna1_mod[45] + 1)) - (KS37*(kbg16 - lg16 + kg16*dna1_mod[48]))/(60*FG6*kg16*dna1_mod[48]) - (KS52*(kbg16 - lg16 + kg16*dna1_mod[48]))/(60*FG6*kg16*dna1_mod[48]) + (KS6*dna1_mod[44]*dna1_mod[45]*dna1_mod[53]*dna1_mod[54])/(60*((60*(KS24/(60*F_S28*dna1_mod[53] + 60) - (KS28*dna1_mod[30]*dna1_mod[69])/60 + KS23/((F_S26*dna1_mod[21] + 1)*(60*F_S25*dna1_mod[20] + 60)*(F_S27*dna1_mod[53] + 1)) + (KS6*dna1_mod[44]*dna1_mod[45]*dna1_mod[53]*dna1_mod[54])/60))/(KS6*dna1_mod[44]*dna1_mod[45]*dna1_mod[53]*dna1_mod[54]) + 1))))/KS38 + 1)*(F_S23*dna1_mod[63] + 1)) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/(60*(F_S21*dna1_mod[50] + 1)*(F_S20*dna1_mod[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna1_mod[63] + 60) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))))/(kg30*dna1_mod[43] + kg9*dna1_mod[43]*dna1_mod[66] + kg27*dna1_mod[43]*dna1_mod[66] + kg37*dna1_mod[41]*dna1_mod[66] + kg29*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + kg21*dna1_mod[41]*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + (kg20*dna1_mod[43]*dna1_mod[45]*dna1_mod[66])/(FG9*dna1_mod[58] + 1) + (F_S22*KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))) - 1))))/KS50)
  dna1_mod[119] = np.real(((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*dna1_mod[50] + 60*F_S13*constant_terms_in_u18_design*dna1_mod[45]))/(15*(F_S13*dna1_mod[45] + 1)))**(1/2)/(2*F_S1*constant_terms_in_u18_design))
  dna1_mod[120] = np.real(((60*KS3*dna1_mod[52]*(F_S13*dna1_mod[45] + 1))/(KS11*((30*((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*dna1_mod[50] + 60*F_S13*constant_terms_in_u18_design*dna1_mod[45]))/(15*(F_S13*dna1_mod[45] + 1)))**(1/2))/constant_terms_in_u18_design + 60)) - 1)/F_S2)
  dna1_mod[121] = np.real((KS10/((KS43*((K9*dna1_mod[78]*dna1_mod[5])/((F23*dna1_mod[3] + 1)*(3600*km10 + 3600*dna1_mod[5])) - (K8*dna1_mod[77]*dna1_mod[3]*dna1_mod[30]*(F20*dna1_mod[3] + 1))/(3600*km9 + 3600*dna1_mod[3]*dna1_mod[30])))/((F22*K9*dna1_mod[78]*dna1_mod[5])/((F23*dna1_mod[3] + 1)*(3600*km10 + 3600*dna1_mod[5])) + (F21*K8*dna1_mod[77]*dna1_mod[3]*dna1_mod[30]*(F20*dna1_mod[3] + 1))/(3600*km9 + 3600*dna1_mod[3]*dna1_mod[30])) + (KS4*dna1_mod[50])/(F_S3*dna1_mod[55] + 1) + (KS11*dna1_mod[50])/(F_S13*dna1_mod[45] + 1) - (KS39*(kbg16 - lg16 + kg16*dna1_mod[48]))/(FG6*kg16*dna1_mod[48]) + (KS11*dna1_mod[50]*((30*((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*dna1_mod[50] + 60*F_S13*constant_terms_in_u18_design*dna1_mod[45]))/(15*(F_S13*dna1_mod[45] + 1)))**(1/2))/constant_terms_in_u18_design + 60))/(60*(((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*dna1_mod[50] + 60*F_S13*constant_terms_in_u18_design*dna1_mod[45]))/(15*(F_S13*dna1_mod[45] + 1)))**(1/2)/(2*constant_terms_in_u18_design) + 1)*(F_S13*dna1_mod[45] + 1)) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S21*dna1_mod[50] + 1)*(F_S20*dna1_mod[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna1_mod[63] + 60) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))))/(kg30*dna1_mod[43] + kg9*dna1_mod[43]*dna1_mod[66] + kg27*dna1_mod[43]*dna1_mod[66] + kg37*dna1_mod[41]*dna1_mod[66] + kg29*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + kg21*dna1_mod[41]*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + (kg20*dna1_mod[43]*dna1_mod[45]*dna1_mod[66])/(FG9*dna1_mod[58] + 1) + (F_S22*KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60))) - 1))) - 1)/F_S10)
  dna1_mod[122] = np.real(-(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna1_mod[63] + 60) - (KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60)))/(kg30*dna1_mod[43] + kg9*dna1_mod[43]*dna1_mod[66] + kg27*dna1_mod[43]*dna1_mod[66] + kg37*dna1_mod[41]*dna1_mod[66] + kg29*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + kg21*dna1_mod[41]*dna1_mod[42]*dna1_mod[64]*dna1_mod[66] + (kg20*dna1_mod[43]*dna1_mod[45]*dna1_mod[66])/(FG9*dna1_mod[58] + 1) + (F_S22*KS19*dna1_mod[45]*dna1_mod[64]*dna1_mod[65])/((F_S20*dna1_mod[66] + 1)*(60*F_S21*dna1_mod[50] + 60)))) 
  
  AAA = (K31*dna1_mod[100]*km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40])/(12960000*(km43 + dna1_mod[25])*(km44 + dna1_mod[25])))**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[40]))
  BBB = (7200*(F52*dna1_mod[40] + 1)*(km40 + (km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40])/(12960000*(km43 + dna1_mod[25])*(km44 + dna1_mod[25])))**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[40]))/(2*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[40])))*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[40]))
  CCC = (K31*dna1_mod[100]*km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40])/(12960000*(km43 + dna1_mod[25])*(km44 + dna1_mod[25])))**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[40]))
  DDD = (7200*(F52*dna1_mod[40] + 1)*(km40 + (km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40])/(12960000*(km43 + dna1_mod[25])*(km44 + dna1_mod[25])))**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[40]))/(2*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[40])))*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[40]))

  dna1_mod[123] = np.real(-(km0*((K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/(km4 + dna1_mod[1]) - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1)*((3600*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)*((K2*dna1_mod[71]*dna1_mod[0]*((F1*((K2*dna1_mod[71]*dna1_mod[0])/(3600*(km1 + dna1_mod[0])) - (K2*dna1_mod[71]*dna1_mod[1])/(3600*(km2 + dna1_mod[1]))))/((F1*K2*dna1_mod[71]*dna1_mod[0])/(3600*(km1 + dna1_mod[0])) - (F2*K2*dna1_mod[71]*dna1_mod[1])/(3600*(km2 + dna1_mod[1]))) - 1))/(3600*(km1 + dna1_mod[0])) - (K2*dna1_mod[71]*dna1_mod[1]*((F2*((K2*dna1_mod[71]*dna1_mod[0])/(3600*(km1 + dna1_mod[0])) - (K2*dna1_mod[71]*dna1_mod[1])/(3600*(km2 + dna1_mod[1]))))/((F1*K2*dna1_mod[71]*dna1_mod[0])/(3600*(km1 + dna1_mod[0])) - (F2*K2*dna1_mod[71]*dna1_mod[1])/(3600*(km2 + dna1_mod[1]))) - 1))/(3600*(km2 + dna1_mod[1])) + (K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/(3600*(km4 + dna1_mod[1])) + (K5*dna1_mod[74]*dna1_mod[1])/(3600*(km5 + dna1_mod[1])*(F8*dna1_mod[4] + 1)*(F9*dna1_mod[24] + 1)*(F10*dna1_mod[29] + 1)) - (K5*dna1_mod[74]*dna1_mod[3])/(3600*(km6 + dna1_mod[3])*(F11*dna1_mod[4] + 1)*(F12*dna1_mod[24] + 1)*(F13*dna1_mod[29] + 1)) - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))/(3600*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)) + AAA/BBB))/(K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1)) + 1))/((km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1))))/(K1*dna1_mod[70]*(((K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/(km4 + dna1_mod[1]) - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1)*((3600*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)*((K2*dna1_mod[71]*dna1_mod[0]*((F1*((K2*dna1_mod[71]*dna1_mod[0])/(3600*(km1 + dna1_mod[0])) - (K2*dna1_mod[71]*dna1_mod[1])/(3600*(km2 + dna1_mod[1]))))/((F1*K2*dna1_mod[71]*dna1_mod[0])/(3600*(km1 + dna1_mod[0])) - (F2*K2*dna1_mod[71]*dna1_mod[1])/(3600*(km2 + dna1_mod[1]))) - 1))/(3600*(km1 + dna1_mod[0])) - (K2*dna1_mod[71]*dna1_mod[1]*((F2*((K2*dna1_mod[71]*dna1_mod[0])/(3600*(km1 + dna1_mod[0])) - (K2*dna1_mod[71]*dna1_mod[1])/(3600*(km2 + dna1_mod[1]))))/((F1*K2*dna1_mod[71]*dna1_mod[0])/(3600*(km1 + dna1_mod[0])) - (F2*K2*dna1_mod[71]*dna1_mod[1])/(3600*(km2 + dna1_mod[1]))) - 1))/(3600*(km2 + dna1_mod[1])) + (K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/(3600*(km4 + dna1_mod[1])) + (K5*dna1_mod[74]*dna1_mod[1])/(3600*(km5 + dna1_mod[1])*(F8*dna1_mod[4] + 1)*(F9*dna1_mod[24] + 1)*(F10*dna1_mod[29] + 1)) - (K5*dna1_mod[74]*dna1_mod[3])/(3600*(km6 + dna1_mod[3])*(F11*dna1_mod[4] + 1)*(F12*dna1_mod[24] + 1)*(F13*dna1_mod[29] + 1)) - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))/(3600*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)) + CCC/DDD))/(K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1)) + 1))/((km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)))/(K1*dna1_mod[70]) + 1)))
  

  dna1_mod[124] = np.real((km40*km42*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40])/(12960000*(km43 + dna1_mod[25])*(km44 + dna1_mod[25])))**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[40]))/(2*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[40])))
  dna1_mod[125] = np.real((3600*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)*((K2*dna1_mod[71]*dna1_mod[0]*((F1*((K2*dna1_mod[71]*dna1_mod[0])/(3600*(km1 + dna1_mod[0])) - (K2*dna1_mod[71]*dna1_mod[1])/(3600*(km2 + dna1_mod[1]))))/((F1*K2*dna1_mod[71]*dna1_mod[0])/(3600*(km1 + dna1_mod[0])) - (F2*K2*dna1_mod[71]*dna1_mod[1])/(3600*(km2 + dna1_mod[1]))) - 1))/(3600*(km1 + dna1_mod[0])) - (K2*dna1_mod[71]*dna1_mod[1]*((F2*((K2*dna1_mod[71]*dna1_mod[0])/(3600*(km1 + dna1_mod[0])) - (K2*dna1_mod[71]*dna1_mod[1])/(3600*(km2 + dna1_mod[1]))))/((F1*K2*dna1_mod[71]*dna1_mod[0])/(3600*(km1 + dna1_mod[0])) - (F2*K2*dna1_mod[71]*dna1_mod[1])/(3600*(km2 + dna1_mod[1]))) - 1))/(3600*(km2 + dna1_mod[1])) + (K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/(3600*(km4 + dna1_mod[1])) + (K5*dna1_mod[74]*dna1_mod[1])/(3600*(km5 + dna1_mod[1])*(F8*dna1_mod[4] + 1)*(F9*dna1_mod[24] + 1)*(F10*dna1_mod[29] + 1)) - (K5*dna1_mod[74]*dna1_mod[3])/(3600*(km6 + dna1_mod[3])*(F11*dna1_mod[4] + 1)*(F12*dna1_mod[24] + 1)*(F13*dna1_mod[29] + 1)) - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))/(3600*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)) + (K31*dna1_mod[100]*km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40])/(12960000*(km43 + dna1_mod[25])*(km44 + dna1_mod[25])))**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[40]))/(7200*(F52*dna1_mod[40] + 1)*(km40 + (km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[40] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[40] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[40])/(12960000*(km43 + dna1_mod[25])*(km44 + dna1_mod[25])))**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[40]))/(2*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[40])))*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[40]))))/(F3*K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1)))
  
  
  DIV_1 = (3600*(km1 + dna1_mod[0]))
  DIV_2 = (3600*(km2 + dna1_mod[1]))
  DIV_5 = (3600*(km4 + dna1_mod[1]))
  DIV_6 = (3600*(km5 + dna1_mod[1])*(F8*dna1_mod[4] + 1)*(F9*dna1_mod[24] + 1)*(F10*dna1_mod[29] + 1))
  DIV_7 = (3600*(km6 + dna1_mod[3])*(F11*dna1_mod[4] + 1)*(F12*dna1_mod[24] + 1)*(F13*dna1_mod[29] + 1))
  DIV_8 = (3600*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1))
  DIV_9 = (12960000*(km43 + dna1_mod[25])*(km44 + dna1_mod[25]))
  DIV_11 = (F3*K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))
  DIV_12 = (3600*(km39 + dna1_mod[22]*dna1_mod[33])*(F51*dna1_mod[15] + 1))
  DIV_14 = (F34*K22*dna1_mod[91]*dna1_mod[15]*dna1_mod[30])
  DIV_18 = (3600*(km28 + dna1_mod[12]*dna1_mod[30]))
  DIV_19 = (3600*(km30 + dna1_mod[14]*dna1_mod[15])*(F39*dna1_mod[14] + 1)*(F38*dna1_mod[30] + 1))
  DIV_15 = (2*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[41]))

  if(DIV_1 == 0):
      DIV_1 = fixed

  if(DIV_2 == 0):
      DIV_2 = fixed

  DIV_16 = ((F1*K2*dna1_mod[71]*dna1_mod[0])/DIV_1 - (F2*K2*dna1_mod[71]*dna1_mod[1])/DIV_2)
  DIV_17 = ((F1*K2*dna1_mod[71]*dna1_mod[0])/DIV_1 - (F2*K2*dna1_mod[71]*dna1_mod[1])/DIV_2)    
 
  if(DIV_5 == 0):
      DIV_5 = fixed

  if(DIV_6 == 0):
      DIV_6 = fixed

  if(DIV_7 == 0):
      DIV_7 = fixed

  if(DIV_8 == 0):
      DIV_8 = fixed

  if(DIV_9 == 0):
      DIV_9 = fixed    

  if(DIV_11 == 0):
      DIV_11 = fixed

  if(DIV_12 == 0):
      DIV_12 = fixed

  if(DIV_16 == 0):
      DIV_16 = fixed

  if(DIV_17 == 0):
      DIV_17 = fixed

  if(DIV_14 == 0):
      DIV_14 = fixed

  if(DIV_18 == 0):
      DIV_18 = fixed

  if(DIV_19 == 0):
      DIV_19 = fixed

  if(DIV_15 == 0):
      DIV_15 = fixed

  DIV_3 = ((F1*K2*dna1_mod[71]*dna1_mod[0])/DIV_1 - (F2*K2*dna1_mod[71]*dna1_mod[1])/DIV_2)

  if(DIV_3 == 0):
      DIV_3 = fixed

  DIV_4 = ((F1*K2*dna1_mod[71]*dna1_mod[0])/DIV_1 - (F2*K2*dna1_mod[71]*dna1_mod[1])/DIV_2)

  if(DIV_4 == 0):
      DIV_4 = fixed

  ROOT_1 = ((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41])/DIV_9)

  if(ROOT_1 < 0):
      ROOT_1 = 0.01

  DIV_10 = (7200*(F52*dna1_mod[41] + 1)*(km40 + (km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*(ROOT_1)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))/DIV_15)*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[41]))

  if(DIV_10 == 0):
      DIV_10 = fixed

  DIV_13 = (3600*(km29 + dna1_mod[15]*dna1_mod[30])*((3600*F35*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)*((K2*dna1_mod[71]*dna1_mod[0]*((F1*((K2*dna1_mod[71]*dna1_mod[0])/DIV_1 - (K2*dna1_mod[71]*dna1_mod[1])/DIV_2))/DIV_16 - 1))/DIV_1 - (K2*dna1_mod[71]*dna1_mod[1]*((F2*((K2*dna1_mod[71]*dna1_mod[0])/DIV_1 - (K2*dna1_mod[71]*dna1_mod[1])/DIV_2))/DIV_17 - 1))/DIV_2 + (K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/DIV_5 + (K5*dna1_mod[74]*dna1_mod[1])/DIV_6 - (K5*dna1_mod[74]*dna1_mod[3])/DIV_7 - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))/DIV_8 + (K31*dna1_mod[100]*km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*(ROOT_1)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))/DIV_10))/DIV_11 + 1))

  if(DIV_13 == 0):
      DIV_13 = fixed
    
  dna1_mod[126] = np.real((3600*(km29 + dna1_mod[15]*dna1_mod[30])*((3600*F35*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)*((K2*dna1_mod[71]*dna1_mod[0]*((F1*((K2*dna1_mod[71]*dna1_mod[0])/DIV_1 - (K2*dna1_mod[71]*dna1_mod[1])/DIV_2))/DIV_3 - 1))/DIV_1 - (K2*dna1_mod[71]*dna1_mod[1]*((F2*((K2*dna1_mod[71]*dna1_mod[0])/DIV_1 - (K2*dna1_mod[71]*dna1_mod[1])/DIV_2))/DIV_4 - 1))/DIV_2 + (K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/DIV_5 + (K5*dna1_mod[74]*dna1_mod[1])/DIV_6 - (K5*dna1_mod[74]*dna1_mod[3])/DIV_7 - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))/DIV_8 + (K31*dna1_mod[100]*km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*(ROOT_1)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))/DIV_10))/DIV_11 + 1)*((K30*dna1_mod[99]*dna1_mod[22]*dna1_mod[33])/DIV_12 - (K22*dna1_mod[91]*dna1_mod[15]*dna1_mod[30])/DIV_13 + (K21*dna1_mod[90]*dna1_mod[12]*dna1_mod[30]*(F33*dna1_mod[14] + 1))/DIV_18 - (K23*dna1_mod[92]*dna1_mod[14]*dna1_mod[15]*(F36*dna1_mod[14] + 1)*(F37*dna1_mod[15] + 1))/DIV_19))/DIV_14)
              
  dna1_mod[127] = np.real(-((K27*dna1_mod[96]*dna1_mod[20])/(3600*(km36 + dna1_mod[20])) - 3*((K27*dna1_mod[96]*dna1_mod[19]*dna1_mod[31])/(3600*(km35 + dna1_mod[19]*dna1_mod[31]))) + (K18*dna1_mod[87]*dna1_mod[12]*dna1_mod[33]*dna1_mod[34])/(3600*(km25 + dna1_mod[12]*dna1_mod[33]*dna1_mod[34])) - (K20*dna1_mod[89]*dna1_mod[14]*dna1_mod[31]*dna1_mod[32]*dna1_mod[38])/(3600*(km27 + dna1_mod[14]*dna1_mod[31]*dna1_mod[32]*dna1_mod[38])) + (K19*dna1_mod[88]*dna1_mod[30]*dna1_mod[33]*dna1_mod[36]*dna1_mod[37])/(3600*(km26 + dna1_mod[30]*dna1_mod[33]*dna1_mod[36]*dna1_mod[37])) - (K26*dna1_mod[95]*dna1_mod[18]*dna1_mod[33]*dna1_mod[39])/(3600*(km34 + dna1_mod[18]*dna1_mod[33]*dna1_mod[39])*(F47*dna1_mod[19] + 1)*(F46*dna1_mod[30] + 1)) + (K25*dna1_mod[94]*dna1_mod[17]*dna1_mod[33]*(F42*dna1_mod[17] + 1)*(F43*dna1_mod[31] + 1))/(3600*(km33 + dna1_mod[17]*dna1_mod[33])*(F44*dna1_mod[30] + 1)*(F45*dna1_mod[32] + 1)))/((F53*K18*dna1_mod[87]*dna1_mod[12]*dna1_mod[33]*dna1_mod[34])/(3600*(km25 + dna1_mod[12]*dna1_mod[33]*dna1_mod[34])) - (F55*K26*dna1_mod[95]*dna1_mod[18]*dna1_mod[33]*dna1_mod[39])/(3600*(km34 + dna1_mod[18]*dna1_mod[33]*dna1_mod[39])*(F47*dna1_mod[19] + 1)*(F46*dna1_mod[30] + 1)) + (F54*K25*dna1_mod[94]*dna1_mod[17]*dna1_mod[33]*(F42*dna1_mod[17] + 1)*(F43*dna1_mod[31] + 1))/(3600*(km33 + dna1_mod[17]*dna1_mod[33])*(F44*dna1_mod[30] + 1)*(F45*dna1_mod[32] + 1))))
  dna1_mod[128] = np.real(-((K24*dna1_mod[93]*dna1_mod[17])/(3600*km32 + 3600*dna1_mod[17]) - (K24*dna1_mod[93]*dna1_mod[16])/(3600*km31 + 3600*dna1_mod[16]) + (K23*dna1_mod[92]*dna1_mod[14]*dna1_mod[15]*(F36*dna1_mod[14] + 1)*(F37*dna1_mod[15] + 1))/((3600*km30 + 3600*dna1_mod[14]*dna1_mod[15])*(F39*dna1_mod[14] + 1)*(F38*dna1_mod[30] + 1)))/((F40*K24*dna1_mod[93]*dna1_mod[16])/(3600*km31 + 3600*dna1_mod[16]) - (F41*K24*dna1_mod[93]*dna1_mod[17])/(3600*km32 + 3600*dna1_mod[17])))
  dna1_mod[129] = np.real(((3600*km37 + 3600*dna1_mod[20]*dna1_mod[37])*((K27*dna1_mod[96]*dna1_mod[20])/(3600*km36 + 3600*dna1_mod[20]) - (K29*dna1_mod[98]*dna1_mod[21])/(3600*km38 + 3600*dna1_mod[21]) - 3*((K27*dna1_mod[96]*dna1_mod[19]*dna1_mod[31])/(3600*km35 + 3600*dna1_mod[19]*dna1_mod[31])) + (4*K28*dna1_mod[97]*dna1_mod[20]*dna1_mod[37])/(3600*km37 + 3600*dna1_mod[20]*dna1_mod[37]) - (2*K20*dna1_mod[89]*dna1_mod[14]*dna1_mod[31]*dna1_mod[32]*dna1_mod[38])/(3600*km27 + 3600*dna1_mod[14]*dna1_mod[31]*dna1_mod[32]*dna1_mod[38]) + (2*K19*dna1_mod[88]*dna1_mod[30]*dna1_mod[33]*dna1_mod[36]*dna1_mod[37])/(3600*km26 + 3600*dna1_mod[30]*dna1_mod[33]*dna1_mod[36]*dna1_mod[37])))/(4*F48*K28*dna1_mod[97]*dna1_mod[20]*dna1_mod[37]))
  dna1_mod[130] = np.real(((KS27*dna1_mod[45])/(60*(F_S30*dna1_mod[68] + 1)) - (KS26*dna1_mod[30])/(60*(F_S29*dna1_mod[69] + 1)) + (KS5*dna1_mod[44]*dna1_mod[68])/(60*(F_S4*dna1_mod[55] + 1)) + (K10*dna1_mod[79]*dna1_mod[4])/(3600*km11 + 3600*dna1_mod[4]) + (K7*dna1_mod[76]*dna1_mod[4])/((F19*dna1_mod[5] + 1)*(3600*km8 + 3600*dna1_mod[4])) - (K10*dna1_mod[79]*dna1_mod[6]*dna1_mod[7])/(3600*km12 + 3600*dna1_mod[6]*dna1_mod[7]) - (K6*dna1_mod[75]*dna1_mod[3]*dna1_mod[30]*(F15*dna1_mod[5] + 1))/((3600*km7 + 3600*dna1_mod[3]*dna1_mod[30])*(F17*dna1_mod[16] + 1)*(F16*dna1_mod[30] + 1)))/(KS16/60 + (F18*K7*dna1_mod[76]*dna1_mod[4])/((F19*dna1_mod[5] + 1)*(3600*km8 + 3600*dna1_mod[4])) + (F14_modified*K6*dna1_mod[75]*dna1_mod[3]*dna1_mod[30]*(F15*dna1_mod[5] + 1))/((3600*km7 + 3600*dna1_mod[3]*dna1_mod[30])*(F17*dna1_mod[16] + 1)*(F16*dna1_mod[30] + 1)))) 
  
            
  A = (3600*(km1 + dna1_mod[0]))
  B = (3600*(km2 + dna1_mod[1]))
  C = (3600*(km1 + dna1_mod[0]))
  D = (3600*(km2 + dna1_mod[1]))
  E = (3600*(km1 + dna1_mod[0]))
  F = (3600*(km1 + dna1_mod[0]))
  G = (3600*(km2 + dna1_mod[1]))
  H = (3600*(km1 + dna1_mod[0]))

  if(A == 0):
      A = fixed

  if(B == 0):
      B = fixed

  if(C == 0):
      C = fixed

  if(D == 0):
      D = fixed

  if(E == 0):
      E = fixed

  if(F == 0):
      F = fixed

  if(G == 0):
      G = fixed

  if(H == 0):
      H = fixed
            
  I = (3600*(km2 + dna1_mod[1]))

  if(I == 0):
      I = fixed

  J = (3600*(km2 + dna1_mod[1]))
  K = (3600*(km4 + dna1_mod[1]))
  L = (3600*(km5 + dna1_mod[1])*(F8*dna1_mod[4] + 1)*(F9*dna1_mod[24] + 1)*(F10*dna1_mod[29] + 1))
  M = (3600*(km6 + dna1_mod[3])*(F11*dna1_mod[4] + 1)*(F12*dna1_mod[24] + 1)*(F13*dna1_mod[29] + 1))
  N = (3600*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1))
  P = ((F1*K2*dna1_mod[71]*dna1_mod[0])/H - (F2*K2*dna1_mod[71]*dna1_mod[1])/I)
  Q = (12960000*(km43 + dna1_mod[25])*(km44 + dna1_mod[25]))
  RR = (2*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[41]))

  if(J == 0):
      J = fixed

  if(K == 0):
      K = fixed

  if(L == 0):
      L = fixed

  if(M == 0):
      M = fixed

  if(N == 0):
      N = fixed

  if(P == 0):
      P = fixed

  if(Q == 0):
      Q = fixed

  if(RR == 0):
      RR = fixed

  S = ((F1*K2*dna1_mod[71]*dna1_mod[0])/H - (F2*K2*dna1_mod[71]*dna1_mod[1])/I)
  T = ((F1*K2*dna1_mod[71]*dna1_mod[0])/H - (F2*K2*dna1_mod[71]*dna1_mod[1])/I)
  U = (3600*(km5 + dna1_mod[1])*(F8*dna1_mod[4] + 1)*(F9*dna1_mod[24] + 1)*(F10*dna1_mod[29] + 1))
  V = (3600*(km6 + dna1_mod[3])*(F11*dna1_mod[4] + 1)*(F12*dna1_mod[24] + 1)*(F13*dna1_mod[29] + 1))
  W = (3600*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1))
  X = (12960000*(km43 + dna1_mod[25])*(km44 + dna1_mod[25]))

  if(S == 0):
      S = fixed

  if(T == 0):
      T = fixed

  if(U == 0):
      U = fixed

  if(V == 0):
      V = fixed

  if(W == 0):
      W = fixed

  if(X == 0):
      X = fixed           

  Y = (7200*(F52*dna1_mod[41] + 1)*(km40 + (km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41])/X)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))/RR)*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[41]))
  Z = (3600*(km6 + dna1_mod[3])*(F11*dna1_mod[4] + 1)*(F12*dna1_mod[24] + 1)*(F13*dna1_mod[29] + 1))
  AA = (3600*(km5 + dna1_mod[1])*(F8*dna1_mod[4] + 1)*(F9*dna1_mod[24] + 1)*(F10*dna1_mod[29] + 1))
  BB = (12960000*(km43 + dna1_mod[25])*(km44 + dna1_mod[25]))
  CC = (7200*(F52*dna1_mod[41] + 1)*(km40 + (km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41])/Q)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))/RR)*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[41]))

  if(Y == 0):
      Y = fixed

  if(Z == 0):
      Z = fixed

  if(AA == 0):
      AA = fixed

  if(BB == 0):
      BB = fixed

  if(CC == 0):
      CC = fixed

  DD = (3600*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1))
  EE = (3600*(km6 + dna1_mod[3])*(F11*dna1_mod[4] + 1)*(F12*dna1_mod[24] + 1)*(F13*dna1_mod[29] + 1))

  if(DD == 0):
      DD = fixed

  if(EE == 0):
      EE = fixed

  FF = (3600*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1))
  GG = (2*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[41]))
  HH = (12960000*(km43 + dna1_mod[25])*(km44 + dna1_mod[25]))

  if(FF == 0):
      FF = fixed

  if(GG == 0):
      GG = fixed

  if(HH == 0):
      HH = fixed

  II = (3600*(km30 + dna1_mod[14]*dna1_mod[15])*(F39*dna1_mod[14] + 1)*(F38*dna1_mod[30] + 1))
  JJ = (F3*K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))
  KK = (3600*(km28 + dna1_mod[12]*dna1_mod[30]))
  LL = (F3*K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))

  if(II == 0):
      II = fixed

  if(JJ == 0):
      JJ = fixed

  if(KK == 0):
      KK = fixed

  if(LL == 0):
      LL = fixed

  MM = ((F1*K2*dna1_mod[71]*dna1_mod[0])/H - (F2*K2*dna1_mod[71]*dna1_mod[1])/I)
  NN = (7200*(F52*dna1_mod[41] + 1)*(km40 + (km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41])/HH))**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))/RR)*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[41])
  OO = ((F1*K2*dna1_mod[71]*dna1_mod[0])/H - (F2*K2*dna1_mod[71]*dna1_mod[1])/I)
  PP = (3600*(km5 + dna1_mod[1])*(F8*dna1_mod[4] + 1)*(F9*dna1_mod[24] + 1)*(F10*dna1_mod[29] + 1))
  QQ = (F34*K22*dna1_mod[91]*dna1_mod[15]*dna1_mod[30])
  SS = (3600*(km21 + dna1_mod[10]))
  TT = (3600*(km22 + dna1_mod[11]))
  UU = ((F1*K2*dna1_mod[71]*dna1_mod[0])/H - (F2*K2*dna1_mod[71]*dna1_mod[1])/I)
  VV = (3600*(km39 + dna1_mod[22]*dna1_mod[33])*(F51*dna1_mod[15] + 1))
  XXX = (4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41])
  YYY = (K31*dna1_mod[100]*km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41])/HH)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))
  ZZZ = (K31*dna1_mod[100]*km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41])/HH)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))
  WWW = (K31*dna1_mod[100]*km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41])/HH)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))
  UUU = (7200*(F52*dna1_mod[41] + 1)*(km40 + (km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41])/HH)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))/GG)*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[41]))

  if(MM == 0):
      MM = fixed

  if(NN == 0):
      NN = fixed

  if(OO == 0):
      OO = fixed

  if(PP == 0):
      PP = fixed

  if(QQ == 0):
      QQ = fixed

  if(SS == 0):
      SS = fixed

  if(TT == 0):
      TT = fixed

  if(UU == 0):
      UU = fixed

  if(VV == 0):
      VV = fixed

  if(XXX == 0):
      XXX = fixed

  if(YYY == 0):
      YYY = fixed

  if(ZZZ == 0):
      ZZZ = fixed

  if(WWW == 0):
      WWW = fixed

  if(UUU == 0):
      UUU = fixed

  CD = ((F1*K2*dna1_mod[71]*dna1_mod[0])/C - (F2*K2*dna1_mod[71]*dna1_mod[1])/D)

  if(CD == 0):
      CD = fixed   

  VVV = (3600*(km29 + dna1_mod[15]*dna1_mod[30])*((3600*F35*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)*((K2*dna1_mod[71]*dna1_mod[0]*((F1*((K2*dna1_mod[71]*dna1_mod[0])/H - (K2*dna1_mod[71]*dna1_mod[1])/I))/((F1*K2*dna1_mod[71]*dna1_mod[0])/H - (F2*K2*dna1_mod[71]*dna1_mod[1])/I) - 1))/H - (K2*dna1_mod[71]*dna1_mod[1]*((F2*((K2*dna1_mod[71]*dna1_mod[0])/H - (K2*dna1_mod[71]*dna1_mod[1])/I))/MM - 1))/I + (K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/(3600*(km4 + dna1_mod[1])) + (K5*dna1_mod[74]*dna1_mod[1])/AA - (K5*dna1_mod[74]*dna1_mod[3])/Z - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))/DD + YYY/UUU))/JJ + 1))

  if(VVV == 0):
      VVV = fixed

  XXXX = (7200*(F52*dna1_mod[41] + 1)*(km40 + (km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41])/HH)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))/RR)*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[41]))

  if(XXXX == 0):
      XXXX = fixed


  YYYY = (3600*(km29 + dna1_mod[15]*dna1_mod[30])*((3600*F35*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)*((K2*dna1_mod[71]*dna1_mod[0]*((F1*((K2*dna1_mod[71]*dna1_mod[0])/H - (K2*dna1_mod[71]*dna1_mod[1])/I))/UU - 1))/H - (K2*dna1_mod[71]*dna1_mod[1]*((F2*((K2*dna1_mod[71]*dna1_mod[0])/H - (K2*dna1_mod[71]*dna1_mod[1])/I))/OO - 1))/I + (K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/(3600*(km4 + dna1_mod[1])) + (K5*dna1_mod[74]*dna1_mod[1])/PP - (K5*dna1_mod[74]*dna1_mod[3])/EE - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))/(3600*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)) + (K31*dna1_mod[100]*km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*(XXX/HH)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))/(7200*(F52*dna1_mod[41] + 1)*(km40 + (km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41])/HH)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))/RR)*(2*K31*dna1_mod[100]*km42**2*dna1_mod[1]**2 + K33*dna1_mod[102]*km40**2*dna1_mod[24]**2 + F52*K33*dna1_mod[102]*km40**2*dna1_mod[24]**2*dna1_mod[41]))))/LL + 1))

  if(YYYY == 0):
      YYYY = fixed

  BBBB = (3600*(km29 + dna1_mod[15]*dna1_mod[30])*((3600*F35*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)*((K2*dna1_mod[71]*dna1_mod[0]*((F1*((K2*dna1_mod[71]*dna1_mod[0])/H - (K2*dna1_mod[71]*dna1_mod[1])/I))/UU - 1))/H - (K2*dna1_mod[71]*dna1_mod[1]*((F2*((K2*dna1_mod[71]*dna1_mod[0])/H - (K2*dna1_mod[71]*dna1_mod[1])/I))/UU - 1))/I + (K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/K + (K5*dna1_mod[74]*dna1_mod[1])/PP - (K5*dna1_mod[74]*dna1_mod[3])/EE - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))/FF + (K31*dna1_mod[100]*km40*km42*dna1_mod[1]*(2*K31*dna1_mod[100]*km42*dna1_mod[1] - 3600*((4*K31**2*dna1_mod[100]**2*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*km44*dna1_mod[1]**2 + K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2 + 4*K31**2*dna1_mod[100]**2*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K31**2*dna1_mod[100]**2*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2 + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 8*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25] - 8*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25] + 4*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25] - 4*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25] + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + F52**2*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*km44*dna1_mod[24]**2*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 2*F52*K33**2*dna1_mod[102]**2*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2 + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*dna1_mod[1]**2*dna1_mod[25]**2*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*dna1_mod[24]**2*dna1_mod[25]**2*dna1_mod[41] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25] + 4*F52**2*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 - 4*F52**2*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41]**2 + 8*F52*K31*K34*dna1_mod[100]*dna1_mod[103]*km42**2*km44*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K31*K35*dna1_mod[100]*dna1_mod[104]*km42**2*km43*dna1_mod[1]**2*dna1_mod[25]*dna1_mod[41] + 8*F52*K33*K34*dna1_mod[102]*dna1_mod[103]*km40**2*km44*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] - 8*F52*K33*K35*dna1_mod[102]*dna1_mod[104]*km40**2*km43*dna1_mod[24]**2*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]**2*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km43*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41] + 4*F52*K31*K33*dna1_mod[100]*dna1_mod[102]*km40*km42*km44*dna1_mod[1]*dna1_mod[24]*dna1_mod[25]*dna1_mod[41])/HH)**(1/2) + K33*dna1_mod[102]*km40*dna1_mod[24] + F52*K33*dna1_mod[102]*km40*dna1_mod[24]*dna1_mod[41]))/XXXX))/LL + 1))

  if(BBBB == 0):
      BBBB = fixed


  AAAA = (K22*dna1_mod[91]*dna1_mod[15]*dna1_mod[30])

  if(AAAA == 0):
      AAAA = fixed

  WWWW = (3600*(km23 + dna1_mod[11]*dna1_mod[31])*(F29*dna1_mod[14] + 1)*(F28*dna1_mod[30] + 1)*(F32*dna1_mod[36] + 1)*((3600*F31*(km29 + dna1_mod[15]*dna1_mod[30])*((3600*F35*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)*((K2*dna1_mod[71]*dna1_mod[0]*((F1*((K2*dna1_mod[71]*dna1_mod[0])/H - (K2*dna1_mod[71]*dna1_mod[1])/I))/S - 1))/H - (K2*dna1_mod[71]*dna1_mod[1]*((F2*((K2*dna1_mod[71]*dna1_mod[0])/H - (K2*dna1_mod[71]*dna1_mod[1])/I))/T - 1))/I + (K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/K + (K5*dna1_mod[74]*dna1_mod[1])/U - (K5*dna1_mod[74]*dna1_mod[3])/V - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))/W + WWW/Y))/LL + 1)*((K30*dna1_mod[99]*dna1_mod[22]*dna1_mod[33])/VV - AAAA/VVV + (K21*dna1_mod[90]*dna1_mod[12]*dna1_mod[30]*(F33*dna1_mod[14] + 1))/KK - (K23*dna1_mod[92]*dna1_mod[14]*dna1_mod[15]*(F36*dna1_mod[14] + 1)*(F37*dna1_mod[15] + 1))/II))/QQ + 1)*((K15*dna1_mod[84]*dna1_mod[10])/SS - (K15*dna1_mod[84]*dna1_mod[11])/TT + (K22*dna1_mod[91]*dna1_mod[15]*dna1_mod[30]*((3600*(km29 + dna1_mod[15]*dna1_mod[30])*((3600*F35*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)*((K2*dna1_mod[71]*dna1_mod[0]*((F1*((K2*dna1_mod[71]*dna1_mod[0])/H - (K2*dna1_mod[71]*dna1_mod[1])/I))/UU - 1))/H - (K2*dna1_mod[71]*dna1_mod[1]*((F2*((K2*dna1_mod[71]*dna1_mod[0])/H - (K2*dna1_mod[71]*dna1_mod[1])/I))/UU - 1))/I + (K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/K + (K5*dna1_mod[74]*dna1_mod[1])/PP - (K5*dna1_mod[74]*dna1_mod[3])/EE - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))/FF + YYY/NN))/LL + 1)*((K30*dna1_mod[99]*dna1_mod[22]*dna1_mod[33])/VV - AAAA/YYYY + (K21*dna1_mod[90]*dna1_mod[12]*dna1_mod[30]*(F33*dna1_mod[14] + 1))/KK - (K23*dna1_mod[92]*dna1_mod[14]*dna1_mod[15]*(F36*dna1_mod[14] + 1)*(F37*dna1_mod[15] + 1))/II))/AAAA + 1))/BBBB))

  if(WWWW == 0):
      WWWW = fixed

  dna1_mod[131] = np.real(((K16*dna1_mod[85]*dna1_mod[11]*dna1_mod[31]*(F27*dna1_mod[4] + 1)*((3600*F26*(km3 + dna1_mod[2]*dna1_mod[30])*(F5*dna1_mod[1] + 1)*(F6*dna1_mod[14] + 1)*((K2*dna1_mod[71]*dna1_mod[0]*((F1*((K2*dna1_mod[71]*dna1_mod[0])/A - (K2*dna1_mod[71]*dna1_mod[1])/B))/CD - 1))/E - (K2*dna1_mod[71]*dna1_mod[1]*((F2*((K2*dna1_mod[71]*dna1_mod[0])/F - (K2*dna1_mod[71]*dna1_mod[1])/G))/P - 1))/J + (K4*dna1_mod[73]*dna1_mod[1]*(F7*dna1_mod[1] + 1))/K + (K5*dna1_mod[74]*dna1_mod[1])/L - (K5*dna1_mod[74]*dna1_mod[3])/M - (K3*dna1_mod[72]*dna1_mod[2]*dna1_mod[30]*(F4*dna1_mod[31] + 1))/N + ZZZ/CC))/LL + 1))/WWWW - 1)/F30)
  
  dna1_mod[132] = np.real(-((K2*dna1_mod[71]*dna1_mod[0])/H - (K2*dna1_mod[71]*dna1_mod[1])/I)/UU)
  dna1_mod[133] = np.real(-((K13*dna1_mod[82]*dna1_mod[9])/(3600*(km18 + dna1_mod[9])) + (K14*dna1_mod[83]*dna1_mod[9])/(1800*(km19 + dna1_mod[9])) - (K14*dna1_mod[83]*dna1_mod[10])/(1800*(km20 + dna1_mod[10])) - (K15*dna1_mod[84]*dna1_mod[10])/(3600*(km21 + dna1_mod[10])) + (K15*dna1_mod[84]*dna1_mod[11])/(3600*(km22 + dna1_mod[11])) - (K13*dna1_mod[82]*dna1_mod[8]*dna1_mod[31])/(3600*(km17 + dna1_mod[8]*dna1_mod[31])))/((F24*K14*dna1_mod[83]*dna1_mod[9])/(1800*(km19 + dna1_mod[9])) - (F25*K14*dna1_mod[83]*dna1_mod[10])/(1800*(km20 + dna1_mod[10]))))


  DNA_SIZE = 134
  for i in range(0,DNA_SIZE):
      
      if(dna1_mod[i] < 0):
          dna1_mod[i] = 0.001
            
      if(dna1_mod[i] > 1):
          dna1_mod[i] = 0.999


  constant_terms_in_u18_design = ((KS7*dna2_mod[43])/60 + (KS29*dna2_mod[43])/60 + KS13/(60*(F_S15*dna2_mod[43] + 1)*(F_S14*dna2_mod[58] + 1)) + (KS9*dna2_mod[47])/(60*(F_S8*dna2_mod[43] + 1)*(F_S9*dna2_mod[46] + 1)) - (KS36*(kbg16 - lg16 + kg16*dna2_mod[48]))/(60*FG6*kg16*dna2_mod[48]) + (KS8*dna2_mod[41]*dna2_mod[42]*dna2_mod[43]*dna2_mod[46]*dna2_mod[49])/(60*(F_S6*dna2_mod[56] + 1)*(F_S7*dna2_mod[57] + 1)))
  dna2_mod[107] = np.real(-(60*((KS31*(kbg16 - lg16 + kg16*dna2_mod[48]))/(60*FG6*kg16*dna2_mod[48]) - (KS8*dna2_mod[41]*dna2_mod[42]*dna2_mod[43]*dna2_mod[46]*dna2_mod[49])/(60*(F_S6*dna2_mod[56] + 1)*(F_S7*dna2_mod[57] + 1))))/KS1)
  dna2_mod[108] = np.real(((kg28*dna2_mod[45]*dna2_mod[64]*dna2_mod[66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna2_mod[63] + 60) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))))/((kbg28 - lg28)*(kg30*dna2_mod[43] + kg9*dna2_mod[43]*dna2_mod[66] + kg27*dna2_mod[43]*dna2_mod[66] + kg37*dna2_mod[41]*dna2_mod[66] + kg29*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + kg21*dna2_mod[41]*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + (kg20*dna2_mod[43]*dna2_mod[45]*dna2_mod[66])/(FG9*dna2_mod[58] + 1) + (F_S22*KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60)))) - 1)/FG13)
  dna2_mod[109] = np.real(-(60*((KS29*dna2_mod[43])/60 + (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/(60*(F_S21*dna2_mod[50] + 1)*(F_S20*dna2_mod[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna2_mod[63] + 60) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))))/(kg30*dna2_mod[43] + kg9*dna2_mod[43]*dna2_mod[66] + kg27*dna2_mod[43]*dna2_mod[66] + kg37*dna2_mod[41]*dna2_mod[66] + kg29*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + kg21*dna2_mod[41]*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + (kg20*dna2_mod[43]*dna2_mod[45]*dna2_mod[66])/(FG9*dna2_mod[58] + 1) + (F_S22*KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))) - 1))))/KS54)
  dna2_mod[110] = np.real(-((KS49*((KS33*((KS29*dna2_mod[43])/60 + (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/(60*(F_S21*dna2_mod[50] + 1)*(F_S20*dna2_mod[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna2_mod[63] + 60) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))))/(kg30*dna2_mod[43] + kg9*dna2_mod[43]*dna2_mod[66] + kg27*dna2_mod[43]*dna2_mod[66] + kg37*dna2_mod[41]*dna2_mod[66] + kg29*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + kg21*dna2_mod[41]*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + (kg20*dna2_mod[43]*dna2_mod[45]*dna2_mod[66])/(FG9*dna2_mod[58] + 1) + (F_S22*KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))) - 1))))/KS54 - (KS32*((kg28*dna2_mod[45]*dna2_mod[64]*dna2_mod[66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna2_mod[63] + 60) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))))/((kbg28 - lg28)*(kg30*dna2_mod[43] + kg9*dna2_mod[43]*dna2_mod[66] + kg27*dna2_mod[43]*dna2_mod[66] + kg37*dna2_mod[41]*dna2_mod[66] + kg29*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + kg21*dna2_mod[41]*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + (kg20*dna2_mod[43]*dna2_mod[45]*dna2_mod[66])/(FG9*dna2_mod[58] + 1) + (F_S22*KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60)))) - 1))/(60*FG13) - (KS2*dna2_mod[44]*dna2_mod[51])/60 + (KS17*dna2_mod[42])/(60*(F_S17*dna2_mod[44] + 1)*(F_S18*dna2_mod[45] + 1)) + (KS8*dna2_mod[41]*dna2_mod[42]*dna2_mod[43]*dna2_mod[46]*dna2_mod[49])/(60*(F_S6*dna2_mod[56] + 1)*(F_S7*dna2_mod[57] + 1))))/KS35 - (KS2*dna2_mod[44]*dna2_mod[51])/60)/(KS48/60 - (KS34*KS49)/(60*KS35)))
  dna2_mod[111] = np.real((60*((KS2*dna2_mod[44]*dna2_mod[51])/60 + (KS48*((KS49*((KS33*((KS29*dna2_mod[43])/60 + (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/(60*(F_S21*dna2_mod[50] + 1)*(F_S20*dna2_mod[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna2_mod[63] + 60) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))))/(kg30*dna2_mod[43] + kg9*dna2_mod[43]*dna2_mod[66] + kg27*dna2_mod[43]*dna2_mod[66] + kg37*dna2_mod[41]*dna2_mod[66] + kg29*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + kg21*dna2_mod[41]*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + (kg20*dna2_mod[43]*dna2_mod[45]*dna2_mod[66])/(FG9*dna2_mod[58] + 1) + (F_S22*KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))) - 1))))/KS54 - (KS32*((kg28*dna2_mod[45]*dna2_mod[64]*dna2_mod[66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna2_mod[63] + 60) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))))/((kbg28 - lg28)*(kg30*dna2_mod[43] + kg9*dna2_mod[43]*dna2_mod[66] + kg27*dna2_mod[43]*dna2_mod[66] + kg37*dna2_mod[41]*dna2_mod[66] + kg29*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + kg21*dna2_mod[41]*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + (kg20*dna2_mod[43]*dna2_mod[45]*dna2_mod[66])/(FG9*dna2_mod[58] + 1) + (F_S22*KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60)))) - 1))/(60*FG13) - (KS2*dna2_mod[44]*dna2_mod[51])/60 + (KS17*dna2_mod[42])/(60*(F_S17*dna2_mod[44] + 1)*(F_S18*dna2_mod[45] + 1)) + (KS8*dna2_mod[41]*dna2_mod[42]*dna2_mod[43]*dna2_mod[46]*dna2_mod[49])/(60*(F_S6*dna2_mod[56] + 1)*(F_S7*dna2_mod[57] + 1))))/KS35 - (KS2*dna2_mod[44]*dna2_mod[51])/60))/(60*(KS48/60 - (KS34*KS49)/(60*KS35)))))/KS49)
  dna2_mod[112] = np.real((kbg16 - lg16 + kg16*dna2_mod[48])/(FG6*kg16*dna2_mod[48]))
  dna2_mod[113] = np.real((60*((KS2*dna2_mod[44]*dna2_mod[51])/60 - (KS4*dna2_mod[50])/(60*(F_S3*dna2_mod[55] + 1)) + (KS18*dna2_mod[44])/(60*(F_S19*dna2_mod[45] + 1)) + (KS5*dna2_mod[44]*dna2_mod[68])/(60*(F_S4*dna2_mod[55] + 1)) + (KS14*dna2_mod[44]*dna2_mod[48])/(60*(F_S16*dna2_mod[55] + 1)) + (KS17*dna2_mod[42])/(60*(F_S17*dna2_mod[44] + 1)*(F_S18*dna2_mod[45] + 1)) - (KS37*(kbg16 - lg16 + kg16*dna2_mod[48]))/(60*FG6*kg16*dna2_mod[48]) - (KS52*(kbg16 - lg16 + kg16*dna2_mod[48]))/(60*FG6*kg16*dna2_mod[48]) + (KS6*dna2_mod[44]*dna2_mod[45]*dna2_mod[53]*dna2_mod[54])/(60*((60*(KS24/(60*F_S28*dna2_mod[53] + 60) - (KS28*dna2_mod[30]*dna2_mod[69])/60 + KS23/((F_S26*dna2_mod[21] + 1)*(60*F_S25*dna2_mod[20] + 60)*(F_S27*dna2_mod[53] + 1)) + (KS6*dna2_mod[44]*dna2_mod[45]*dna2_mod[53]*dna2_mod[54])/60))/(KS6*dna2_mod[44]*dna2_mod[45]*dna2_mod[53]*dna2_mod[54]) + 1))))/KS38)
  dna2_mod[114] = np.real(-((K9*dna2_mod[78]*dna2_mod[5])/((F23*dna2_mod[3] + 1)*(3600*km10 + 3600*dna2_mod[5])) - (K8*dna2_mod[77]*dna2_mod[3]*dna2_mod[30]*(F20*dna2_mod[3] + 1))/(3600*km9 + 3600*dna2_mod[3]*dna2_mod[30]))/((F22*K9*dna2_mod[78]*dna2_mod[5])/((F23*dna2_mod[3] + 1)*(3600*km10 + 3600*dna2_mod[5])) + (F21*K8*dna2_mod[77]*dna2_mod[3]*dna2_mod[30]*(F20*dna2_mod[3] + 1))/(3600*km9 + 3600*dna2_mod[3]*dna2_mod[30])))
  dna2_mod[115] = np.real((60*(KS24/(60*F_S28*dna2_mod[53] + 60) - (KS28*dna2_mod[30]*dna2_mod[69])/60 + KS23/((F_S26*dna2_mod[21] + 1)*(60*F_S25*dna2_mod[20] + 60)*(F_S27*dna2_mod[53] + 1)) + (KS6*dna2_mod[44]*dna2_mod[45]*dna2_mod[53]*dna2_mod[54])/60))/(F_S5*KS6*dna2_mod[44]*dna2_mod[45]*dna2_mod[53]*dna2_mod[54]))
  dna2_mod[116] = np.real(-(kbg23 - lg23)/kg23)
  dna2_mod[117] = np.real((60*((KS47*((KS49*((KS33*((KS29*dna2_mod[43])/60 + (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/(60*(F_S21*dna2_mod[50] + 1)*(F_S20*dna2_mod[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna2_mod[63] + 60) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))))/(kg30*dna2_mod[43] + kg9*dna2_mod[43]*dna2_mod[66] + kg27*dna2_mod[43]*dna2_mod[66] + kg37*dna2_mod[41]*dna2_mod[66] + kg29*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + kg21*dna2_mod[41]*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + (kg20*dna2_mod[43]*dna2_mod[45]*dna2_mod[66])/(FG9*dna2_mod[58] + 1) + (F_S22*KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))) - 1))))/KS54 - (KS32*((kg28*dna2_mod[45]*dna2_mod[64]*dna2_mod[66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna2_mod[63] + 60) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))))/((kbg28 - lg28)*(kg30*dna2_mod[43] + kg9*dna2_mod[43]*dna2_mod[66] + kg27*dna2_mod[43]*dna2_mod[66] + kg37*dna2_mod[41]*dna2_mod[66] + kg29*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + kg21*dna2_mod[41]*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + (kg20*dna2_mod[43]*dna2_mod[45]*dna2_mod[66])/(FG9*dna2_mod[58] + 1) + (F_S22*KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60)))) - 1))/(60*FG13) - (KS2*dna2_mod[44]*dna2_mod[51])/60 + (KS17*dna2_mod[42])/(60*(F_S17*dna2_mod[44] + 1)*(F_S18*dna2_mod[45] + 1)) + (KS8*dna2_mod[41]*dna2_mod[42]*dna2_mod[43]*dna2_mod[46]*dna2_mod[49])/(60*(F_S6*dna2_mod[56] + 1)*(F_S7*dna2_mod[57] + 1))))/KS35 - (KS2*dna2_mod[44]*dna2_mod[51])/60))/(60*(KS48/60 - (KS34*KS49)/(60*KS35))) - KS13/(60*(F_S15*dna2_mod[43] + 1)*(F_S14*dna2_mod[58] + 1)) + (KS6*dna2_mod[44]*dna2_mod[45]*dna2_mod[53]*dna2_mod[54])/(60*((60*(KS24/(60*F_S28*dna2_mod[53] + 60) - (KS28*dna2_mod[30]*dna2_mod[69])/60 + KS23/((F_S26*dna2_mod[21] + 1)*(60*F_S25*dna2_mod[20] + 60)*(F_S27*dna2_mod[53] + 1)) + (KS6*dna2_mod[44]*dna2_mod[45]*dna2_mod[53]*dna2_mod[54])/60))/(KS6*dna2_mod[44]*dna2_mod[45]*dna2_mod[53]*dna2_mod[54]) + 1))))/KS46)
  dna2_mod[118] = np.real((60*((KS22*dna2_mod[64])/60 + (KS25*dna2_mod[64])/60 - (KS20*dna2_mod[48]*dna2_mod[58])/(60*((60*F_S11*((KS2*dna2_mod[44]*dna2_mod[51])/60 - (KS4*dna2_mod[50])/(60*(F_S3*dna2_mod[55] + 1)) + (KS18*dna2_mod[44])/(60*(F_S19*dna2_mod[45] + 1)) + (KS5*dna2_mod[44]*dna2_mod[68])/(60*(F_S4*dna2_mod[55] + 1)) + (KS14*dna2_mod[44]*dna2_mod[48])/(60*(F_S16*dna2_mod[55] + 1)) + (KS17*dna2_mod[42])/(60*(F_S17*dna2_mod[44] + 1)*(F_S18*dna2_mod[45] + 1)) - (KS37*(kbg16 - lg16 + kg16*dna2_mod[48]))/(60*FG6*kg16*dna2_mod[48]) - (KS52*(kbg16 - lg16 + kg16*dna2_mod[48]))/(60*FG6*kg16*dna2_mod[48]) + (KS6*dna2_mod[44]*dna2_mod[45]*dna2_mod[53]*dna2_mod[54])/(60*((60*(KS24/(60*F_S28*dna2_mod[53] + 60) - (KS28*dna2_mod[30]*dna2_mod[69])/60 + KS23/((F_S26*dna2_mod[21] + 1)*(60*F_S25*dna2_mod[20] + 60)*(F_S27*dna2_mod[53] + 1)) + (KS6*dna2_mod[44]*dna2_mod[45]*dna2_mod[53]*dna2_mod[54])/60))/(KS6*dna2_mod[44]*dna2_mod[45]*dna2_mod[53]*dna2_mod[54]) + 1))))/KS38 + 1)*(F_S23*dna2_mod[63] + 1)) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/(60*(F_S21*dna2_mod[50] + 1)*(F_S20*dna2_mod[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna2_mod[63] + 60) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))))/(kg30*dna2_mod[43] + kg9*dna2_mod[43]*dna2_mod[66] + kg27*dna2_mod[43]*dna2_mod[66] + kg37*dna2_mod[41]*dna2_mod[66] + kg29*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + kg21*dna2_mod[41]*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + (kg20*dna2_mod[43]*dna2_mod[45]*dna2_mod[66])/(FG9*dna2_mod[58] + 1) + (F_S22*KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))) - 1))))/KS50)
  dna2_mod[119] = np.real(((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*dna2_mod[50] + 60*F_S13*constant_terms_in_u18_design*dna2_mod[45]))/(15*(F_S13*dna2_mod[45] + 1)))**(1/2)/(2*F_S1*constant_terms_in_u18_design))
  dna2_mod[120] = np.real(((60*KS3*dna2_mod[52]*(F_S13*dna2_mod[45] + 1))/(KS11*((30*((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*dna2_mod[50] + 60*F_S13*constant_terms_in_u18_design*dna2_mod[45]))/(15*(F_S13*dna2_mod[45] + 1)))**(1/2))/constant_terms_in_u18_design + 60)) - 1)/F_S2)
  dna2_mod[121] = np.real((KS10/((KS43*((K9*dna2_mod[78]*dna2_mod[5])/((F23*dna2_mod[3] + 1)*(3600*km10 + 3600*dna2_mod[5])) - (K8*dna2_mod[77]*dna2_mod[3]*dna2_mod[30]*(F20*dna2_mod[3] + 1))/(3600*km9 + 3600*dna2_mod[3]*dna2_mod[30])))/((F22*K9*dna2_mod[78]*dna2_mod[5])/((F23*dna2_mod[3] + 1)*(3600*km10 + 3600*dna2_mod[5])) + (F21*K8*dna2_mod[77]*dna2_mod[3]*dna2_mod[30]*(F20*dna2_mod[3] + 1))/(3600*km9 + 3600*dna2_mod[3]*dna2_mod[30])) + (KS4*dna2_mod[50])/(F_S3*dna2_mod[55] + 1) + (KS11*dna2_mod[50])/(F_S13*dna2_mod[45] + 1) - (KS39*(kbg16 - lg16 + kg16*dna2_mod[48]))/(FG6*kg16*dna2_mod[48]) + (KS11*dna2_mod[50]*((30*((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*dna2_mod[50] + 60*F_S13*constant_terms_in_u18_design*dna2_mod[45]))/(15*(F_S13*dna2_mod[45] + 1)))**(1/2))/constant_terms_in_u18_design + 60))/(60*(((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*dna2_mod[50] + 60*F_S13*constant_terms_in_u18_design*dna2_mod[45]))/(15*(F_S13*dna2_mod[45] + 1)))**(1/2)/(2*constant_terms_in_u18_design) + 1)*(F_S13*dna2_mod[45] + 1)) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S21*dna2_mod[50] + 1)*(F_S20*dna2_mod[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna2_mod[63] + 60) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))))/(kg30*dna2_mod[43] + kg9*dna2_mod[43]*dna2_mod[66] + kg27*dna2_mod[43]*dna2_mod[66] + kg37*dna2_mod[41]*dna2_mod[66] + kg29*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + kg21*dna2_mod[41]*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + (kg20*dna2_mod[43]*dna2_mod[45]*dna2_mod[66])/(FG9*dna2_mod[58] + 1) + (F_S22*KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60))) - 1))) - 1)/F_S10)
  dna2_mod[122] = np.real(-(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna2_mod[63] + 60) - (KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60)))/(kg30*dna2_mod[43] + kg9*dna2_mod[43]*dna2_mod[66] + kg27*dna2_mod[43]*dna2_mod[66] + kg37*dna2_mod[41]*dna2_mod[66] + kg29*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + kg21*dna2_mod[41]*dna2_mod[42]*dna2_mod[64]*dna2_mod[66] + (kg20*dna2_mod[43]*dna2_mod[45]*dna2_mod[66])/(FG9*dna2_mod[58] + 1) + (F_S22*KS19*dna2_mod[45]*dna2_mod[64]*dna2_mod[65])/((F_S20*dna2_mod[66] + 1)*(60*F_S21*dna2_mod[50] + 60)))) 
  
  AAA = (K31*dna2_mod[100]*km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40])/(12960000*(km43 + dna2_mod[25])*(km44 + dna2_mod[25])))**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[40]))
  BBB = (7200*(F52*dna2_mod[40] + 1)*(km40 + (km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40])/(12960000*(km43 + dna2_mod[25])*(km44 + dna2_mod[25])))**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[40]))/(2*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[40])))*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[40]))
  CCC = (K31*dna2_mod[100]*km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40])/(12960000*(km43 + dna2_mod[25])*(km44 + dna2_mod[25])))**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[40]))
  DDD = (7200*(F52*dna2_mod[40] + 1)*(km40 + (km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40])/(12960000*(km43 + dna2_mod[25])*(km44 + dna2_mod[25])))**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[40]))/(2*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[40])))*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[40]))

  dna2_mod[123] = np.real(-(km0*((K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/(km4 + dna2_mod[1]) - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1)*((3600*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)*((K2*dna2_mod[71]*dna2_mod[0]*((F1*((K2*dna2_mod[71]*dna2_mod[0])/(3600*(km1 + dna2_mod[0])) - (K2*dna2_mod[71]*dna2_mod[1])/(3600*(km2 + dna2_mod[1]))))/((F1*K2*dna2_mod[71]*dna2_mod[0])/(3600*(km1 + dna2_mod[0])) - (F2*K2*dna2_mod[71]*dna2_mod[1])/(3600*(km2 + dna2_mod[1]))) - 1))/(3600*(km1 + dna2_mod[0])) - (K2*dna2_mod[71]*dna2_mod[1]*((F2*((K2*dna2_mod[71]*dna2_mod[0])/(3600*(km1 + dna2_mod[0])) - (K2*dna2_mod[71]*dna2_mod[1])/(3600*(km2 + dna2_mod[1]))))/((F1*K2*dna2_mod[71]*dna2_mod[0])/(3600*(km1 + dna2_mod[0])) - (F2*K2*dna2_mod[71]*dna2_mod[1])/(3600*(km2 + dna2_mod[1]))) - 1))/(3600*(km2 + dna2_mod[1])) + (K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/(3600*(km4 + dna2_mod[1])) + (K5*dna2_mod[74]*dna2_mod[1])/(3600*(km5 + dna2_mod[1])*(F8*dna2_mod[4] + 1)*(F9*dna2_mod[24] + 1)*(F10*dna2_mod[29] + 1)) - (K5*dna2_mod[74]*dna2_mod[3])/(3600*(km6 + dna2_mod[3])*(F11*dna2_mod[4] + 1)*(F12*dna2_mod[24] + 1)*(F13*dna2_mod[29] + 1)) - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))/(3600*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)) + AAA/BBB))/(K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1)) + 1))/((km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1))))/(K1*dna2_mod[70]*(((K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/(km4 + dna2_mod[1]) - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1)*((3600*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)*((K2*dna2_mod[71]*dna2_mod[0]*((F1*((K2*dna2_mod[71]*dna2_mod[0])/(3600*(km1 + dna2_mod[0])) - (K2*dna2_mod[71]*dna2_mod[1])/(3600*(km2 + dna2_mod[1]))))/((F1*K2*dna2_mod[71]*dna2_mod[0])/(3600*(km1 + dna2_mod[0])) - (F2*K2*dna2_mod[71]*dna2_mod[1])/(3600*(km2 + dna2_mod[1]))) - 1))/(3600*(km1 + dna2_mod[0])) - (K2*dna2_mod[71]*dna2_mod[1]*((F2*((K2*dna2_mod[71]*dna2_mod[0])/(3600*(km1 + dna2_mod[0])) - (K2*dna2_mod[71]*dna2_mod[1])/(3600*(km2 + dna2_mod[1]))))/((F1*K2*dna2_mod[71]*dna2_mod[0])/(3600*(km1 + dna2_mod[0])) - (F2*K2*dna2_mod[71]*dna2_mod[1])/(3600*(km2 + dna2_mod[1]))) - 1))/(3600*(km2 + dna2_mod[1])) + (K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/(3600*(km4 + dna2_mod[1])) + (K5*dna2_mod[74]*dna2_mod[1])/(3600*(km5 + dna2_mod[1])*(F8*dna2_mod[4] + 1)*(F9*dna2_mod[24] + 1)*(F10*dna2_mod[29] + 1)) - (K5*dna2_mod[74]*dna2_mod[3])/(3600*(km6 + dna2_mod[3])*(F11*dna2_mod[4] + 1)*(F12*dna2_mod[24] + 1)*(F13*dna2_mod[29] + 1)) - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))/(3600*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)) + CCC/DDD))/(K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1)) + 1))/((km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)))/(K1*dna2_mod[70]) + 1)))
  
  dna2_mod[124] = np.real((km40*km42*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40])/(12960000*(km43 + dna2_mod[25])*(km44 + dna2_mod[25])))**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[40]))/(2*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[40])))
  dna2_mod[125] = np.real((3600*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)*((K2*dna2_mod[71]*dna2_mod[0]*((F1*((K2*dna2_mod[71]*dna2_mod[0])/(3600*(km1 + dna2_mod[0])) - (K2*dna2_mod[71]*dna2_mod[1])/(3600*(km2 + dna2_mod[1]))))/((F1*K2*dna2_mod[71]*dna2_mod[0])/(3600*(km1 + dna2_mod[0])) - (F2*K2*dna2_mod[71]*dna2_mod[1])/(3600*(km2 + dna2_mod[1]))) - 1))/(3600*(km1 + dna2_mod[0])) - (K2*dna2_mod[71]*dna2_mod[1]*((F2*((K2*dna2_mod[71]*dna2_mod[0])/(3600*(km1 + dna2_mod[0])) - (K2*dna2_mod[71]*dna2_mod[1])/(3600*(km2 + dna2_mod[1]))))/((F1*K2*dna2_mod[71]*dna2_mod[0])/(3600*(km1 + dna2_mod[0])) - (F2*K2*dna2_mod[71]*dna2_mod[1])/(3600*(km2 + dna2_mod[1]))) - 1))/(3600*(km2 + dna2_mod[1])) + (K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/(3600*(km4 + dna2_mod[1])) + (K5*dna2_mod[74]*dna2_mod[1])/(3600*(km5 + dna2_mod[1])*(F8*dna2_mod[4] + 1)*(F9*dna2_mod[24] + 1)*(F10*dna2_mod[29] + 1)) - (K5*dna2_mod[74]*dna2_mod[3])/(3600*(km6 + dna2_mod[3])*(F11*dna2_mod[4] + 1)*(F12*dna2_mod[24] + 1)*(F13*dna2_mod[29] + 1)) - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))/(3600*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)) + (K31*dna2_mod[100]*km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40])/(12960000*(km43 + dna2_mod[25])*(km44 + dna2_mod[25])))**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[40]))/(7200*(F52*dna2_mod[40] + 1)*(km40 + (km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[40] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[40] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[40])/(12960000*(km43 + dna2_mod[25])*(km44 + dna2_mod[25])))**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[40]))/(2*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[40])))*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[40]))))/(F3*K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1)))
  
              
  DIV_1 = (3600*(km1 + dna2_mod[0]))
  DIV_2 = (3600*(km2 + dna2_mod[1]))
  DIV_5 = (3600*(km4 + dna2_mod[1]))
  DIV_6 = (3600*(km5 + dna2_mod[1])*(F8*dna2_mod[4] + 1)*(F9*dna2_mod[24] + 1)*(F10*dna2_mod[29] + 1))
  DIV_7 = (3600*(km6 + dna2_mod[3])*(F11*dna2_mod[4] + 1)*(F12*dna2_mod[24] + 1)*(F13*dna2_mod[29] + 1))
  DIV_8 = (3600*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1))
  DIV_9 = (12960000*(km43 + dna2_mod[25])*(km44 + dna2_mod[25]))
  DIV_11 = (F3*K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))
  DIV_12 = (3600*(km39 + dna2_mod[22]*dna2_mod[33])*(F51*dna2_mod[15] + 1))
  DIV_14 = (F34*K22*dna2_mod[91]*dna2_mod[15]*dna2_mod[30])
  DIV_18 = (3600*(km28 + dna2_mod[12]*dna2_mod[30]))
  DIV_19 = (3600*(km30 + dna2_mod[14]*dna2_mod[15])*(F39*dna2_mod[14] + 1)*(F38*dna2_mod[30] + 1))
  DIV_15 = (2*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[41]))

  if(DIV_1 == 0):
      DIV_1 = fixed

  if(DIV_2 == 0):
      DIV_2 = fixed

  DIV_16 = ((F1*K2*dna2_mod[71]*dna2_mod[0])/DIV_1 - (F2*K2*dna2_mod[71]*dna2_mod[1])/DIV_2)
  DIV_17 = ((F1*K2*dna2_mod[71]*dna2_mod[0])/DIV_1 - (F2*K2*dna2_mod[71]*dna2_mod[1])/DIV_2)
 
  if(DIV_5 == 0):
      DIV_5 = fixed

  if(DIV_6 == 0):
      DIV_6 = fixed

  if(DIV_7 == 0):
      DIV_7 = fixed

  if(DIV_8 == 0):
      DIV_8 = fixed

  if(DIV_9 == 0):
      DIV_9 = fixed    

  if(DIV_11 == 0):
      DIV_11 = fixed

  if(DIV_12 == 0):
      DIV_12 = fixed

  if(DIV_16 == 0):
      DIV_16 = fixed

  if(DIV_17 == 0):
      DIV_17 = fixed

  if(DIV_14 == 0):
      DIV_14 = fixed

  if(DIV_18 == 0):
      DIV_18 = fixed

  if(DIV_19 == 0):
      DIV_19 = fixed

  if(DIV_15 == 0):
      DIV_15 = fixed

  DIV_3 = ((F1*K2*dna2_mod[71]*dna2_mod[0])/DIV_1 - (F2*K2*dna2_mod[71]*dna2_mod[1])/DIV_2)

  if(DIV_3 == 0):
      DIV_3 = fixed

  DIV_4 = ((F1*K2*dna2_mod[71]*dna2_mod[0])/DIV_1 - (F2*K2*dna2_mod[71]*dna2_mod[1])/DIV_2)

  if(DIV_4 == 0):
      DIV_4 = fixed

  ROOT_1 = ((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41])/DIV_9)

  if(ROOT_1 < 0):
      ROOT_1 = 0.01

  DIV_10 = (7200*(F52*dna2_mod[41] + 1)*(km40 + (km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*(ROOT_1)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))/DIV_15)*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[41]))

  if(DIV_10 == 0):
      DIV_10 = fixed

  DIV_13 = (3600*(km29 + dna2_mod[15]*dna2_mod[30])*((3600*F35*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)*((K2*dna2_mod[71]*dna2_mod[0]*((F1*((K2*dna2_mod[71]*dna2_mod[0])/DIV_1 - (K2*dna2_mod[71]*dna2_mod[1])/DIV_2))/DIV_16 - 1))/DIV_1 - (K2*dna2_mod[71]*dna2_mod[1]*((F2*((K2*dna2_mod[71]*dna2_mod[0])/DIV_1 - (K2*dna2_mod[71]*dna2_mod[1])/DIV_2))/DIV_17 - 1))/DIV_2 + (K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/DIV_5 + (K5*dna2_mod[74]*dna2_mod[1])/DIV_6 - (K5*dna2_mod[74]*dna2_mod[3])/DIV_7 - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))/DIV_8 + (K31*dna2_mod[100]*km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*(ROOT_1)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))/DIV_10))/DIV_11 + 1))

  if(DIV_13 == 0):
      DIV_13 = fixed
    
  dna2_mod[126] = np.real((3600*(km29 + dna2_mod[15]*dna2_mod[30])*((3600*F35*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)*((K2*dna2_mod[71]*dna2_mod[0]*((F1*((K2*dna2_mod[71]*dna2_mod[0])/DIV_1 - (K2*dna2_mod[71]*dna2_mod[1])/DIV_2))/DIV_3 - 1))/DIV_1 - (K2*dna2_mod[71]*dna2_mod[1]*((F2*((K2*dna2_mod[71]*dna2_mod[0])/DIV_1 - (K2*dna2_mod[71]*dna2_mod[1])/DIV_2))/DIV_4 - 1))/DIV_2 + (K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/DIV_5 + (K5*dna2_mod[74]*dna2_mod[1])/DIV_6 - (K5*dna2_mod[74]*dna2_mod[3])/DIV_7 - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))/DIV_8 + (K31*dna2_mod[100]*km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*(ROOT_1)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))/DIV_10))/DIV_11 + 1)*((K30*dna2_mod[99]*dna2_mod[22]*dna2_mod[33])/DIV_12 - (K22*dna2_mod[91]*dna2_mod[15]*dna2_mod[30])/DIV_13 + (K21*dna2_mod[90]*dna2_mod[12]*dna2_mod[30]*(F33*dna2_mod[14] + 1))/DIV_18 - (K23*dna2_mod[92]*dna2_mod[14]*dna2_mod[15]*(F36*dna2_mod[14] + 1)*(F37*dna2_mod[15] + 1))/DIV_19))/DIV_14)
              

  
  dna2_mod[127] = np.real(-((K27*dna2_mod[96]*dna2_mod[20])/(3600*(km36 + dna2_mod[20])) - 3*((K27*dna2_mod[96]*dna2_mod[19]*dna2_mod[31])/(3600*(km35 + dna2_mod[19]*dna2_mod[31]))) + (K18*dna2_mod[87]*dna2_mod[12]*dna2_mod[33]*dna2_mod[34])/(3600*(km25 + dna2_mod[12]*dna2_mod[33]*dna2_mod[34])) - (K20*dna2_mod[89]*dna2_mod[14]*dna2_mod[31]*dna2_mod[32]*dna2_mod[38])/(3600*(km27 + dna2_mod[14]*dna2_mod[31]*dna2_mod[32]*dna2_mod[38])) + (K19*dna2_mod[88]*dna2_mod[30]*dna2_mod[33]*dna2_mod[36]*dna2_mod[37])/(3600*(km26 + dna2_mod[30]*dna2_mod[33]*dna2_mod[36]*dna2_mod[37])) - (K26*dna2_mod[95]*dna2_mod[18]*dna2_mod[33]*dna2_mod[39])/(3600*(km34 + dna2_mod[18]*dna2_mod[33]*dna2_mod[39])*(F47*dna2_mod[19] + 1)*(F46*dna2_mod[30] + 1)) + (K25*dna2_mod[94]*dna2_mod[17]*dna2_mod[33]*(F42*dna2_mod[17] + 1)*(F43*dna2_mod[31] + 1))/(3600*(km33 + dna2_mod[17]*dna2_mod[33])*(F44*dna2_mod[30] + 1)*(F45*dna2_mod[32] + 1)))/((F53*K18*dna2_mod[87]*dna2_mod[12]*dna2_mod[33]*dna2_mod[34])/(3600*(km25 + dna2_mod[12]*dna2_mod[33]*dna2_mod[34])) - (F55*K26*dna2_mod[95]*dna2_mod[18]*dna2_mod[33]*dna2_mod[39])/(3600*(km34 + dna2_mod[18]*dna2_mod[33]*dna2_mod[39])*(F47*dna2_mod[19] + 1)*(F46*dna2_mod[30] + 1)) + (F54*K25*dna2_mod[94]*dna2_mod[17]*dna2_mod[33]*(F42*dna2_mod[17] + 1)*(F43*dna2_mod[31] + 1))/(3600*(km33 + dna2_mod[17]*dna2_mod[33])*(F44*dna2_mod[30] + 1)*(F45*dna2_mod[32] + 1))))
  dna2_mod[128] = np.real(-((K24*dna2_mod[93]*dna2_mod[17])/(3600*km32 + 3600*dna2_mod[17]) - (K24*dna2_mod[93]*dna2_mod[16])/(3600*km31 + 3600*dna2_mod[16]) + (K23*dna2_mod[92]*dna2_mod[14]*dna2_mod[15]*(F36*dna2_mod[14] + 1)*(F37*dna2_mod[15] + 1))/((3600*km30 + 3600*dna2_mod[14]*dna2_mod[15])*(F39*dna2_mod[14] + 1)*(F38*dna2_mod[30] + 1)))/((F40*K24*dna2_mod[93]*dna2_mod[16])/(3600*km31 + 3600*dna2_mod[16]) - (F41*K24*dna2_mod[93]*dna2_mod[17])/(3600*km32 + 3600*dna2_mod[17])))
  dna2_mod[129] = np.real(((3600*km37 + 3600*dna2_mod[20]*dna2_mod[37])*((K27*dna2_mod[96]*dna2_mod[20])/(3600*km36 + 3600*dna2_mod[20]) - (K29*dna2_mod[98]*dna2_mod[21])/(3600*km38 + 3600*dna2_mod[21]) - 3*((K27*dna2_mod[96]*dna2_mod[19]*dna2_mod[31])/(3600*km35 + 3600*dna2_mod[19]*dna2_mod[31])) + (4*K28*dna2_mod[97]*dna2_mod[20]*dna2_mod[37])/(3600*km37 + 3600*dna2_mod[20]*dna2_mod[37]) - (2*K20*dna2_mod[89]*dna2_mod[14]*dna2_mod[31]*dna2_mod[32]*dna2_mod[38])/(3600*km27 + 3600*dna2_mod[14]*dna2_mod[31]*dna2_mod[32]*dna2_mod[38]) + (2*K19*dna2_mod[88]*dna2_mod[30]*dna2_mod[33]*dna2_mod[36]*dna2_mod[37])/(3600*km26 + 3600*dna2_mod[30]*dna2_mod[33]*dna2_mod[36]*dna2_mod[37])))/(4*F48*K28*dna2_mod[97]*dna2_mod[20]*dna2_mod[37]))
  dna2_mod[130] = np.real(((KS27*dna2_mod[45])/(60*(F_S30*dna2_mod[68] + 1)) - (KS26*dna2_mod[30])/(60*(F_S29*dna2_mod[69] + 1)) + (KS5*dna2_mod[44]*dna2_mod[68])/(60*(F_S4*dna2_mod[55] + 1)) + (K10*dna2_mod[79]*dna2_mod[4])/(3600*km11 + 3600*dna2_mod[4]) + (K7*dna2_mod[76]*dna2_mod[4])/((F19*dna2_mod[5] + 1)*(3600*km8 + 3600*dna2_mod[4])) - (K10*dna2_mod[79]*dna2_mod[6]*dna2_mod[7])/(3600*km12 + 3600*dna2_mod[6]*dna2_mod[7]) - (K6*dna2_mod[75]*dna2_mod[3]*dna2_mod[30]*(F15*dna2_mod[5] + 1))/((3600*km7 + 3600*dna2_mod[3]*dna2_mod[30])*(F17*dna2_mod[16] + 1)*(F16*dna2_mod[30] + 1)))/(KS16/60 + (F18*K7*dna2_mod[76]*dna2_mod[4])/((F19*dna2_mod[5] + 1)*(3600*km8 + 3600*dna2_mod[4])) + (F14_modified*K6*dna2_mod[75]*dna2_mod[3]*dna2_mod[30]*(F15*dna2_mod[5] + 1))/((3600*km7 + 3600*dna2_mod[3]*dna2_mod[30])*(F17*dna2_mod[16] + 1)*(F16*dna2_mod[30] + 1)))) 
  
  
            
  A = (3600*(km1 + dna2_mod[0]))
  B = (3600*(km2 + dna2_mod[1]))
  C = (3600*(km1 + dna2_mod[0]))
  D = (3600*(km2 + dna2_mod[1]))
  E = (3600*(km1 + dna2_mod[0]))
  F = (3600*(km1 + dna2_mod[0]))
  G = (3600*(km2 + dna2_mod[1]))
  H = (3600*(km1 + dna2_mod[0]))

  if(A == 0):
      A = fixed

  if(B == 0):
      B = fixed

  if(C == 0):
      C = fixed

  if(D == 0):
      D = fixed

  if(E == 0):
      E = fixed

  if(F == 0):
      F = fixed

  if(G == 0):
      G = fixed

  if(H == 0):
      H = fixed
            
  I = (3600*(km2 + dna2_mod[1]))

  if(I == 0):
      I = fixed

  J = (3600*(km2 + dna2_mod[1]))
  K = (3600*(km4 + dna2_mod[1]))
  L = (3600*(km5 + dna2_mod[1])*(F8*dna2_mod[4] + 1)*(F9*dna2_mod[24] + 1)*(F10*dna2_mod[29] + 1))
  M = (3600*(km6 + dna2_mod[3])*(F11*dna2_mod[4] + 1)*(F12*dna2_mod[24] + 1)*(F13*dna2_mod[29] + 1))
  N = (3600*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1))
  P = ((F1*K2*dna2_mod[71]*dna2_mod[0])/H - (F2*K2*dna2_mod[71]*dna2_mod[1])/I)
  Q = (12960000*(km43 + dna2_mod[25])*(km44 + dna2_mod[25]))
  RR = (2*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[41]))

  if(J == 0):
      J = fixed

  if(K == 0):
      K = fixed

  if(L == 0):
      L = fixed

  if(M == 0):
      M = fixed

  if(N == 0):
      N = fixed

  if(P == 0):
      P = fixed

  if(Q == 0):
      Q = fixed

  if(RR == 0):
      RR = fixed

  S = ((F1*K2*dna2_mod[71]*dna2_mod[0])/H - (F2*K2*dna2_mod[71]*dna2_mod[1])/I)
  T = ((F1*K2*dna2_mod[71]*dna2_mod[0])/H - (F2*K2*dna2_mod[71]*dna2_mod[1])/I)
  U = (3600*(km5 + dna2_mod[1])*(F8*dna2_mod[4] + 1)*(F9*dna2_mod[24] + 1)*(F10*dna2_mod[29] + 1))
  V = (3600*(km6 + dna2_mod[3])*(F11*dna2_mod[4] + 1)*(F12*dna2_mod[24] + 1)*(F13*dna2_mod[29] + 1))
  W = (3600*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1))
  X = (12960000*(km43 + dna2_mod[25])*(km44 + dna2_mod[25]))

  if(S == 0):
      S = fixed

  if(T == 0):
      T = fixed

  if(U == 0):
      U = fixed

  if(V == 0):
      V = fixed

  if(W == 0):
      W = fixed

  if(X == 0):
      X = fixed           

  Y = (7200*(F52*dna2_mod[41] + 1)*(km40 + (km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41])/X)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))/RR)*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[41]))
  Z = (3600*(km6 + dna2_mod[3])*(F11*dna2_mod[4] + 1)*(F12*dna2_mod[24] + 1)*(F13*dna2_mod[29] + 1))
  AA = (3600*(km5 + dna2_mod[1])*(F8*dna2_mod[4] + 1)*(F9*dna2_mod[24] + 1)*(F10*dna2_mod[29] + 1))
  BB = (12960000*(km43 + dna2_mod[25])*(km44 + dna2_mod[25]))
  CC = (7200*(F52*dna2_mod[41] + 1)*(km40 + (km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41])/Q)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))/RR)*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[41]))

  if(Y == 0):
      Y = fixed

  if(Z == 0):
      Z = fixed

  if(AA == 0):
      AA = fixed

  if(BB == 0):
      BB = fixed

  if(CC == 0):
      CC = fixed

  DD = (3600*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1))
  EE = (3600*(km6 + dna2_mod[3])*(F11*dna2_mod[4] + 1)*(F12*dna2_mod[24] + 1)*(F13*dna2_mod[29] + 1))

  if(DD == 0):
      DD = fixed

  if(EE == 0):
      EE = fixed

  FF = (3600*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1))
  GG = (2*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[41]))
  HH = (12960000*(km43 + dna2_mod[25])*(km44 + dna2_mod[25]))

  if(FF == 0):
      FF = fixed

  if(GG == 0):
      GG = fixed

  if(HH == 0):
      HH = fixed

  II = (3600*(km30 + dna2_mod[14]*dna2_mod[15])*(F39*dna2_mod[14] + 1)*(F38*dna2_mod[30] + 1))
  JJ = (F3*K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))
  KK = (3600*(km28 + dna2_mod[12]*dna2_mod[30]))
  LL = (F3*K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))

  if(II == 0):
      II = fixed

  if(JJ == 0):
      JJ = fixed

  if(KK == 0):
      KK = fixed

  if(LL == 0):
      LL = fixed

  MM = ((F1*K2*dna2_mod[71]*dna2_mod[0])/H - (F2*K2*dna2_mod[71]*dna2_mod[1])/I)
  NN = (7200*(F52*dna2_mod[41] + 1)*(km40 + (km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41])/HH))**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))/RR)*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[41])
  OO = ((F1*K2*dna2_mod[71]*dna2_mod[0])/H - (F2*K2*dna2_mod[71]*dna2_mod[1])/I)
  PP = (3600*(km5 + dna2_mod[1])*(F8*dna2_mod[4] + 1)*(F9*dna2_mod[24] + 1)*(F10*dna2_mod[29] + 1))
  QQ = (F34*K22*dna2_mod[91]*dna2_mod[15]*dna2_mod[30])
  SS = (3600*(km21 + dna2_mod[10]))
  TT = (3600*(km22 + dna2_mod[11]))
  UU = ((F1*K2*dna2_mod[71]*dna2_mod[0])/H - (F2*K2*dna2_mod[71]*dna2_mod[1])/I)
  VV = (3600*(km39 + dna2_mod[22]*dna2_mod[33])*(F51*dna2_mod[15] + 1))
  XXX = (4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41])
  YYY = (K31*dna2_mod[100]*km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41])/HH)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))
  ZZZ = (K31*dna2_mod[100]*km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41])/HH)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))
  WWW = (K31*dna2_mod[100]*km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41])/HH)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))
  UUU = (7200*(F52*dna2_mod[41] + 1)*(km40 + (km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41])/HH)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))/GG)*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[41]))

  if(MM == 0):
      MM = fixed

  if(NN == 0):
      NN = fixed

  if(OO == 0):
      OO = fixed

  if(PP == 0):
      PP = fixed

  if(QQ == 0):
      QQ = fixed

  if(SS == 0):
      SS = fixed

  if(TT == 0):
      TT = fixed

  if(UU == 0):
      UU = fixed

  if(VV == 0):
      VV = fixed

  if(XXX == 0):
      XXX = fixed

  if(YYY == 0):
      YYY = fixed

  if(ZZZ == 0):
      ZZZ = fixed

  if(WWW == 0):
      WWW = fixed

  if(UUU == 0):
      UUU = fixed

  CD = ((F1*K2*dna2_mod[71]*dna2_mod[0])/C - (F2*K2*dna2_mod[71]*dna2_mod[1])/D)

  if(CD == 0):
      CD = fixed   

  VVV = (3600*(km29 + dna2_mod[15]*dna2_mod[30])*((3600*F35*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)*((K2*dna2_mod[71]*dna2_mod[0]*((F1*((K2*dna2_mod[71]*dna2_mod[0])/H - (K2*dna2_mod[71]*dna2_mod[1])/I))/((F1*K2*dna2_mod[71]*dna2_mod[0])/H - (F2*K2*dna2_mod[71]*dna2_mod[1])/I) - 1))/H - (K2*dna2_mod[71]*dna2_mod[1]*((F2*((K2*dna2_mod[71]*dna2_mod[0])/H - (K2*dna2_mod[71]*dna2_mod[1])/I))/MM - 1))/I + (K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/(3600*(km4 + dna2_mod[1])) + (K5*dna2_mod[74]*dna2_mod[1])/AA - (K5*dna2_mod[74]*dna2_mod[3])/Z - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))/DD + YYY/UUU))/JJ + 1))

  if(VVV == 0):
      VVV = fixed

  XXXX = (7200*(F52*dna2_mod[41] + 1)*(km40 + (km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41])/HH)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))/RR)*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[41]))

  if(XXXX == 0):
      XXXX = fixed


  YYYY = (3600*(km29 + dna2_mod[15]*dna2_mod[30])*((3600*F35*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)*((K2*dna2_mod[71]*dna2_mod[0]*((F1*((K2*dna2_mod[71]*dna2_mod[0])/H - (K2*dna2_mod[71]*dna2_mod[1])/I))/UU - 1))/H - (K2*dna2_mod[71]*dna2_mod[1]*((F2*((K2*dna2_mod[71]*dna2_mod[0])/H - (K2*dna2_mod[71]*dna2_mod[1])/I))/OO - 1))/I + (K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/(3600*(km4 + dna2_mod[1])) + (K5*dna2_mod[74]*dna2_mod[1])/PP - (K5*dna2_mod[74]*dna2_mod[3])/EE - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))/(3600*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)) + (K31*dna2_mod[100]*km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*(XXX/HH)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))/(7200*(F52*dna2_mod[41] + 1)*(km40 + (km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41])/HH)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))/RR)*(2*K31*dna2_mod[100]*km42**2*dna2_mod[1]**2 + K33*dna2_mod[102]*km40**2*dna2_mod[24]**2 + F52*K33*dna2_mod[102]*km40**2*dna2_mod[24]**2*dna2_mod[41]))))/LL + 1))

  if(YYYY == 0):
      YYYY = fixed

  BBBB = (3600*(km29 + dna2_mod[15]*dna2_mod[30])*((3600*F35*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)*((K2*dna2_mod[71]*dna2_mod[0]*((F1*((K2*dna2_mod[71]*dna2_mod[0])/H - (K2*dna2_mod[71]*dna2_mod[1])/I))/UU - 1))/H - (K2*dna2_mod[71]*dna2_mod[1]*((F2*((K2*dna2_mod[71]*dna2_mod[0])/H - (K2*dna2_mod[71]*dna2_mod[1])/I))/UU - 1))/I + (K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/K + (K5*dna2_mod[74]*dna2_mod[1])/PP - (K5*dna2_mod[74]*dna2_mod[3])/EE - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))/FF + (K31*dna2_mod[100]*km40*km42*dna2_mod[1]*(2*K31*dna2_mod[100]*km42*dna2_mod[1] - 3600*((4*K31**2*dna2_mod[100]**2*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*km44*dna2_mod[1]**2 + K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2 + 4*K31**2*dna2_mod[100]**2*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K31**2*dna2_mod[100]**2*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2 + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 8*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25] - 8*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25] + 4*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25] - 4*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25] + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + F52**2*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*km44*dna2_mod[24]**2*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 2*F52*K33**2*dna2_mod[102]**2*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2 + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*dna2_mod[1]**2*dna2_mod[25]**2*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*dna2_mod[24]**2*dna2_mod[25]**2*dna2_mod[41] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25] + 4*F52**2*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 - 4*F52**2*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41]**2 + 8*F52*K31*K34*dna2_mod[100]*dna2_mod[103]*km42**2*km44*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K31*K35*dna2_mod[100]*dna2_mod[104]*km42**2*km43*dna2_mod[1]**2*dna2_mod[25]*dna2_mod[41] + 8*F52*K33*K34*dna2_mod[102]*dna2_mod[103]*km40**2*km44*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] - 8*F52*K33*K35*dna2_mod[102]*dna2_mod[104]*km40**2*km43*dna2_mod[24]**2*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]**2*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km43*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41] + 4*F52*K31*K33*dna2_mod[100]*dna2_mod[102]*km40*km42*km44*dna2_mod[1]*dna2_mod[24]*dna2_mod[25]*dna2_mod[41])/HH)**(1/2) + K33*dna2_mod[102]*km40*dna2_mod[24] + F52*K33*dna2_mod[102]*km40*dna2_mod[24]*dna2_mod[41]))/XXXX))/LL + 1))

  if(BBBB == 0):
      BBBB = fixed


  AAAA = (K22*dna2_mod[91]*dna2_mod[15]*dna2_mod[30])

  if(AAAA == 0):
      AAAA = fixed

  WWWW = (3600*(km23 + dna2_mod[11]*dna2_mod[31])*(F29*dna2_mod[14] + 1)*(F28*dna2_mod[30] + 1)*(F32*dna2_mod[36] + 1)*((3600*F31*(km29 + dna2_mod[15]*dna2_mod[30])*((3600*F35*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)*((K2*dna2_mod[71]*dna2_mod[0]*((F1*((K2*dna2_mod[71]*dna2_mod[0])/H - (K2*dna2_mod[71]*dna2_mod[1])/I))/S - 1))/H - (K2*dna2_mod[71]*dna2_mod[1]*((F2*((K2*dna2_mod[71]*dna2_mod[0])/H - (K2*dna2_mod[71]*dna2_mod[1])/I))/T - 1))/I + (K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/K + (K5*dna2_mod[74]*dna2_mod[1])/U - (K5*dna2_mod[74]*dna2_mod[3])/V - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))/W + WWW/Y))/LL + 1)*((K30*dna2_mod[99]*dna2_mod[22]*dna2_mod[33])/VV - AAAA/VVV + (K21*dna2_mod[90]*dna2_mod[12]*dna2_mod[30]*(F33*dna2_mod[14] + 1))/KK - (K23*dna2_mod[92]*dna2_mod[14]*dna2_mod[15]*(F36*dna2_mod[14] + 1)*(F37*dna2_mod[15] + 1))/II))/QQ + 1)*((K15*dna2_mod[84]*dna2_mod[10])/SS - (K15*dna2_mod[84]*dna2_mod[11])/TT + (K22*dna2_mod[91]*dna2_mod[15]*dna2_mod[30]*((3600*(km29 + dna2_mod[15]*dna2_mod[30])*((3600*F35*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)*((K2*dna2_mod[71]*dna2_mod[0]*((F1*((K2*dna2_mod[71]*dna2_mod[0])/H - (K2*dna2_mod[71]*dna2_mod[1])/I))/UU - 1))/H - (K2*dna2_mod[71]*dna2_mod[1]*((F2*((K2*dna2_mod[71]*dna2_mod[0])/H - (K2*dna2_mod[71]*dna2_mod[1])/I))/UU - 1))/I + (K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/K + (K5*dna2_mod[74]*dna2_mod[1])/PP - (K5*dna2_mod[74]*dna2_mod[3])/EE - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))/FF + YYY/NN))/LL + 1)*((K30*dna2_mod[99]*dna2_mod[22]*dna2_mod[33])/VV - AAAA/YYYY + (K21*dna2_mod[90]*dna2_mod[12]*dna2_mod[30]*(F33*dna2_mod[14] + 1))/KK - (K23*dna2_mod[92]*dna2_mod[14]*dna2_mod[15]*(F36*dna2_mod[14] + 1)*(F37*dna2_mod[15] + 1))/II))/AAAA + 1))/BBBB))

  if(WWWW == 0):
      WWWW = fixed

  dna2_mod[131] = np.real(((K16*dna2_mod[85]*dna2_mod[11]*dna2_mod[31]*(F27*dna2_mod[4] + 1)*((3600*F26*(km3 + dna2_mod[2]*dna2_mod[30])*(F5*dna2_mod[1] + 1)*(F6*dna2_mod[14] + 1)*((K2*dna2_mod[71]*dna2_mod[0]*((F1*((K2*dna2_mod[71]*dna2_mod[0])/A - (K2*dna2_mod[71]*dna2_mod[1])/B))/CD - 1))/E - (K2*dna2_mod[71]*dna2_mod[1]*((F2*((K2*dna2_mod[71]*dna2_mod[0])/F - (K2*dna2_mod[71]*dna2_mod[1])/G))/P - 1))/J + (K4*dna2_mod[73]*dna2_mod[1]*(F7*dna2_mod[1] + 1))/K + (K5*dna2_mod[74]*dna2_mod[1])/L - (K5*dna2_mod[74]*dna2_mod[3])/M - (K3*dna2_mod[72]*dna2_mod[2]*dna2_mod[30]*(F4*dna2_mod[31] + 1))/N + ZZZ/CC))/LL + 1))/WWWW - 1)/F30)
  
  dna2_mod[132] = np.real(-((K2*dna2_mod[71]*dna2_mod[0])/H - (K2*dna2_mod[71]*dna2_mod[1])/I)/UU)
  dna2_mod[133] = np.real(-((K13*dna2_mod[82]*dna2_mod[9])/(3600*(km18 + dna2_mod[9])) + (K14*dna2_mod[83]*dna2_mod[9])/(1800*(km19 + dna2_mod[9])) - (K14*dna2_mod[83]*dna2_mod[10])/(1800*(km20 + dna2_mod[10])) - (K15*dna2_mod[84]*dna2_mod[10])/(3600*(km21 + dna2_mod[10])) + (K15*dna2_mod[84]*dna2_mod[11])/(3600*(km22 + dna2_mod[11])) - (K13*dna2_mod[82]*dna2_mod[8]*dna2_mod[31])/(3600*(km17 + dna2_mod[8]*dna2_mod[31])))/((F24*K14*dna2_mod[83]*dna2_mod[9])/(1800*(km19 + dna2_mod[9])) - (F25*K14*dna2_mod[83]*dna2_mod[10])/(1800*(km20 + dna2_mod[10]))))

  DNA_SIZE = 134
  for i in range(0,DNA_SIZE):
      
      if(dna2_mod[i] < 0):
          dna2_mod[i] = 0.001
            
      if(dna2_mod[i] > 1):
          dna2_mod[i] = 0.999


  return dna1_mod, dna2_mod     




def mutate(dna):
    
  """
  For each gene in the DNA, there is a 1/mutation_chance chance that it will be
  switched out with a random character. This ensures diversity in the
  population, and ensures that is difficult to get stuck in local minima.
  """
  fixed = 0.01
  DNA_SIZE    = 134
  dna_out = dna
  mutation_chance = 100
  for c in range(DNA_SIZE-27):  
#    print("hello\n") 
    if int(random.random()*mutation_chance) > 70:
      dna_out[c] = dna_out[c] + random.uniform(-.01, .01)
    else:
      dna_out[c] = dna_out[c]
      
  constant_terms_in_u18_design = ((KS7*dna_out[43])/60 + (KS29*dna_out[43])/60 + KS13/(60*(F_S15*dna_out[43] + 1)*(F_S14*dna_out[58] + 1)) + (KS9*dna_out[47])/(60*(F_S8*dna_out[43] + 1)*(F_S9*dna_out[46] + 1)) - (KS36*(kbg16 - lg16 + kg16*dna_out[48]))/(60*FG6*kg16*dna_out[48]) + (KS8*dna_out[41]*dna_out[42]*dna_out[43]*dna_out[46]*dna_out[49])/(60*(F_S6*dna_out[56] + 1)*(F_S7*dna_out[57] + 1)))
  dna_out[107] = np.real(-(60*((KS31*(kbg16 - lg16 + kg16*dna_out[48]))/(60*FG6*kg16*dna_out[48]) - (KS8*dna_out[41]*dna_out[42]*dna_out[43]*dna_out[46]*dna_out[49])/(60*(F_S6*dna_out[56] + 1)*(F_S7*dna_out[57] + 1))))/KS1)
  dna_out[108] = np.real(((kg28*dna_out[45]*dna_out[64]*dna_out[66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna_out[63] + 60) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))))/((kbg28 - lg28)*(kg30*dna_out[43] + kg9*dna_out[43]*dna_out[66] + kg27*dna_out[43]*dna_out[66] + kg37*dna_out[41]*dna_out[66] + kg29*dna_out[42]*dna_out[64]*dna_out[66] + kg21*dna_out[41]*dna_out[42]*dna_out[64]*dna_out[66] + (kg20*dna_out[43]*dna_out[45]*dna_out[66])/(FG9*dna_out[58] + 1) + (F_S22*KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60)))) - 1)/FG13)
  dna_out[109] = np.real(-(60*((KS29*dna_out[43])/60 + (KS19*dna_out[45]*dna_out[64]*dna_out[65])/(60*(F_S21*dna_out[50] + 1)*(F_S20*dna_out[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna_out[63] + 60) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))))/(kg30*dna_out[43] + kg9*dna_out[43]*dna_out[66] + kg27*dna_out[43]*dna_out[66] + kg37*dna_out[41]*dna_out[66] + kg29*dna_out[42]*dna_out[64]*dna_out[66] + kg21*dna_out[41]*dna_out[42]*dna_out[64]*dna_out[66] + (kg20*dna_out[43]*dna_out[45]*dna_out[66])/(FG9*dna_out[58] + 1) + (F_S22*KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))) - 1))))/KS54)
  dna_out[110] = np.real(-((KS49*((KS33*((KS29*dna_out[43])/60 + (KS19*dna_out[45]*dna_out[64]*dna_out[65])/(60*(F_S21*dna_out[50] + 1)*(F_S20*dna_out[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna_out[63] + 60) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))))/(kg30*dna_out[43] + kg9*dna_out[43]*dna_out[66] + kg27*dna_out[43]*dna_out[66] + kg37*dna_out[41]*dna_out[66] + kg29*dna_out[42]*dna_out[64]*dna_out[66] + kg21*dna_out[41]*dna_out[42]*dna_out[64]*dna_out[66] + (kg20*dna_out[43]*dna_out[45]*dna_out[66])/(FG9*dna_out[58] + 1) + (F_S22*KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))) - 1))))/KS54 - (KS32*((kg28*dna_out[45]*dna_out[64]*dna_out[66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna_out[63] + 60) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))))/((kbg28 - lg28)*(kg30*dna_out[43] + kg9*dna_out[43]*dna_out[66] + kg27*dna_out[43]*dna_out[66] + kg37*dna_out[41]*dna_out[66] + kg29*dna_out[42]*dna_out[64]*dna_out[66] + kg21*dna_out[41]*dna_out[42]*dna_out[64]*dna_out[66] + (kg20*dna_out[43]*dna_out[45]*dna_out[66])/(FG9*dna_out[58] + 1) + (F_S22*KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60)))) - 1))/(60*FG13) - (KS2*dna_out[44]*dna_out[51])/60 + (KS17*dna_out[42])/(60*(F_S17*dna_out[44] + 1)*(F_S18*dna_out[45] + 1)) + (KS8*dna_out[41]*dna_out[42]*dna_out[43]*dna_out[46]*dna_out[49])/(60*(F_S6*dna_out[56] + 1)*(F_S7*dna_out[57] + 1))))/KS35 - (KS2*dna_out[44]*dna_out[51])/60)/(KS48/60 - (KS34*KS49)/(60*KS35)))
  dna_out[111] = np.real((60*((KS2*dna_out[44]*dna_out[51])/60 + (KS48*((KS49*((KS33*((KS29*dna_out[43])/60 + (KS19*dna_out[45]*dna_out[64]*dna_out[65])/(60*(F_S21*dna_out[50] + 1)*(F_S20*dna_out[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna_out[63] + 60) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))))/(kg30*dna_out[43] + kg9*dna_out[43]*dna_out[66] + kg27*dna_out[43]*dna_out[66] + kg37*dna_out[41]*dna_out[66] + kg29*dna_out[42]*dna_out[64]*dna_out[66] + kg21*dna_out[41]*dna_out[42]*dna_out[64]*dna_out[66] + (kg20*dna_out[43]*dna_out[45]*dna_out[66])/(FG9*dna_out[58] + 1) + (F_S22*KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))) - 1))))/KS54 - (KS32*((kg28*dna_out[45]*dna_out[64]*dna_out[66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna_out[63] + 60) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))))/((kbg28 - lg28)*(kg30*dna_out[43] + kg9*dna_out[43]*dna_out[66] + kg27*dna_out[43]*dna_out[66] + kg37*dna_out[41]*dna_out[66] + kg29*dna_out[42]*dna_out[64]*dna_out[66] + kg21*dna_out[41]*dna_out[42]*dna_out[64]*dna_out[66] + (kg20*dna_out[43]*dna_out[45]*dna_out[66])/(FG9*dna_out[58] + 1) + (F_S22*KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60)))) - 1))/(60*FG13) - (KS2*dna_out[44]*dna_out[51])/60 + (KS17*dna_out[42])/(60*(F_S17*dna_out[44] + 1)*(F_S18*dna_out[45] + 1)) + (KS8*dna_out[41]*dna_out[42]*dna_out[43]*dna_out[46]*dna_out[49])/(60*(F_S6*dna_out[56] + 1)*(F_S7*dna_out[57] + 1))))/KS35 - (KS2*dna_out[44]*dna_out[51])/60))/(60*(KS48/60 - (KS34*KS49)/(60*KS35)))))/KS49)
  dna_out[112] = np.real((kbg16 - lg16 + kg16*dna_out[48])/(FG6*kg16*dna_out[48]))
  dna_out[113] = np.real((60*((KS2*dna_out[44]*dna_out[51])/60 - (KS4*dna_out[50])/(60*(F_S3*dna_out[55] + 1)) + (KS18*dna_out[44])/(60*(F_S19*dna_out[45] + 1)) + (KS5*dna_out[44]*dna_out[68])/(60*(F_S4*dna_out[55] + 1)) + (KS14*dna_out[44]*dna_out[48])/(60*(F_S16*dna_out[55] + 1)) + (KS17*dna_out[42])/(60*(F_S17*dna_out[44] + 1)*(F_S18*dna_out[45] + 1)) - (KS37*(kbg16 - lg16 + kg16*dna_out[48]))/(60*FG6*kg16*dna_out[48]) - (KS52*(kbg16 - lg16 + kg16*dna_out[48]))/(60*FG6*kg16*dna_out[48]) + (KS6*dna_out[44]*dna_out[45]*dna_out[53]*dna_out[54])/(60*((60*(KS24/(60*F_S28*dna_out[53] + 60) - (KS28*dna_out[30]*dna_out[69])/60 + KS23/((F_S26*dna_out[21] + 1)*(60*F_S25*dna_out[20] + 60)*(F_S27*dna_out[53] + 1)) + (KS6*dna_out[44]*dna_out[45]*dna_out[53]*dna_out[54])/60))/(KS6*dna_out[44]*dna_out[45]*dna_out[53]*dna_out[54]) + 1))))/KS38)
  dna_out[114] = np.real(-((K9*dna_out[78]*dna_out[5])/((F23*dna_out[3] + 1)*(3600*km10 + 3600*dna_out[5])) - (K8*dna_out[77]*dna_out[3]*dna_out[30]*(F20*dna_out[3] + 1))/(3600*km9 + 3600*dna_out[3]*dna_out[30]))/((F22*K9*dna_out[78]*dna_out[5])/((F23*dna_out[3] + 1)*(3600*km10 + 3600*dna_out[5])) + (F21*K8*dna_out[77]*dna_out[3]*dna_out[30]*(F20*dna_out[3] + 1))/(3600*km9 + 3600*dna_out[3]*dna_out[30])))
  dna_out[115] = np.real((60*(KS24/(60*F_S28*dna_out[53] + 60) - (KS28*dna_out[30]*dna_out[69])/60 + KS23/((F_S26*dna_out[21] + 1)*(60*F_S25*dna_out[20] + 60)*(F_S27*dna_out[53] + 1)) + (KS6*dna_out[44]*dna_out[45]*dna_out[53]*dna_out[54])/60))/(F_S5*KS6*dna_out[44]*dna_out[45]*dna_out[53]*dna_out[54]))
  dna_out[116] = np.real(-(kbg23 - lg23)/kg23)
  dna_out[117] = np.real((60*((KS47*((KS49*((KS33*((KS29*dna_out[43])/60 + (KS19*dna_out[45]*dna_out[64]*dna_out[65])/(60*(F_S21*dna_out[50] + 1)*(F_S20*dna_out[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna_out[63] + 60) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))))/(kg30*dna_out[43] + kg9*dna_out[43]*dna_out[66] + kg27*dna_out[43]*dna_out[66] + kg37*dna_out[41]*dna_out[66] + kg29*dna_out[42]*dna_out[64]*dna_out[66] + kg21*dna_out[41]*dna_out[42]*dna_out[64]*dna_out[66] + (kg20*dna_out[43]*dna_out[45]*dna_out[66])/(FG9*dna_out[58] + 1) + (F_S22*KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))) - 1))))/KS54 - (KS32*((kg28*dna_out[45]*dna_out[64]*dna_out[66]*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna_out[63] + 60) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))))/((kbg28 - lg28)*(kg30*dna_out[43] + kg9*dna_out[43]*dna_out[66] + kg27*dna_out[43]*dna_out[66] + kg37*dna_out[41]*dna_out[66] + kg29*dna_out[42]*dna_out[64]*dna_out[66] + kg21*dna_out[41]*dna_out[42]*dna_out[64]*dna_out[66] + (kg20*dna_out[43]*dna_out[45]*dna_out[66])/(FG9*dna_out[58] + 1) + (F_S22*KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60)))) - 1))/(60*FG13) - (KS2*dna_out[44]*dna_out[51])/60 + (KS17*dna_out[42])/(60*(F_S17*dna_out[44] + 1)*(F_S18*dna_out[45] + 1)) + (KS8*dna_out[41]*dna_out[42]*dna_out[43]*dna_out[46]*dna_out[49])/(60*(F_S6*dna_out[56] + 1)*(F_S7*dna_out[57] + 1))))/KS35 - (KS2*dna_out[44]*dna_out[51])/60))/(60*(KS48/60 - (KS34*KS49)/(60*KS35))) - KS13/(60*(F_S15*dna_out[43] + 1)*(F_S14*dna_out[58] + 1)) + (KS6*dna_out[44]*dna_out[45]*dna_out[53]*dna_out[54])/(60*((60*(KS24/(60*F_S28*dna_out[53] + 60) - (KS28*dna_out[30]*dna_out[69])/60 + KS23/((F_S26*dna_out[21] + 1)*(60*F_S25*dna_out[20] + 60)*(F_S27*dna_out[53] + 1)) + (KS6*dna_out[44]*dna_out[45]*dna_out[53]*dna_out[54])/60))/(KS6*dna_out[44]*dna_out[45]*dna_out[53]*dna_out[54]) + 1))))/KS46)
  dna_out[118] = np.real((60*((KS22*dna_out[64])/60 + (KS25*dna_out[64])/60 - (KS20*dna_out[48]*dna_out[58])/(60*((60*F_S11*((KS2*dna_out[44]*dna_out[51])/60 - (KS4*dna_out[50])/(60*(F_S3*dna_out[55] + 1)) + (KS18*dna_out[44])/(60*(F_S19*dna_out[45] + 1)) + (KS5*dna_out[44]*dna_out[68])/(60*(F_S4*dna_out[55] + 1)) + (KS14*dna_out[44]*dna_out[48])/(60*(F_S16*dna_out[55] + 1)) + (KS17*dna_out[42])/(60*(F_S17*dna_out[44] + 1)*(F_S18*dna_out[45] + 1)) - (KS37*(kbg16 - lg16 + kg16*dna_out[48]))/(60*FG6*kg16*dna_out[48]) - (KS52*(kbg16 - lg16 + kg16*dna_out[48]))/(60*FG6*kg16*dna_out[48]) + (KS6*dna_out[44]*dna_out[45]*dna_out[53]*dna_out[54])/(60*((60*(KS24/(60*F_S28*dna_out[53] + 60) - (KS28*dna_out[30]*dna_out[69])/60 + KS23/((F_S26*dna_out[21] + 1)*(60*F_S25*dna_out[20] + 60)*(F_S27*dna_out[53] + 1)) + (KS6*dna_out[44]*dna_out[45]*dna_out[53]*dna_out[54])/60))/(KS6*dna_out[44]*dna_out[45]*dna_out[53]*dna_out[54]) + 1))))/KS38 + 1)*(F_S23*dna_out[63] + 1)) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/(60*(F_S21*dna_out[50] + 1)*(F_S20*dna_out[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna_out[63] + 60) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))))/(kg30*dna_out[43] + kg9*dna_out[43]*dna_out[66] + kg27*dna_out[43]*dna_out[66] + kg37*dna_out[41]*dna_out[66] + kg29*dna_out[42]*dna_out[64]*dna_out[66] + kg21*dna_out[41]*dna_out[42]*dna_out[64]*dna_out[66] + (kg20*dna_out[43]*dna_out[45]*dna_out[66])/(FG9*dna_out[58] + 1) + (F_S22*KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))) - 1))))/KS50)
  dna_out[119] = np.real(((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*dna_out[50] + 60*F_S13*constant_terms_in_u18_design*dna_out[45]))/(15*(F_S13*dna_out[45] + 1)))**(1/2)/(2*F_S1*constant_terms_in_u18_design))
  dna_out[120] = np.real(((60*KS3*dna_out[52]*(F_S13*dna_out[45] + 1))/(KS11*((30*((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*dna_out[50] + 60*F_S13*constant_terms_in_u18_design*dna_out[45]))/(15*(F_S13*dna_out[45] + 1)))**(1/2))/constant_terms_in_u18_design + 60)) - 1)/F_S2)
  dna_out[121] = np.real((KS10/((KS43*((K9*dna_out[78]*dna_out[5])/((F23*dna_out[3] + 1)*(3600*km10 + 3600*dna_out[5])) - (K8*dna_out[77]*dna_out[3]*dna_out[30]*(F20*dna_out[3] + 1))/(3600*km9 + 3600*dna_out[3]*dna_out[30])))/((F22*K9*dna_out[78]*dna_out[5])/((F23*dna_out[3] + 1)*(3600*km10 + 3600*dna_out[5])) + (F21*K8*dna_out[77]*dna_out[3]*dna_out[30]*(F20*dna_out[3] + 1))/(3600*km9 + 3600*dna_out[3]*dna_out[30])) + (KS4*dna_out[50])/(F_S3*dna_out[55] + 1) + (KS11*dna_out[50])/(F_S13*dna_out[45] + 1) - (KS39*(kbg16 - lg16 + kg16*dna_out[48]))/(FG6*kg16*dna_out[48]) + (KS11*dna_out[50]*((30*((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*dna_out[50] + 60*F_S13*constant_terms_in_u18_design*dna_out[45]))/(15*(F_S13*dna_out[45] + 1)))**(1/2))/constant_terms_in_u18_design + 60))/(60*(((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*dna_out[50] + 60*F_S13*constant_terms_in_u18_design*dna_out[45]))/(15*(F_S13*dna_out[45] + 1)))**(1/2)/(2*constant_terms_in_u18_design) + 1)*(F_S13*dna_out[45] + 1)) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S21*dna_out[50] + 1)*(F_S20*dna_out[66] + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna_out[63] + 60) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))))/(kg30*dna_out[43] + kg9*dna_out[43]*dna_out[66] + kg27*dna_out[43]*dna_out[66] + kg37*dna_out[41]*dna_out[66] + kg29*dna_out[42]*dna_out[64]*dna_out[66] + kg21*dna_out[41]*dna_out[42]*dna_out[64]*dna_out[66] + (kg20*dna_out[43]*dna_out[45]*dna_out[66])/(FG9*dna_out[58] + 1) + (F_S22*KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))) - 1))) - 1)/F_S10)
  dna_out[122] = np.real(-(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*dna_out[63] + 60) - (KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60)))/(kg30*dna_out[43] + kg9*dna_out[43]*dna_out[66] + kg27*dna_out[43]*dna_out[66] + kg37*dna_out[41]*dna_out[66] + kg29*dna_out[42]*dna_out[64]*dna_out[66] + kg21*dna_out[41]*dna_out[42]*dna_out[64]*dna_out[66] + (kg20*dna_out[43]*dna_out[45]*dna_out[66])/(FG9*dna_out[58] + 1) + (F_S22*KS19*dna_out[45]*dna_out[64]*dna_out[65])/((F_S20*dna_out[66] + 1)*(60*F_S21*dna_out[50] + 60))))          
  dna_out[123] = np.real(-(km0*((K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/(km4 + dna_out[1]) - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1)*((3600*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)*((K2*dna_out[71]*dna_out[0]*((F1*((K2*dna_out[71]*dna_out[0])/(3600*(km1 + dna_out[0])) - (K2*dna_out[71]*dna_out[1])/(3600*(km2 + dna_out[1]))))/((F1*K2*dna_out[71]*dna_out[0])/(3600*(km1 + dna_out[0])) - (F2*K2*dna_out[71]*dna_out[1])/(3600*(km2 + dna_out[1]))) - 1))/(3600*(km1 + dna_out[0])) - (K2*dna_out[71]*dna_out[1]*((F2*((K2*dna_out[71]*dna_out[0])/(3600*(km1 + dna_out[0])) - (K2*dna_out[71]*dna_out[1])/(3600*(km2 + dna_out[1]))))/((F1*K2*dna_out[71]*dna_out[0])/(3600*(km1 + dna_out[0])) - (F2*K2*dna_out[71]*dna_out[1])/(3600*(km2 + dna_out[1]))) - 1))/(3600*(km2 + dna_out[1])) + (K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/(3600*(km4 + dna_out[1])) + (K5*dna_out[74]*dna_out[1])/(3600*(km5 + dna_out[1])*(F8*dna_out[4] + 1)*(F9*dna_out[24] + 1)*(F10*dna_out[29] + 1)) - (K5*dna_out[74]*dna_out[3])/(3600*(km6 + dna_out[3])*(F11*dna_out[4] + 1)*(F12*dna_out[24] + 1)*(F13*dna_out[29] + 1)) - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))/(3600*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)) + (K31*dna_out[100]*km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40])/(12960000*(km43 + dna_out[25])*(km44 + dna_out[25])))**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[40]))/(7200*(F52*dna_out[40] + 1)*(km40 + (km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40])/(12960000*(km43 + dna_out[25])*(km44 + dna_out[25])))**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[40]))/(2*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[40])))*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[40]))))/(K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1)) + 1))/((km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1))))/(K1*dna_out[70]*(((K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/(km4 + dna_out[1]) - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1)*((3600*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)*((K2*dna_out[71]*dna_out[0]*((F1*((K2*dna_out[71]*dna_out[0])/(3600*(km1 + dna_out[0])) - (K2*dna_out[71]*dna_out[1])/(3600*(km2 + dna_out[1]))))/((F1*K2*dna_out[71]*dna_out[0])/(3600*(km1 + dna_out[0])) - (F2*K2*dna_out[71]*dna_out[1])/(3600*(km2 + dna_out[1]))) - 1))/(3600*(km1 + dna_out[0])) - (K2*dna_out[71]*dna_out[1]*((F2*((K2*dna_out[71]*dna_out[0])/(3600*(km1 + dna_out[0])) - (K2*dna_out[71]*dna_out[1])/(3600*(km2 + dna_out[1]))))/((F1*K2*dna_out[71]*dna_out[0])/(3600*(km1 + dna_out[0])) - (F2*K2*dna_out[71]*dna_out[1])/(3600*(km2 + dna_out[1]))) - 1))/(3600*(km2 + dna_out[1])) + (K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/(3600*(km4 + dna_out[1])) + (K5*dna_out[74]*dna_out[1])/(3600*(km5 + dna_out[1])*(F8*dna_out[4] + 1)*(F9*dna_out[24] + 1)*(F10*dna_out[29] + 1)) - (K5*dna_out[74]*dna_out[3])/(3600*(km6 + dna_out[3])*(F11*dna_out[4] + 1)*(F12*dna_out[24] + 1)*(F13*dna_out[29] + 1)) - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))/(3600*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)) + (K31*dna_out[100]*km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40])/(12960000*(km43 + dna_out[25])*(km44 + dna_out[25])))**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[40]))/(7200*(F52*dna_out[40] + 1)*(km40 + (km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40])/(12960000*(km43 + dna_out[25])*(km44 + dna_out[25])))**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[40]))/(2*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[40])))*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[40]))))/(K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1)) + 1))/((km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)))/(K1*dna_out[70]) + 1)))
  dna_out[124] = np.real((km40*km42*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40])/(12960000*(km43 + dna_out[25])*(km44 + dna_out[25])))**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[40]))/(2*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[40])))
  dna_out[125] = np.real((3600*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)*((K2*dna_out[71]*dna_out[0]*((F1*((K2*dna_out[71]*dna_out[0])/(3600*(km1 + dna_out[0])) - (K2*dna_out[71]*dna_out[1])/(3600*(km2 + dna_out[1]))))/((F1*K2*dna_out[71]*dna_out[0])/(3600*(km1 + dna_out[0])) - (F2*K2*dna_out[71]*dna_out[1])/(3600*(km2 + dna_out[1]))) - 1))/(3600*(km1 + dna_out[0])) - (K2*dna_out[71]*dna_out[1]*((F2*((K2*dna_out[71]*dna_out[0])/(3600*(km1 + dna_out[0])) - (K2*dna_out[71]*dna_out[1])/(3600*(km2 + dna_out[1]))))/((F1*K2*dna_out[71]*dna_out[0])/(3600*(km1 + dna_out[0])) - (F2*K2*dna_out[71]*dna_out[1])/(3600*(km2 + dna_out[1]))) - 1))/(3600*(km2 + dna_out[1])) + (K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/(3600*(km4 + dna_out[1])) + (K5*dna_out[74]*dna_out[1])/(3600*(km5 + dna_out[1])*(F8*dna_out[4] + 1)*(F9*dna_out[24] + 1)*(F10*dna_out[29] + 1)) - (K5*dna_out[74]*dna_out[3])/(3600*(km6 + dna_out[3])*(F11*dna_out[4] + 1)*(F12*dna_out[24] + 1)*(F13*dna_out[29] + 1)) - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))/(3600*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)) + (K31*dna_out[100]*km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40])/(12960000*(km43 + dna_out[25])*(km44 + dna_out[25])))**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[40]))/(7200*(F52*dna_out[40] + 1)*(km40 + (km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[40] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[40] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[40] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[40] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[40])/(12960000*(km43 + dna_out[25])*(km44 + dna_out[25])))**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[40]))/(2*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[40])))*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[40]))))/(F3*K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1)))
  
              
  DIV_1 = (3600*(km1 + dna_out[0]))
  DIV_2 = (3600*(km2 + dna_out[1]))
  DIV_5 = (3600*(km4 + dna_out[1]))
  DIV_6 = (3600*(km5 + dna_out[1])*(F8*dna_out[4] + 1)*(F9*dna_out[24] + 1)*(F10*dna_out[29] + 1))
  DIV_7 = (3600*(km6 + dna_out[3])*(F11*dna_out[4] + 1)*(F12*dna_out[24] + 1)*(F13*dna_out[29] + 1))
  DIV_8 = (3600*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1))
  DIV_9 = (12960000*(km43 + dna_out[25])*(km44 + dna_out[25]))
  DIV_11 = (F3*K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))
  DIV_12 = (3600*(km39 + dna_out[22]*dna_out[33])*(F51*dna_out[15] + 1))
  DIV_16 = ((F1*K2*dna_out[71]*dna_out[0])/DIV_1 - (F2*K2*dna_out[71]*dna_out[1])/DIV_2)
  DIV_17 = ((F1*K2*dna_out[71]*dna_out[0])/DIV_1 - (F2*K2*dna_out[71]*dna_out[1])/DIV_2)
  DIV_14 = (F34*K22*dna_out[91]*dna_out[15]*dna_out[30])
  DIV_18 = (3600*(km28 + dna_out[12]*dna_out[30]))
  DIV_19 = (3600*(km30 + dna_out[14]*dna_out[15])*(F39*dna_out[14] + 1)*(F38*dna_out[30] + 1))
  DIV_15 = (2*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[41]))

  if(DIV_1 == 0):
      DIV_1 = fixed

  if(DIV_2 == 0):
      DIV_2 = fixed
 
  if(DIV_5 == 0):
      DIV_5 = fixed

  if(DIV_6 == 0):
      DIV_6 = fixed

  if(DIV_7 == 0):
      DIV_7 = fixed

  if(DIV_8 == 0):
      DIV_8 = fixed

  if(DIV_9 == 0):
      DIV_9 = fixed    

  if(DIV_11 == 0):
      DIV_11 = fixed

  if(DIV_12 == 0):
      DIV_12 = fixed

  if(DIV_16 == 0):
      DIV_16 = fixed

  if(DIV_17 == 0):
      DIV_17 = fixed

  if(DIV_14 == 0):
      DIV_14 = fixed

  if(DIV_18 == 0):
      DIV_18 = fixed

  if(DIV_19 == 0):
      DIV_19 = fixed

  if(DIV_15 == 0):
      DIV_15 = fixed

  DIV_3 = ((F1*K2*dna_out[71]*dna_out[0])/DIV_1 - (F2*K2*dna_out[71]*dna_out[1])/DIV_2)

  if(DIV_3 == 0):
      DIV_3 = fixed

  DIV_4 = ((F1*K2*dna_out[71]*dna_out[0])/DIV_1 - (F2*K2*dna_out[71]*dna_out[1])/DIV_2)

  if(DIV_4 == 0):
      DIV_4 = fixed

  ROOT_1 = ((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41])/DIV_9)

  if(ROOT_1 < 0):
      ROOT_1 = 0.01

  DIV_10 = (7200*(F52*dna_out[41] + 1)*(km40 + (km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*(ROOT_1)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))/DIV_15)*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[41]))

  if(DIV_10 == 0):
      DIV_10 = fixed

  DIV_13 = (3600*(km29 + dna_out[15]*dna_out[30])*((3600*F35*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)*((K2*dna_out[71]*dna_out[0]*((F1*((K2*dna_out[71]*dna_out[0])/DIV_1 - (K2*dna_out[71]*dna_out[1])/DIV_2))/DIV_16 - 1))/DIV_1 - (K2*dna_out[71]*dna_out[1]*((F2*((K2*dna_out[71]*dna_out[0])/DIV_1 - (K2*dna_out[71]*dna_out[1])/DIV_2))/DIV_17 - 1))/DIV_2 + (K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/DIV_5 + (K5*dna_out[74]*dna_out[1])/DIV_6 - (K5*dna_out[74]*dna_out[3])/DIV_7 - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))/DIV_8 + (K31*dna_out[100]*km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*(ROOT_1)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))/DIV_10))/DIV_11 + 1))

  if(DIV_13 == 0):
      DIV_13 = fixed
    
  dna_out[126] = np.real((3600*(km29 + dna_out[15]*dna_out[30])*((3600*F35*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)*((K2*dna_out[71]*dna_out[0]*((F1*((K2*dna_out[71]*dna_out[0])/DIV_1 - (K2*dna_out[71]*dna_out[1])/DIV_2))/DIV_3 - 1))/DIV_1 - (K2*dna_out[71]*dna_out[1]*((F2*((K2*dna_out[71]*dna_out[0])/DIV_1 - (K2*dna_out[71]*dna_out[1])/DIV_2))/DIV_4 - 1))/DIV_2 + (K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/DIV_5 + (K5*dna_out[74]*dna_out[1])/DIV_6 - (K5*dna_out[74]*dna_out[3])/DIV_7 - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))/DIV_8 + (K31*dna_out[100]*km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*(ROOT_1)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))/DIV_10))/DIV_11 + 1)*((K30*dna_out[99]*dna_out[22]*dna_out[33])/DIV_12 - (K22*dna_out[91]*dna_out[15]*dna_out[30])/DIV_13 + (K21*dna_out[90]*dna_out[12]*dna_out[30]*(F33*dna_out[14] + 1))/DIV_18 - (K23*dna_out[92]*dna_out[14]*dna_out[15]*(F36*dna_out[14] + 1)*(F37*dna_out[15] + 1))/DIV_19))/DIV_14)
              
  
  dna_out[127] = np.real(-((K27*dna_out[96]*dna_out[20])/(3600*(km36 + dna_out[20])) - 3*((K27*dna_out[96]*dna_out[19]*dna_out[31])/(3600*(km35 + dna_out[19]*dna_out[31]))) + (K18*dna_out[87]*dna_out[12]*dna_out[33]*dna_out[34])/(3600*(km25 + dna_out[12]*dna_out[33]*dna_out[34])) - (K20*dna_out[89]*dna_out[14]*dna_out[31]*dna_out[32]*dna_out[38])/(3600*(km27 + dna_out[14]*dna_out[31]*dna_out[32]*dna_out[38])) + (K19*dna_out[88]*dna_out[30]*dna_out[33]*dna_out[36]*dna_out[37])/(3600*(km26 + dna_out[30]*dna_out[33]*dna_out[36]*dna_out[37])) - (K26*dna_out[95]*dna_out[18]*dna_out[33]*dna_out[39])/(3600*(km34 + dna_out[18]*dna_out[33]*dna_out[39])*(F47*dna_out[19] + 1)*(F46*dna_out[30] + 1)) + (K25*dna_out[94]*dna_out[17]*dna_out[33]*(F42*dna_out[17] + 1)*(F43*dna_out[31] + 1))/(3600*(km33 + dna_out[17]*dna_out[33])*(F44*dna_out[30] + 1)*(F45*dna_out[32] + 1)))/((F53*K18*dna_out[87]*dna_out[12]*dna_out[33]*dna_out[34])/(3600*(km25 + dna_out[12]*dna_out[33]*dna_out[34])) - (F55*K26*dna_out[95]*dna_out[18]*dna_out[33]*dna_out[39])/(3600*(km34 + dna_out[18]*dna_out[33]*dna_out[39])*(F47*dna_out[19] + 1)*(F46*dna_out[30] + 1)) + (F54*K25*dna_out[94]*dna_out[17]*dna_out[33]*(F42*dna_out[17] + 1)*(F43*dna_out[31] + 1))/(3600*(km33 + dna_out[17]*dna_out[33])*(F44*dna_out[30] + 1)*(F45*dna_out[32] + 1))))
  dna_out[128] = np.real(-((K24*dna_out[93]*dna_out[17])/(3600*km32 + 3600*dna_out[17]) - (K24*dna_out[93]*dna_out[16])/(3600*km31 + 3600*dna_out[16]) + (K23*dna_out[92]*dna_out[14]*dna_out[15]*(F36*dna_out[14] + 1)*(F37*dna_out[15] + 1))/((3600*km30 + 3600*dna_out[14]*dna_out[15])*(F39*dna_out[14] + 1)*(F38*dna_out[30] + 1)))/((F40*K24*dna_out[93]*dna_out[16])/(3600*km31 + 3600*dna_out[16]) - (F41*K24*dna_out[93]*dna_out[17])/(3600*km32 + 3600*dna_out[17])))
  dna_out[129] = np.real(((3600*km37 + 3600*dna_out[20]*dna_out[37])*((K27*dna_out[96]*dna_out[20])/(3600*km36 + 3600*dna_out[20]) - (K29*dna_out[98]*dna_out[21])/(3600*km38 + 3600*dna_out[21]) - 3*((K27*dna_out[96]*dna_out[19]*dna_out[31])/(3600*km35 + 3600*dna_out[19]*dna_out[31])) + (4*K28*dna_out[97]*dna_out[20]*dna_out[37])/(3600*km37 + 3600*dna_out[20]*dna_out[37]) - (2*K20*dna_out[89]*dna_out[14]*dna_out[31]*dna_out[32]*dna_out[38])/(3600*km27 + 3600*dna_out[14]*dna_out[31]*dna_out[32]*dna_out[38]) + (2*K19*dna_out[88]*dna_out[30]*dna_out[33]*dna_out[36]*dna_out[37])/(3600*km26 + 3600*dna_out[30]*dna_out[33]*dna_out[36]*dna_out[37])))/(4*F48*K28*dna_out[97]*dna_out[20]*dna_out[37]))
  dna_out[130] = np.real(((KS27*dna_out[45])/(60*(F_S30*dna_out[68] + 1)) - (KS26*dna_out[30])/(60*(F_S29*dna_out[69] + 1)) + (KS5*dna_out[44]*dna_out[68])/(60*(F_S4*dna_out[55] + 1)) + (K10*dna_out[79]*dna_out[4])/(3600*km11 + 3600*dna_out[4]) + (K7*dna_out[76]*dna_out[4])/((F19*dna_out[5] + 1)*(3600*km8 + 3600*dna_out[4])) - (K10*dna_out[79]*dna_out[6]*dna_out[7])/(3600*km12 + 3600*dna_out[6]*dna_out[7]) - (K6*dna_out[75]*dna_out[3]*dna_out[30]*(F15*dna_out[5] + 1))/((3600*km7 + 3600*dna_out[3]*dna_out[30])*(F17*dna_out[16] + 1)*(F16*dna_out[30] + 1)))/(KS16/60 + (F18*K7*dna_out[76]*dna_out[4])/((F19*dna_out[5] + 1)*(3600*km8 + 3600*dna_out[4])) + (F14_modified*K6*dna_out[75]*dna_out[3]*dna_out[30]*(F15*dna_out[5] + 1))/((3600*km7 + 3600*dna_out[3]*dna_out[30])*(F17*dna_out[16] + 1)*(F16*dna_out[30] + 1)))) 
  
            
  A = (3600*(km1 + dna_out[0]))
  B = (3600*(km2 + dna_out[1]))
  C = (3600*(km1 + dna_out[0]))
  D = (3600*(km2 + dna_out[1]))
  E = (3600*(km1 + dna_out[0]))
  F = (3600*(km1 + dna_out[0]))
  G = (3600*(km2 + dna_out[1]))
  H = (3600*(km1 + dna_out[0]))

  if(A == 0):
      A = fixed

  if(B == 0):
      B = fixed

  if(C == 0):
      C = fixed

  if(D == 0):
      D = fixed

  if(E == 0):
      E = fixed

  if(F == 0):
      F = fixed

  if(G == 0):
      G = fixed

  if(H == 0):
      H = fixed
            
  I = (3600*(km2 + dna_out[1]))

  if(I == 0):
      I = fixed

  J = (3600*(km2 + dna_out[1]))
  K = (3600*(km4 + dna_out[1]))
  L = (3600*(km5 + dna_out[1])*(F8*dna_out[4] + 1)*(F9*dna_out[24] + 1)*(F10*dna_out[29] + 1))
  M = (3600*(km6 + dna_out[3])*(F11*dna_out[4] + 1)*(F12*dna_out[24] + 1)*(F13*dna_out[29] + 1))
  N = (3600*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1))
  P = ((F1*K2*dna_out[71]*dna_out[0])/H - (F2*K2*dna_out[71]*dna_out[1])/I)
  Q = (12960000*(km43 + dna_out[25])*(km44 + dna_out[25]))
  RR = (2*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[41]))

  if(J == 0):
      J = fixed

  if(K == 0):
      K = fixed

  if(L == 0):
      L = fixed

  if(M == 0):
      M = fixed

  if(N == 0):
      N = fixed

  if(P == 0):
      P = fixed

  if(Q == 0):
      Q = fixed

  if(RR == 0):
      RR = fixed

  S = ((F1*K2*dna_out[71]*dna_out[0])/H - (F2*K2*dna_out[71]*dna_out[1])/I)
  T = ((F1*K2*dna_out[71]*dna_out[0])/H - (F2*K2*dna_out[71]*dna_out[1])/I)
  U = (3600*(km5 + dna_out[1])*(F8*dna_out[4] + 1)*(F9*dna_out[24] + 1)*(F10*dna_out[29] + 1))
  V = (3600*(km6 + dna_out[3])*(F11*dna_out[4] + 1)*(F12*dna_out[24] + 1)*(F13*dna_out[29] + 1))
  W = (3600*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1))
  X = (12960000*(km43 + dna_out[25])*(km44 + dna_out[25]))

  if(S == 0):
      S = fixed

  if(T == 0):
      T = fixed

  if(U == 0):
      U = fixed

  if(V == 0):
      V = fixed

  if(W == 0):
      W = fixed

  if(X == 0):
      X = fixed           

  Y = (7200*(F52*dna_out[41] + 1)*(km40 + (km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41])/X)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))/RR)*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[41]))
  Z = (3600*(km6 + dna_out[3])*(F11*dna_out[4] + 1)*(F12*dna_out[24] + 1)*(F13*dna_out[29] + 1))
  AA = (3600*(km5 + dna_out[1])*(F8*dna_out[4] + 1)*(F9*dna_out[24] + 1)*(F10*dna_out[29] + 1))
  BB = (12960000*(km43 + dna_out[25])*(km44 + dna_out[25]))
  CC = (7200*(F52*dna_out[41] + 1)*(km40 + (km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41])/Q)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))/RR)*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[41]))

  if(Y == 0):
      Y = fixed

  if(Z == 0):
      Z = fixed

  if(AA == 0):
      AA = fixed

  if(BB == 0):
      BB = fixed

  if(CC == 0):
      CC = fixed

  DD = (3600*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1))
  EE = (3600*(km6 + dna_out[3])*(F11*dna_out[4] + 1)*(F12*dna_out[24] + 1)*(F13*dna_out[29] + 1))

  if(DD == 0):
      DD = fixed

  if(EE == 0):
      EE = fixed

  FF = (3600*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1))
  GG = (2*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[41]))
  HH = (12960000*(km43 + dna_out[25])*(km44 + dna_out[25]))

  if(FF == 0):
      FF = fixed

  if(GG == 0):
      GG = fixed

  if(HH == 0):
      HH = fixed

  II = (3600*(km30 + dna_out[14]*dna_out[15])*(F39*dna_out[14] + 1)*(F38*dna_out[30] + 1))
  JJ = (F3*K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))
  KK = (3600*(km28 + dna_out[12]*dna_out[30]))
  LL = (F3*K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))

  if(II == 0):
      II = fixed

  if(JJ == 0):
      JJ = fixed

  if(KK == 0):
      KK = fixed

  if(LL == 0):
      LL = fixed

  MM = ((F1*K2*dna_out[71]*dna_out[0])/H - (F2*K2*dna_out[71]*dna_out[1])/I)
  NN = (7200*(F52*dna_out[41] + 1)*(km40 + (km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41])/HH))**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))/RR)*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[41])
  OO = ((F1*K2*dna_out[71]*dna_out[0])/H - (F2*K2*dna_out[71]*dna_out[1])/I)
  PP = (3600*(km5 + dna_out[1])*(F8*dna_out[4] + 1)*(F9*dna_out[24] + 1)*(F10*dna_out[29] + 1))
  QQ = (F34*K22*dna_out[91]*dna_out[15]*dna_out[30])
  SS = (3600*(km21 + dna_out[10]))
  TT = (3600*(km22 + dna_out[11]))
  UU = ((F1*K2*dna_out[71]*dna_out[0])/H - (F2*K2*dna_out[71]*dna_out[1])/I)
  VV = (3600*(km39 + dna_out[22]*dna_out[33])*(F51*dna_out[15] + 1))
  XXX = (4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41])
  YYY = (K31*dna_out[100]*km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41])/HH)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))
  ZZZ = (K31*dna_out[100]*km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41])/HH)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))
  WWW = (K31*dna_out[100]*km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41])/HH)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))
  UUU = (7200*(F52*dna_out[41] + 1)*(km40 + (km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41])/HH)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))/GG)*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[41]))

  if(MM == 0):
      MM = fixed

  if(NN == 0):
      NN = fixed

  if(OO == 0):
      OO = fixed

  if(PP == 0):
      PP = fixed

  if(QQ == 0):
      QQ = fixed

  if(SS == 0):
      SS = fixed

  if(TT == 0):
      TT = fixed

  if(UU == 0):
      UU = fixed

  if(VV == 0):
      VV = fixed

  if(XXX == 0):
      XXX = fixed

  if(YYY == 0):
      YYY = fixed

  if(ZZZ == 0):
      ZZZ = fixed

  if(WWW == 0):
      WWW = fixed

  if(UUU == 0):
      UUU = fixed

  CD = ((F1*K2*dna_out[71]*dna_out[0])/C - (F2*K2*dna_out[71]*dna_out[1])/D)

  if(CD == 0):
      CD = fixed   

  VVV = (3600*(km29 + dna_out[15]*dna_out[30])*((3600*F35*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)*((K2*dna_out[71]*dna_out[0]*((F1*((K2*dna_out[71]*dna_out[0])/H - (K2*dna_out[71]*dna_out[1])/I))/((F1*K2*dna_out[71]*dna_out[0])/H - (F2*K2*dna_out[71]*dna_out[1])/I) - 1))/H - (K2*dna_out[71]*dna_out[1]*((F2*((K2*dna_out[71]*dna_out[0])/H - (K2*dna_out[71]*dna_out[1])/I))/MM - 1))/I + (K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/(3600*(km4 + dna_out[1])) + (K5*dna_out[74]*dna_out[1])/AA - (K5*dna_out[74]*dna_out[3])/Z - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))/DD + YYY/UUU))/JJ + 1))

  if(VVV == 0):
      VVV = fixed

  XXXX = (7200*(F52*dna_out[41] + 1)*(km40 + (km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41])/HH)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))/RR)*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[41]))

  if(XXXX == 0):
      XXXX = fixed


  YYYY = (3600*(km29 + dna_out[15]*dna_out[30])*((3600*F35*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)*((K2*dna_out[71]*dna_out[0]*((F1*((K2*dna_out[71]*dna_out[0])/H - (K2*dna_out[71]*dna_out[1])/I))/UU - 1))/H - (K2*dna_out[71]*dna_out[1]*((F2*((K2*dna_out[71]*dna_out[0])/H - (K2*dna_out[71]*dna_out[1])/I))/OO - 1))/I + (K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/(3600*(km4 + dna_out[1])) + (K5*dna_out[74]*dna_out[1])/PP - (K5*dna_out[74]*dna_out[3])/EE - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))/(3600*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)) + (K31*dna_out[100]*km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*(XXX/HH)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))/(7200*(F52*dna_out[41] + 1)*(km40 + (km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41])/HH)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))/RR)*(2*K31*dna_out[100]*km42**2*dna_out[1]**2 + K33*dna_out[102]*km40**2*dna_out[24]**2 + F52*K33*dna_out[102]*km40**2*dna_out[24]**2*dna_out[41]))))/LL + 1))

  if(YYYY == 0):
      YYYY = fixed

  BBBB = (3600*(km29 + dna_out[15]*dna_out[30])*((3600*F35*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)*((K2*dna_out[71]*dna_out[0]*((F1*((K2*dna_out[71]*dna_out[0])/H - (K2*dna_out[71]*dna_out[1])/I))/UU - 1))/H - (K2*dna_out[71]*dna_out[1]*((F2*((K2*dna_out[71]*dna_out[0])/H - (K2*dna_out[71]*dna_out[1])/I))/UU - 1))/I + (K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/K + (K5*dna_out[74]*dna_out[1])/PP - (K5*dna_out[74]*dna_out[3])/EE - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))/FF + (K31*dna_out[100]*km40*km42*dna_out[1]*(2*K31*dna_out[100]*km42*dna_out[1] - 3600*((4*K31**2*dna_out[100]**2*km42**2*dna_out[1]**2*dna_out[25]**2 + K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*km44*dna_out[1]**2 + K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2 + 4*K31**2*dna_out[100]**2*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K31**2*dna_out[100]**2*km42**2*km44*dna_out[1]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25] + K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2 - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2 + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2 - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 8*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25] - 8*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25] + 4*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25] - 4*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25] + F52**2*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + F52**2*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*km44*dna_out[24]**2*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 2*F52*K33**2*dna_out[102]**2*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2 + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*dna_out[1]**2*dna_out[25]**2*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*dna_out[24]**2*dna_out[25]**2*dna_out[41] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25] + 4*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25] + 4*F52**2*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41]**2 - 4*F52**2*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41]**2 + 8*F52*K31*K34*dna_out[100]*dna_out[103]*km42**2*km44*dna_out[1]**2*dna_out[25]*dna_out[41] - 8*F52*K31*K35*dna_out[100]*dna_out[104]*km42**2*km43*dna_out[1]**2*dna_out[25]*dna_out[41] + 8*F52*K33*K34*dna_out[102]*dna_out[103]*km40**2*km44*dna_out[24]**2*dna_out[25]*dna_out[41] - 8*F52*K33*K35*dna_out[102]*dna_out[104]*km40**2*km43*dna_out[24]**2*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*dna_out[1]*dna_out[24]*dna_out[25]**2*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*km44*dna_out[1]*dna_out[24]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km43*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41] + 4*F52*K31*K33*dna_out[100]*dna_out[102]*km40*km42*km44*dna_out[1]*dna_out[24]*dna_out[25]*dna_out[41])/HH)**(1/2) + K33*dna_out[102]*km40*dna_out[24] + F52*K33*dna_out[102]*km40*dna_out[24]*dna_out[41]))/XXXX))/LL + 1))

  if(BBBB == 0):
      BBBB = fixed


  AAAA = (K22*dna_out[91]*dna_out[15]*dna_out[30])

  if(AAAA == 0):
      AAAA = fixed

  WWWW = (3600*(km23 + dna_out[11]*dna_out[31])*(F29*dna_out[14] + 1)*(F28*dna_out[30] + 1)*(F32*dna_out[36] + 1)*((3600*F31*(km29 + dna_out[15]*dna_out[30])*((3600*F35*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)*((K2*dna_out[71]*dna_out[0]*((F1*((K2*dna_out[71]*dna_out[0])/H - (K2*dna_out[71]*dna_out[1])/I))/S - 1))/H - (K2*dna_out[71]*dna_out[1]*((F2*((K2*dna_out[71]*dna_out[0])/H - (K2*dna_out[71]*dna_out[1])/I))/T - 1))/I + (K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/K + (K5*dna_out[74]*dna_out[1])/U - (K5*dna_out[74]*dna_out[3])/V - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))/W + WWW/Y))/LL + 1)*((K30*dna_out[99]*dna_out[22]*dna_out[33])/VV - AAAA/VVV + (K21*dna_out[90]*dna_out[12]*dna_out[30]*(F33*dna_out[14] + 1))/KK - (K23*dna_out[92]*dna_out[14]*dna_out[15]*(F36*dna_out[14] + 1)*(F37*dna_out[15] + 1))/II))/QQ + 1)*((K15*dna_out[84]*dna_out[10])/SS - (K15*dna_out[84]*dna_out[11])/TT + (K22*dna_out[91]*dna_out[15]*dna_out[30]*((3600*(km29 + dna_out[15]*dna_out[30])*((3600*F35*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)*((K2*dna_out[71]*dna_out[0]*((F1*((K2*dna_out[71]*dna_out[0])/H - (K2*dna_out[71]*dna_out[1])/I))/UU - 1))/H - (K2*dna_out[71]*dna_out[1]*((F2*((K2*dna_out[71]*dna_out[0])/H - (K2*dna_out[71]*dna_out[1])/I))/UU - 1))/I + (K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/K + (K5*dna_out[74]*dna_out[1])/PP - (K5*dna_out[74]*dna_out[3])/EE - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))/FF + YYY/NN))/LL + 1)*((K30*dna_out[99]*dna_out[22]*dna_out[33])/VV - AAAA/YYYY + (K21*dna_out[90]*dna_out[12]*dna_out[30]*(F33*dna_out[14] + 1))/KK - (K23*dna_out[92]*dna_out[14]*dna_out[15]*(F36*dna_out[14] + 1)*(F37*dna_out[15] + 1))/II))/AAAA + 1))/BBBB))

  if(WWWW == 0):
      WWWW = fixed

  dna_out[131] = np.real(((K16*dna_out[85]*dna_out[11]*dna_out[31]*(F27*dna_out[4] + 1)*((3600*F26*(km3 + dna_out[2]*dna_out[30])*(F5*dna_out[1] + 1)*(F6*dna_out[14] + 1)*((K2*dna_out[71]*dna_out[0]*((F1*((K2*dna_out[71]*dna_out[0])/A - (K2*dna_out[71]*dna_out[1])/B))/CD - 1))/E - (K2*dna_out[71]*dna_out[1]*((F2*((K2*dna_out[71]*dna_out[0])/F - (K2*dna_out[71]*dna_out[1])/G))/P - 1))/J + (K4*dna_out[73]*dna_out[1]*(F7*dna_out[1] + 1))/K + (K5*dna_out[74]*dna_out[1])/L - (K5*dna_out[74]*dna_out[3])/M - (K3*dna_out[72]*dna_out[2]*dna_out[30]*(F4*dna_out[31] + 1))/N + ZZZ/CC))/LL + 1))/WWWW - 1)/F30)
  
  dna_out[132] = np.real(-((K2*dna_out[71]*dna_out[0])/H - (K2*dna_out[71]*dna_out[1])/I)/UU)
  dna_out[133] = np.real(-((K13*dna_out[82]*dna_out[9])/(3600*(km18 + dna_out[9])) + (K14*dna_out[83]*dna_out[9])/(1800*(km19 + dna_out[9])) - (K14*dna_out[83]*dna_out[10])/(1800*(km20 + dna_out[10])) - (K15*dna_out[84]*dna_out[10])/(3600*(km21 + dna_out[10])) + (K15*dna_out[84]*dna_out[11])/(3600*(km22 + dna_out[11])) - (K13*dna_out[82]*dna_out[8]*dna_out[31])/(3600*(km17 + dna_out[8]*dna_out[31])))/((F24*K14*dna_out[83]*dna_out[9])/(1800*(km19 + dna_out[9])) - (F25*K14*dna_out[83]*dna_out[10])/(1800*(km20 + dna_out[10]))))
   
  DNA_SIZE = 134
  for i in range(0,DNA_SIZE):
      
      if(dna_out[i] < 0):
          dna_out[i] = 0.001
            
      if(dna_out[i] > 1):
          dna_out[i] = 0.999


  return dna_out



def genetic_controller(iter, st, u_optimal_1, U_Candidate_2, y_pred_2, y_actual_2, y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train):
    
  POP_SIZE    = 20
  GENERATIONS = 5

  Q = 134                     # Number of input of the Biological MIMO system
  R = 107                     # Number of output of the Biological MIMO system
  num_of_past_input = 19
  num_of_input_to_SVM = (Q+R)*num_of_past_input + Q

  qq = 0
  final = 0  
  population = np.zeros((POP_SIZE,Q))

  y_pred_val = np.zeros((POP_SIZE,R))
  y_pred_val_repeat = np.zeros((POP_SIZE,R))
  
  U_Candidate_backup2 = np.zeros((1,num_of_input_to_SVM))
  y_pred_backup2 = np.zeros((1,R))
  y_actual_backup2 = np.zeros((1,R))


  for col in range(0,num_of_input_to_SVM):
     U_Candidate_backup2[0,col] = U_Candidate_2[0,col]
     
  for col in range(0,R):
     y_pred_backup2[0,col] = y_pred_2[0,col]
     
  for col in range(0,R):
     y_actual_backup2[0,col] = y_actual_2[0,col]
  
  population = random_population(iter, u_optimal_1)


  #write_data(population,'initial_population.xlsx')
  #print("population written into excel file...\n\n")

  #  print('abnormal exit')
  #  sys.exit()



  All_Population[:,:,iter] = population
  Population_last_to_check = population
  
  
  
  # Simulate all of the generations.
  fittest_solution = np.zeros((1,Q))
  fittest_solution_previous = np.zeros((1,Q))
  maximum_fitness = 0
  maximum_fitness_prev = 0
  threshold = 0.1
#  flag = 1
  for generation in range(GENERATIONS):
      print("Generation ",generation)
      weighted_population = []

      # Add individuals and their respective fitness levels to the weighted
      # population list. This will be used to pull out individuals via certain
      # probabilities during the selection phase. Then, reset the population list
      # so we can repopulate it after selection.
      fitness_val = np.zeros((POP_SIZE,1))
      pp = 0
      iiii = 0
      for individual in population:   
                 
          print('fitness of indivdual ',iiii,' started')       
          iiii = iiii + 1
          fitness_val[pp,0],U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val[pp,:],iii = fitness(individual, U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)

          # Generate the (individual,fitness) pair, taking in account whether or
          # not we will accidently divide by zero.
          pair = (individual, 1.0/fitness_val[pp,0])
          weighted_population.append(pair)
          pp = pp + 1
          
      population = np.zeros((POP_SIZE,Q))
#      print("\nStarting crossover ")

      # Select two random individuals, based on their fitness probabilites, cross
      # their genes over at a random point, mutate them, and add them back to the
      # population for the next iteration.
      pp = 0
      for x in range(int(POP_SIZE/2)):
          # Selection
          ind1 = weighted_choice(weighted_population)
          ind2 = weighted_choice(weighted_population)
#          print("size of ind1 = ",ind1.shape,"\n")
          # Crossover
          ind1_mod, ind2_mod = crossover(ind1, ind2)
#          print("size of ind1_mod = ",ind1_mod.shape,"\n")
          # Mutate and add back into the population.
          population[pp] = mutate(ind1_mod)
          population[pp+1] = mutate(ind2_mod)         
          pp = pp + 2

#      print("\nEnding crossover ")


      # Display the highest-ranked string after all generations have been iterated
      # over. This will be the closest string to the OPTIMAL string, meaning it
      # will have the smallest fitness value. Finally, exit the program.
      fittest_solution = population[0,:]
      
      maximum_fitness,U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val_repeat[0,:],iii = fitness(population[0,:], U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)
#      print("\nStart finding fittest solution ")
      
      
      
      qq = 0
      final = 0
      for individual in population:    
          ind_fitness,U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val_repeat[0,:],iii = fitness(population[qq,:], U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)
          
          
          if ind_fitness > maximum_fitness:
              fittest_solution = individual
              maximum_fitness = ind_fitness
              final = qq
          qq = qq + 1

      print("Fittest solution: ",fittest_solution)
      print("Fitness value is: ", maximum_fitness)

      if((1/maximum_fitness) < threshold):
          break    
      else:
          if(generation > 0 and (maximum_fitness_prev - maximum_fitness) > 0):
              fittest_solution = fittest_solution_previous
              maximum_fitness = maximum_fitness_prev
              print("\nGeneration creation interrupted...!!! previous maximum fit solution returned...\n")
              break
          else:
              fittest_solution_previous = fittest_solution
              maximum_fitness_prev = maximum_fitness
          
  return maximum_fitness,fittest_solution, y_pred_2, y_pred_val[final,:], Population_last_to_check, y_actual,iii,Population_last_to_check

def genetic_controller2(iter, st, u_optimal_1, U_Candidate_2, y_pred_2, y_actual_2, y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train):
    
  POP_SIZE    = 20
  GENERATIONS = 5

  Q = 134                     # Number of input of the Biological MIMO system
  R = 107                     # Number of output of the Biological MIMO system
  num_of_past_input = 19
  num_of_input_to_SVM = (Q+R)*num_of_past_input + Q

  qq = 0
  final = 0  
  population = np.zeros((POP_SIZE,Q))

  y_pred_val = np.zeros((POP_SIZE,R))
  y_pred_val_repeat = np.zeros((POP_SIZE,R))
  
  U_Candidate_backup2 = np.zeros((1,num_of_input_to_SVM))
  y_pred_backup2 = np.zeros((1,R))
  y_actual_backup2 = np.zeros((1,R))


  for col in range(0,num_of_input_to_SVM):
     U_Candidate_backup2[0,col] = U_Candidate_2[0,col]
     
  for col in range(0,R):
     y_pred_backup2[0,col] = y_pred_2[0,col]
     
  for col in range(0,R):
     y_actual_backup2[0,col] = y_actual_2[0,col]
  
  population = random_population(iter, u_optimal_1)


  #write_data(population,'initial_population.xlsx')
  #print("population written into excel file...\n\n")

  #  print('abnormal exit')
  #  sys.exit()



  All_Population[:,:,iter] = population
  Population_last_to_check = population
  
  
  
  # Simulate all of the generations.
  fittest_solution = np.zeros((1,Q))
  fittest_solution_previous = np.zeros((1,Q))
  maximum_fitness = 0
  maximum_fitness_prev = 0
  threshold = 0.1
#  flag = 1
  for generation in range(GENERATIONS):
      print("Generation ",generation)
      weighted_population = []

      # Add individuals and their respective fitness levels to the weighted
      # population list. This will be used to pull out individuals via certain
      # probabilities during the selection phase. Then, reset the population list
      # so we can repopulate it after selection.
      fitness_val = np.zeros((POP_SIZE,1))
      pp = 0
      iiii = 0
      for individual in population:   
                 
          print('fitness of indivdual ',iiii,' started')       
          iiii = iiii + 1
          fitness_val[pp,0],U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val[pp,:],iii = fitness2(individual, U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)

          # Generate the (individual,fitness) pair, taking in account whether or
          # not we will accidently divide by zero.
          pair = (individual, 1.0/fitness_val[pp,0])
          weighted_population.append(pair)
          pp = pp + 1
          
      population = np.zeros((POP_SIZE,Q))
#      print("\nStarting crossover ")

      # Select two random individuals, based on their fitness probabilites, cross
      # their genes over at a random point, mutate them, and add them back to the
      # population for the next iteration.
      pp = 0
      for x in range(int(POP_SIZE/2)):
          # Selection
          ind1 = weighted_choice(weighted_population)
          ind2 = weighted_choice(weighted_population)
#          print("size of ind1 = ",ind1.shape,"\n")
          # Crossover
          ind1_mod, ind2_mod = crossover(ind1, ind2)
#          print("size of ind1_mod = ",ind1_mod.shape,"\n")
          # Mutate and add back into the population.
          population[pp] = mutate(ind1_mod)
          population[pp+1] = mutate(ind2_mod)         
          pp = pp + 2

#      print("\nEnding crossover ")


      # Display the highest-ranked string after all generations have been iterated
      # over. This will be the closest string to the OPTIMAL string, meaning it
      # will have the smallest fitness value. Finally, exit the program.
      fittest_solution = population[0,:]
      
      maximum_fitness,U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val_repeat[0,:],iii = fitness2(population[0,:], U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)
#      print("\nStart finding fittest solution ")
      
      
      
      qq = 0
      final = 0
      for individual in population:    
          ind_fitness,U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val_repeat[0,:],iii = fitness2(population[qq,:], U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)
          
          
          if ind_fitness > maximum_fitness:
              fittest_solution = individual
              maximum_fitness = ind_fitness
              final = qq
          qq = qq + 1

      print("Fittest solution: ",fittest_solution)
      print("Fitness value is: ", maximum_fitness)

      if((1/maximum_fitness) < threshold):
          break    
      else:
          if(generation > 0 and (maximum_fitness_prev - maximum_fitness) > 0):
              fittest_solution = fittest_solution_previous
              maximum_fitness = maximum_fitness_prev
              print("\nGeneration creation interrupted...!!! previous maximum fit solution returned...\n")
              break
          else:
              fittest_solution_previous = fittest_solution
              maximum_fitness_prev = maximum_fitness
          
  return maximum_fitness,fittest_solution, y_pred_2, y_pred_val[final,:], Population_last_to_check, y_actual,iii,Population_last_to_check


def genetic_controller3(iter, st, u_optimal_1, U_Candidate_2, y_pred_2, y_actual_2, y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train):
    
  POP_SIZE    = 20
  GENERATIONS = 5

  Q = 134                     # Number of input of the Biological MIMO system
  R = 107                     # Number of output of the Biological MIMO system
  num_of_past_input = 19
  num_of_input_to_SVM = (Q+R)*num_of_past_input + Q

  qq = 0
  final = 0  
  population = np.zeros((POP_SIZE,Q))

  y_pred_val = np.zeros((POP_SIZE,R))
  y_pred_val_repeat = np.zeros((POP_SIZE,R))
  
  U_Candidate_backup2 = np.zeros((1,num_of_input_to_SVM))
  y_pred_backup2 = np.zeros((1,R))
  y_actual_backup2 = np.zeros((1,R))


  for col in range(0,num_of_input_to_SVM):
     U_Candidate_backup2[0,col] = U_Candidate_2[0,col]
     
  for col in range(0,R):
     y_pred_backup2[0,col] = y_pred_2[0,col]
     
  for col in range(0,R):
     y_actual_backup2[0,col] = y_actual_2[0,col]
  
  population = random_population(iter, u_optimal_1)


  #write_data(population,'initial_population.xlsx')
  #print("population written into excel file...\n\n")

  #  print('abnormal exit')
  #  sys.exit()



  All_Population[:,:,iter] = population
  Population_last_to_check = population
  
  
  
  # Simulate all of the generations.
  fittest_solution = np.zeros((1,Q))
  fittest_solution_previous = np.zeros((1,Q))
  maximum_fitness = 0
  maximum_fitness_prev = 0
  threshold = 0.1
#  flag = 1
  for generation in range(GENERATIONS):
      print("Generation ",generation)
      weighted_population = []

      # Add individuals and their respective fitness levels to the weighted
      # population list. This will be used to pull out individuals via certain
      # probabilities during the selection phase. Then, reset the population list
      # so we can repopulate it after selection.
      fitness_val = np.zeros((POP_SIZE,1))
      pp = 0
      iiii = 0
      for individual in population:   
                 
          print('fitness of indivdual ',iiii,' started')       
          iiii = iiii + 1
          fitness_val[pp,0],U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val[pp,:],iii = fitness3(individual, U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)

          # Generate the (individual,fitness) pair, taking in account whether or
          # not we will accidently divide by zero.
          pair = (individual, 1.0/fitness_val[pp,0])
          weighted_population.append(pair)
          pp = pp + 1
          
      population = np.zeros((POP_SIZE,Q))
#      print("\nStarting crossover ")

      # Select two random individuals, based on their fitness probabilites, cross
      # their genes over at a random point, mutate them, and add them back to the
      # population for the next iteration.
      pp = 0
      for x in range(int(POP_SIZE/2)):
          # Selection
          ind1 = weighted_choice(weighted_population)
          ind2 = weighted_choice(weighted_population)
#          print("size of ind1 = ",ind1.shape,"\n")
          # Crossover
          ind1_mod, ind2_mod = crossover(ind1, ind2)
#          print("size of ind1_mod = ",ind1_mod.shape,"\n")
          # Mutate and add back into the population.
          population[pp] = mutate(ind1_mod)
          population[pp+1] = mutate(ind2_mod)         
          pp = pp + 2

#      print("\nEnding crossover ")


      # Display the highest-ranked string after all generations have been iterated
      # over. This will be the closest string to the OPTIMAL string, meaning it
      # will have the smallest fitness value. Finally, exit the program.
      fittest_solution = population[0,:]
      
      maximum_fitness,U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val_repeat[0,:],iii = fitness3(population[0,:], U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)
#      print("\nStart finding fittest solution ")
      
      
      
      qq = 0
      final = 0
      for individual in population:    
          ind_fitness,U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val_repeat[0,:],iii = fitness3(population[qq,:], U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)
          
          
          if ind_fitness > maximum_fitness:
              fittest_solution = individual
              maximum_fitness = ind_fitness
              final = qq
          qq = qq + 1

      print("Fittest solution: ",fittest_solution)
      print("Fitness value is: ", maximum_fitness)

      if((1/maximum_fitness) < threshold):
          break    
      else:
          if(generation > 0 and (maximum_fitness_prev - maximum_fitness) > 0):
              fittest_solution = fittest_solution_previous
              maximum_fitness = maximum_fitness_prev
              print("\nGeneration creation interrupted...!!! previous maximum fit solution returned...\n")
              break
          else:
              fittest_solution_previous = fittest_solution
              maximum_fitness_prev = maximum_fitness
          
  return maximum_fitness,fittest_solution, y_pred_2, y_pred_val[final,:], Population_last_to_check, y_actual,iii,Population_last_to_check


def genetic_controller4(iter, st, u_optimal_1, U_Candidate_2, y_pred_2, y_actual_2, y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train):
    
  POP_SIZE    = 20
  GENERATIONS = 5

  Q = 134                     # Number of input of the Biological MIMO system
  R = 107                     # Number of output of the Biological MIMO system
  num_of_past_input = 19
  num_of_input_to_SVM = (Q+R)*num_of_past_input + Q

  qq = 0
  final = 0  
  population = np.zeros((POP_SIZE,Q))

  y_pred_val = np.zeros((POP_SIZE,R))
  y_pred_val_repeat = np.zeros((POP_SIZE,R))
  
  U_Candidate_backup2 = np.zeros((1,num_of_input_to_SVM))
  y_pred_backup2 = np.zeros((1,R))
  y_actual_backup2 = np.zeros((1,R))


  for col in range(0,num_of_input_to_SVM):
     U_Candidate_backup2[0,col] = U_Candidate_2[0,col]
     
  for col in range(0,R):
     y_pred_backup2[0,col] = y_pred_2[0,col]
     
  for col in range(0,R):
     y_actual_backup2[0,col] = y_actual_2[0,col]
  
  population = random_population(iter, u_optimal_1)


  #write_data(population,'initial_population.xlsx')
  #print("population written into excel file...\n\n")

  #  print('abnormal exit')
  #  sys.exit()



  All_Population[:,:,iter] = population
  Population_last_to_check = population
  
  
  
  # Simulate all of the generations.
  fittest_solution = np.zeros((1,Q))
  fittest_solution_previous = np.zeros((1,Q))
  maximum_fitness = 0
  maximum_fitness_prev = 0
  threshold = 0.1
#  flag = 1
  for generation in range(GENERATIONS):
      print("Generation ",generation)
      weighted_population = []

      # Add individuals and their respective fitness levels to the weighted
      # population list. This will be used to pull out individuals via certain
      # probabilities during the selection phase. Then, reset the population list
      # so we can repopulate it after selection.
      fitness_val = np.zeros((POP_SIZE,1))
      pp = 0
      iiii = 0
      for individual in population:   
                 
          print('fitness of indivdual ',iiii,' started')       
          iiii = iiii + 1
          fitness_val[pp,0],U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val[pp,:],iii = fitness4(individual, U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)

          # Generate the (individual,fitness) pair, taking in account whether or
          # not we will accidently divide by zero.
          pair = (individual, 1.0/fitness_val[pp,0])
          weighted_population.append(pair)
          pp = pp + 1
          
      population = np.zeros((POP_SIZE,Q))
#      print("\nStarting crossover ")

      # Select two random individuals, based on their fitness probabilites, cross
      # their genes over at a random point, mutate them, and add them back to the
      # population for the next iteration.
      pp = 0
      for x in range(int(POP_SIZE/2)):
          # Selection
          ind1 = weighted_choice(weighted_population)
          ind2 = weighted_choice(weighted_population)
#          print("size of ind1 = ",ind1.shape,"\n")
          # Crossover
          ind1_mod, ind2_mod = crossover(ind1, ind2)
#          print("size of ind1_mod = ",ind1_mod.shape,"\n")
          # Mutate and add back into the population.
          population[pp] = mutate(ind1_mod)
          population[pp+1] = mutate(ind2_mod)         
          pp = pp + 2

#      print("\nEnding crossover ")


      # Display the highest-ranked string after all generations have been iterated
      # over. This will be the closest string to the OPTIMAL string, meaning it
      # will have the smallest fitness value. Finally, exit the program.
      fittest_solution = population[0,:]
      
      maximum_fitness,U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val_repeat[0,:],iii = fitness4(population[0,:], U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)
#      print("\nStart finding fittest solution ")
      
      
      
      qq = 0
      final = 0
      for individual in population:    
          ind_fitness,U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val_repeat[0,:],iii = fitness4(population[qq,:], U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)
          
          
          if ind_fitness > maximum_fitness:
              fittest_solution = individual
              maximum_fitness = ind_fitness
              final = qq
          qq = qq + 1

      print("Fittest solution: ",fittest_solution)
      print("Fitness value is: ", maximum_fitness)

      if((1/maximum_fitness) < threshold):
          break    
      else:
          if(generation > 0 and (maximum_fitness_prev - maximum_fitness) > 0):
              fittest_solution = fittest_solution_previous
              maximum_fitness = maximum_fitness_prev
              print("\nGeneration creation interrupted...!!! previous maximum fit solution returned...\n")
              break
          else:
              fittest_solution_previous = fittest_solution
              maximum_fitness_prev = maximum_fitness
          
  return maximum_fitness,fittest_solution, y_pred_2, y_pred_val[final,:], Population_last_to_check, y_actual,iii,Population_last_to_check



def genetic_controller5(iter, st, u_optimal_1, U_Candidate_2, y_pred_2, y_actual_2, y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train):
    
  POP_SIZE    = 20
  GENERATIONS = 5

  Q = 134                     # Number of input of the Biological MIMO system
  R = 107                     # Number of output of the Biological MIMO system
  num_of_past_input = 19
  num_of_input_to_SVM = (Q+R)*num_of_past_input + Q

  qq = 0
  final = 0  
  population = np.zeros((POP_SIZE,Q))

  y_pred_val = np.zeros((POP_SIZE,R))
  y_pred_val_repeat = np.zeros((POP_SIZE,R))
  
  U_Candidate_backup2 = np.zeros((1,num_of_input_to_SVM))
  y_pred_backup2 = np.zeros((1,R))
  y_actual_backup2 = np.zeros((1,R))


  for col in range(0,num_of_input_to_SVM):
     U_Candidate_backup2[0,col] = U_Candidate_2[0,col]
     
  for col in range(0,R):
     y_pred_backup2[0,col] = y_pred_2[0,col]
     
  for col in range(0,R):
     y_actual_backup2[0,col] = y_actual_2[0,col]
  
  population = random_population(iter, u_optimal_1)


  #write_data(population,'initial_population.xlsx')
  #print("population written into excel file...\n\n")

  #  print('abnormal exit')
  #  sys.exit()



  All_Population[:,:,iter] = population
  Population_last_to_check = population
  
  
  
  # Simulate all of the generations.
  fittest_solution = np.zeros((1,Q))
  fittest_solution_previous = np.zeros((1,Q))
  maximum_fitness = 0
  maximum_fitness_prev = 0
  threshold = 0.1
#  flag = 1
  for generation in range(GENERATIONS):
      print("Generation ",generation)
      weighted_population = []

      # Add individuals and their respective fitness levels to the weighted
      # population list. This will be used to pull out individuals via certain
      # probabilities during the selection phase. Then, reset the population list
      # so we can repopulate it after selection.
      fitness_val = np.zeros((POP_SIZE,1))
      pp = 0
      iiii = 0
      for individual in population:   
                 
          print('fitness of indivdual ',iiii,' started')       
          iiii = iiii + 1
          fitness_val[pp,0],U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val[pp,:],iii = fitness5(individual, U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)

          # Generate the (individual,fitness) pair, taking in account whether or
          # not we will accidently divide by zero.
          pair = (individual, 1.0/fitness_val[pp,0])
          weighted_population.append(pair)
          pp = pp + 1
          
      population = np.zeros((POP_SIZE,Q))
#      print("\nStarting crossover ")

      # Select two random individuals, based on their fitness probabilites, cross
      # their genes over at a random point, mutate them, and add them back to the
      # population for the next iteration.
      pp = 0
      for x in range(int(POP_SIZE/2)):
          # Selection
          ind1 = weighted_choice(weighted_population)
          ind2 = weighted_choice(weighted_population)
#          print("size of ind1 = ",ind1.shape,"\n")
          # Crossover
          ind1_mod, ind2_mod = crossover(ind1, ind2)
#          print("size of ind1_mod = ",ind1_mod.shape,"\n")
          # Mutate and add back into the population.
          population[pp] = mutate(ind1_mod)
          population[pp+1] = mutate(ind2_mod)         
          pp = pp + 2

#      print("\nEnding crossover ")


      # Display the highest-ranked string after all generations have been iterated
      # over. This will be the closest string to the OPTIMAL string, meaning it
      # will have the smallest fitness value. Finally, exit the program.
      fittest_solution = population[0,:]
      
      maximum_fitness,U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val_repeat[0,:],iii = fitness5(population[0,:], U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)
#      print("\nStart finding fittest solution ")
      
      
      
      qq = 0
      final = 0
      for individual in population:    
          ind_fitness,U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val_repeat[0,:],iii = fitness5(population[qq,:], U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)
          
          
          if ind_fitness > maximum_fitness:
              fittest_solution = individual
              maximum_fitness = ind_fitness
              final = qq
          qq = qq + 1

      print("Fittest solution: ",fittest_solution)
      print("Fitness value is: ", maximum_fitness)

      if((1/maximum_fitness) < threshold):
          break    
      else:
          if(generation > 0 and (maximum_fitness_prev - maximum_fitness) > 0):
              fittest_solution = fittest_solution_previous
              maximum_fitness = maximum_fitness_prev
              print("\nGeneration creation interrupted...!!! previous maximum fit solution returned...\n")
              break
          else:
              fittest_solution_previous = fittest_solution
              maximum_fitness_prev = maximum_fitness
          
  return maximum_fitness,fittest_solution, y_pred_2, y_pred_val[final,:], Population_last_to_check, y_actual,iii,Population_last_to_check




def genetic_controller6(iter, st, u_optimal_1, U_Candidate_2, y_pred_2, y_actual_2, y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train):
    
  POP_SIZE    = 20
  GENERATIONS = 5

  Q = 134                     # Number of input of the Biological MIMO system
  R = 107                     # Number of output of the Biological MIMO system
  num_of_past_input = 19
  num_of_input_to_SVM = (Q+R)*num_of_past_input + Q

  qq = 0
  final = 0  
  population = np.zeros((POP_SIZE,Q))

  y_pred_val = np.zeros((POP_SIZE,R))
  y_pred_val_repeat = np.zeros((POP_SIZE,R))
  
  U_Candidate_backup2 = np.zeros((1,num_of_input_to_SVM))
  y_pred_backup2 = np.zeros((1,R))
  y_actual_backup2 = np.zeros((1,R))


  for col in range(0,num_of_input_to_SVM):
     U_Candidate_backup2[0,col] = U_Candidate_2[0,col]
     
  for col in range(0,R):
     y_pred_backup2[0,col] = y_pred_2[0,col]
     
  for col in range(0,R):
     y_actual_backup2[0,col] = y_actual_2[0,col]
  
  population = random_population(iter, u_optimal_1)


  #write_data(population,'initial_population.xlsx')
  #print("population written into excel file...\n\n")

  #  print('abnormal exit')
  #  sys.exit()



  All_Population[:,:,iter] = population
  Population_last_to_check = population
  
  
  
  # Simulate all of the generations.
  fittest_solution = np.zeros((1,Q))
  fittest_solution_previous = np.zeros((1,Q))
  maximum_fitness = 0
  maximum_fitness_prev = 0
  threshold = 0.1
#  flag = 1
  for generation in range(GENERATIONS):
      print("Generation ",generation)
      weighted_population = []

      # Add individuals and their respective fitness levels to the weighted
      # population list. This will be used to pull out individuals via certain
      # probabilities during the selection phase. Then, reset the population list
      # so we can repopulate it after selection.
      fitness_val = np.zeros((POP_SIZE,1))
      pp = 0
      iiii = 0
      for individual in population:   
                 
          print('fitness of indivdual ',iiii,' started')       
          iiii = iiii + 1
          fitness_val[pp,0],U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val[pp,:],iii = fitness6(individual, U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)

          # Generate the (individual,fitness) pair, taking in account whether or
          # not we will accidently divide by zero.
          pair = (individual, 1.0/fitness_val[pp,0])
          weighted_population.append(pair)
          pp = pp + 1
          
      population = np.zeros((POP_SIZE,Q))
#      print("\nStarting crossover ")

      # Select two random individuals, based on their fitness probabilites, cross
      # their genes over at a random point, mutate them, and add them back to the
      # population for the next iteration.
      pp = 0
      for x in range(int(POP_SIZE/2)):
          # Selection
          ind1 = weighted_choice(weighted_population)
          ind2 = weighted_choice(weighted_population)
#          print("size of ind1 = ",ind1.shape,"\n")
          # Crossover
          ind1_mod, ind2_mod = crossover(ind1, ind2)
#          print("size of ind1_mod = ",ind1_mod.shape,"\n")
          # Mutate and add back into the population.
          population[pp] = mutate(ind1_mod)
          population[pp+1] = mutate(ind2_mod)         
          pp = pp + 2

#      print("\nEnding crossover ")


      # Display the highest-ranked string after all generations have been iterated
      # over. This will be the closest string to the OPTIMAL string, meaning it
      # will have the smallest fitness value. Finally, exit the program.
      fittest_solution = population[0,:]
      
      maximum_fitness,U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val_repeat[0,:],iii = fitness6(population[0,:], U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)
#      print("\nStart finding fittest solution ")
      
      
      
      qq = 0
      final = 0
      for individual in population:    
          ind_fitness,U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_pred_val_repeat[0,:],iii = fitness6(population[qq,:], U_Candidate_backup2,y_pred_backup2,y_actual_backup2,y_ref_for_fixed_first_22,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)
          
          
          if ind_fitness > maximum_fitness:
              fittest_solution = individual
              maximum_fitness = ind_fitness
              final = qq
          qq = qq + 1

      print("Fittest solution: ",fittest_solution)
      print("Fitness value is: ", maximum_fitness)

      if((1/maximum_fitness) < threshold):
          break    
      else:
          if(generation > 0 and (maximum_fitness_prev - maximum_fitness) > 0):
              fittest_solution = fittest_solution_previous
              maximum_fitness = maximum_fitness_prev
              print("\nGeneration creation interrupted...!!! previous maximum fit solution returned...\n")
              break
          else:
              fittest_solution_previous = fittest_solution
              maximum_fitness_prev = maximum_fitness
          
  return maximum_fitness,fittest_solution, y_pred_2, y_pred_val[final,:], Population_last_to_check, y_actual,iii,Population_last_to_check



def initial_processing():

    R = 107                     # Number of output of the Biological MIMO system

    tr = heli_data()                                          
    tr.Create_batch_train()                                   #Create the batch for training data
    tr1 = heli_data()
    tr1.Create_initial_u_candidate()                          #Create the batch for u_candidate

    X_train = tr.inputs                                         #Fetch train inputs
    Y_train = tr.outputs                                         #Fetch train outputs

    U_Candidate = tr1.inputs                                     #fetch initial U_candidate
    print("Initial Candidate Input is extracted...\t\t\t{:.2f}s".format(tm.time()-st))

    y_pred = SVM_model_predict_output(U_Candidate)

    #### reference creation initially
    y_ref = np.array([y_pred[0,0], y_pred[0,1],y_pred[0,2],y_pred[0,3],y_pred[0,4],y_pred[0,5],y_pred[0,6],y_pred[0,7],y_pred[0,8],y_pred[0,9],y_pred[0,10],y_pred[0,11],y_pred[0,12],y_pred[0,13],y_pred[0,14],y_pred[0,15],y_pred[0,16],y_pred[0,17],y_pred[0,18],y_pred[0,19],y_pred[0,20],y_pred[0,21],y_pred[0,22],y_pred[0,23],y_pred[0,24],y_pred[0,25], d_ribose_5p_reference,y_pred[0,27],y_pred[0,28],y_pred[0,29], ATP_reference ,y_pred[0,31],y_pred[0,32],y_pred[0,33],y_pred[0,34],y_pred[0,35],y_pred[0,36],y_pred[0,37],y_pred[0,38],y_pred[0,39],y_pred[0,40],y_pred[0,41],y_pred[0,42],y_pred[0,43],y_pred[0,44],y_pred[0,45],y_pred[0,46],y_pred[0,47],y_pred[0,48],y_pred[0,49],y_pred[0,50],y_pred[0,51],y_pred[0,52],y_pred[0,53],y_pred[0,54],y_pred[0,55],y_pred[0,56],y_pred[0,57],y_pred[0,58],y_pred[0,59],y_pred[0,60],y_pred[0,61],y_pred[0,62],y_pred[0,63],y_pred[0,64],y_pred[0,65],y_pred[0,66],y_pred[0,67],y_pred[0,68],y_pred[0,69],y_pred[0,70],y_pred[0,71],y_pred[0,72],y_pred[0,73],y_pred[0,74],y_pred[0,75],y_pred[0,76],y_pred[0,77],y_pred[0,78],y_pred[0,79],y_pred[0,80],y_pred[0,81],y_pred[0,82],y_pred[0,83],y_pred[0,84],y_pred[0,85],y_pred[0,86],y_pred[0,87],y_pred[0,88],y_pred[0,89],y_pred[0,90],y_pred[0,91],y_pred[0,92],y_pred[0,93],y_pred[0,94],y_pred[0,95],y_pred[0,96],y_pred[0,97],y_pred[0,98],y_pred[0,99],y_pred[0,100],y_pred[0,101],y_pred[0,102],y_pred[0,103],y_pred[0,104],y_pred[0,105],y_pred[0,106]])

    y_ref_transpose = np.zeros((1,R))
    for i in range(0,R):
        y_ref_transpose[0,i] = y_ref[i]
       
    y_ref_for_fixed_27_and_31 = y_ref_transpose
    y_actual = plant_output_prediction(U_Candidate,X_train, Y_train)

    return X_train, Y_train, U_Candidate, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref



def initial_processing_2(U_Candidate2):

    R = 107                     # Number of output of the Biological MIMO system

    tr = heli_data()                                          
    tr.Create_batch_train()                                   #Create the batch for training data
    #tr1 = heli_data()
    #tr1.Create_initial_u_candidate()                          #Create the batch for u_candidate

    X_train = tr.inputs                                         #Fetch train inputs
    Y_train = tr.outputs                                         #Fetch train outputs

    
    
    y_pred = SVM_model_predict_output(U_Candidate2)

    #### reference creation initially
    y_ref = np.array([y_pred[0,0], y_pred[0,1],y_pred[0,2],y_pred[0,3],y_pred[0,4],y_pred[0,5],y_pred[0,6],y_pred[0,7],y_pred[0,8],y_pred[0,9],y_pred[0,10],y_pred[0,11],y_pred[0,12],y_pred[0,13],y_pred[0,14],y_pred[0,15],y_pred[0,16],y_pred[0,17],y_pred[0,18],y_pred[0,19],y_pred[0,20],y_pred[0,21],y_pred[0,22],y_pred[0,23],y_pred[0,24],y_pred[0,25], y_pred[0,26] ,y_pred[0,27],y_pred[0,28],y_pred[0,29], y_pred[0,30] ,y_pred[0,31],y_pred[0,32],y_pred[0,33],y_pred[0,34],y_pred[0,35],y_pred[0,36],y_pred[0,37],y_pred[0,38],y_pred[0,39],y_pred[0,40],y_pred[0,41],y_pred[0,42],y_pred[0,43],y_pred[0,44],y_pred[0,45],y_pred[0,46],y_pred[0,47],y_pred[0,48],y_pred[0,49],y_pred[0,50],y_pred[0,51],y_pred[0,52],y_pred[0,53],y_pred[0,54],y_pred[0,55],y_pred[0,56],y_pred[0,57],y_pred[0,58],y_pred[0,59],y_pred[0,60],y_pred[0,61],y_pred[0,62],y_pred[0,63],y_pred[0,64],y_pred[0,65],y_pred[0,66],y_pred[0,67],y_pred[0,68],y_pred[0,69],y_pred[0,70],y_pred[0,71],y_pred[0,72],y_pred[0,73],y_pred[0,74],y_pred[0,75],y_pred[0,76],y_pred[0,77],y_pred[0,78],y_pred[0,79],y_pred[0,80],y_pred[0,81],y_pred[0,82],y_pred[0,83],y_pred[0,84],pyruvate_kinase_reference,y_pred[0,86],y_pred[0,87],y_pred[0,88],y_pred[0,89],y_pred[0,90],y_pred[0,91],y_pred[0,92],y_pred[0,93],y_pred[0,94],y_pred[0,95],y_pred[0,96],y_pred[0,97],y_pred[0,98],y_pred[0,99],y_pred[0,100],y_pred[0,101],y_pred[0,102],y_pred[0,103],y_pred[0,104],y_pred[0,105],y_pred[0,106]])

    y_ref_transpose = np.zeros((1,R))
    for i in range(0,R):
        y_ref_transpose[0,i] = y_ref[i]
       
    y_ref_for_fixed_27_and_31 = y_ref_transpose
    y_actual = plant_output_prediction(U_Candidate2,X_train, Y_train)

    return X_train, Y_train, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref

def initial_processing_3(U_Candidate3):

    R = 107                     # Number of output of the Biological MIMO system

    tr = heli_data()                                          
    tr.Create_batch_train()                                   #Create the batch for training data
    #tr1 = heli_data()
    #tr1.Create_initial_u_candidate()                          #Create the batch for u_candidate

    X_train = tr.inputs                                         #Fetch train inputs
    Y_train = tr.outputs                                         #Fetch train outputs

    
    
    y_pred = SVM_model_predict_output(U_Candidate3)

    #### reference creation initially
    y_ref = np.array([y_pred[0,0], y_pred[0,1],y_pred[0,2],y_pred[0,3],y_pred[0,4],y_pred[0,5],y_pred[0,6],y_pred[0,7],y_pred[0,8],y_pred[0,9],y_pred[0,10],y_pred[0,11],y_pred[0,12],y_pred[0,13],y_pred[0,14],y_pred[0,15],y_pred[0,16],y_pred[0,17],y_pred[0,18],y_pred[0,19],y_pred[0,20],y_pred[0,21],y_pred[0,22],y_pred[0,23],y_pred[0,24],y_pred[0,25], y_pred[0,26] ,y_pred[0,27],y_pred[0,28],y_pred[0,29], y_pred[0,30] ,y_pred[0,31],y_pred[0,32],y_pred[0,33],y_pred[0,34],y_pred[0,35],y_pred[0,36],y_pred[0,37],y_pred[0,38],y_pred[0,39],y_pred[0,40],y_pred[0,41],y_pred[0,42],y_pred[0,43],y_pred[0,44],y_pred[0,45],y_pred[0,46],y_pred[0,47],y_pred[0,48],y_pred[0,49],y_pred[0,50],y_pred[0,51],y_pred[0,52],y_pred[0,53],y_pred[0,54],y_pred[0,55],y_pred[0,56],y_pred[0,57],y_pred[0,58],y_pred[0,59],y_pred[0,60],y_pred[0,61],y_pred[0,62],y_pred[0,63],y_pred[0,64],y_pred[0,65],y_pred[0,66],y_pred[0,67],y_pred[0,68],y_pred[0,69],y_pred[0,70],y_pred[0,71],y_pred[0,72],y_pred[0,73],y_pred[0,74],y_pred[0,75],y_pred[0,76],y_pred[0,77],y_pred[0,78],y_pred[0,79],y_pred[0,80],y_pred[0,81],y_pred[0,82],y_pred[0,83],y_pred[0,84],y_pred[0,85],y_pred[0,86],y_pred[0,87],y_pred[0,88],y_pred[0,89],y_pred[0,90],y_pred[0,91],y_pred[0,92],y_pred[0,93],y_pred[0,94],y_pred[0,95],y_pred[0,96],y_pred[0,97],y_pred[0,98],y_pred[0,99],glucose_6_phosphate_dehydrogenase_reference,y_pred[0,101],y_pred[0,102],y_pred[0,103],y_pred[0,104],y_pred[0,105],y_pred[0,106]])

    y_ref_transpose = np.zeros((1,R))
    for i in range(0,R):
        y_ref_transpose[0,i] = y_ref[i]
       
    y_ref_for_fixed_27_and_31 = y_ref_transpose
    y_actual = plant_output_prediction(U_Candidate3,X_train, Y_train)

    return X_train, Y_train, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref


def initial_processing_4(U_Candidate4):

    R = 107                     # Number of output of the Biological MIMO system

    tr = heli_data()                                          
    tr.Create_batch_train()                                   #Create the batch for training data
    #tr1 = heli_data()
    #tr1.Create_initial_u_candidate()                          #Create the batch for u_candidate

    X_train = tr.inputs                                         #Fetch train inputs
    Y_train = tr.outputs                                         #Fetch train outputs

    
    
    y_pred = SVM_model_predict_output(U_Candidate4)

    #### reference creation initially
    y_ref = np.array([y_pred[0,0], y_pred[0,1],y_pred[0,2],y_pred[0,3],y_pred[0,4],y_pred[0,5],y_pred[0,6],y_pred[0,7],y_pred[0,8],y_pred[0,9],y_pred[0,10],y_pred[0,11],y_pred[0,12],y_pred[0,13],y_pred[0,14],y_pred[0,15],y_pred[0,16],y_pred[0,17],y_pred[0,18],y_pred[0,19],y_pred[0,20],y_pred[0,21],y_pred[0,22],y_pred[0,23],y_pred[0,24],y_pred[0,25], y_pred[0,26] ,y_pred[0,27],y_pred[0,28],y_pred[0,29], y_pred[0,30] ,y_pred[0,31],y_pred[0,32],y_pred[0,33],y_pred[0,34],y_pred[0,35],y_pred[0,36],y_pred[0,37],y_pred[0,38],y_pred[0,39],y_pred[0,40],y_pred[0,41],y_pred[0,42],y_pred[0,43],y_pred[0,44],y_pred[0,45],y_pred[0,46],y_pred[0,47],y_pred[0,48],y_pred[0,49],y_pred[0,50],y_pred[0,51],y_pred[0,52],y_pred[0,53],y_pred[0,54],y_pred[0,55],y_pred[0,56],y_pred[0,57],y_pred[0,58],y_pred[0,59],y_pred[0,60],y_pred[0,61],y_pred[0,62],y_pred[0,63],y_pred[0,64],y_pred[0,65],y_pred[0,66],y_pred[0,67],y_pred[0,68],y_pred[0,69],y_pred[0,70],y_pred[0,71],y_pred[0,72],y_pred[0,73],y_pred[0,74],y_pred[0,75],y_pred[0,76],y_pred[0,77],y_pred[0,78],y_pred[0,79],y_pred[0,80],y_pred[0,81],y_pred[0,82],y_pred[0,83],y_pred[0,84],y_pred[0,85],y_pred[0,86],y_pred[0,87],y_pred[0,88],y_pred[0,89],y_pred[0,90],y_pred[0,91],y_pred[0,92],y_pred[0,93],y_pred[0,94],y_pred[0,95],y_pred[0,96],y_pred[0,97],y_pred[0,98],y_pred[0,99],y_pred[0,100],y_pred[0,101],y_pred[0,102],y_pred[0,103],y_pred[0,104],Transketolase_reference,y_pred[0,106]])

    y_ref_transpose = np.zeros((1,R))
    for i in range(0,R):
        y_ref_transpose[0,i] = y_ref[i]
       
    y_ref_for_fixed_27_and_31 = y_ref_transpose
    y_actual = plant_output_prediction(U_Candidate4,X_train, Y_train)

    return X_train, Y_train, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref


def initial_processing_5(U_Candidate5):

    R = 107                     # Number of output of the Biological MIMO system

    tr = heli_data()                                          
    tr.Create_batch_train()                                   #Create the batch for training data
    #tr1 = heli_data()
    #tr1.Create_initial_u_candidate()                          #Create the batch for u_candidate

    X_train = tr.inputs                                         #Fetch train inputs
    Y_train = tr.outputs                                         #Fetch train outputs

    
    
    y_pred = SVM_model_predict_output(U_Candidate5)

    #### reference creation initially
    y_ref = np.array([y_pred[0,0], y_pred[0,1],y_pred[0,2],y_pred[0,3],y_pred[0,4],y_pred[0,5],y_pred[0,6],y_pred[0,7],y_pred[0,8],y_pred[0,9],y_pred[0,10],y_pred[0,11],y_pred[0,12],y_pred[0,13],y_pred[0,14],y_pred[0,15],y_pred[0,16],y_pred[0,17],y_pred[0,18],y_pred[0,19],y_pred[0,20],y_pred[0,21],y_pred[0,22],y_pred[0,23],y_pred[0,24],y_pred[0,25], y_pred[0,26] ,y_pred[0,27],y_pred[0,28],y_pred[0,29], y_pred[0,30] ,y_pred[0,31],y_pred[0,32],y_pred[0,33],y_pred[0,34],y_pred[0,35],y_pred[0,36],y_pred[0,37],y_pred[0,38],y_pred[0,39],y_pred[0,40],y_pred[0,41],y_pred[0,42],y_pred[0,43],y_pred[0,44],y_pred[0,45],y_pred[0,46],y_pred[0,47],y_pred[0,48],y_pred[0,49],y_pred[0,50],y_pred[0,51],y_pred[0,52],y_pred[0,53],y_pred[0,54],y_pred[0,55],y_pred[0,56],y_pred[0,57],y_pred[0,58],y_pred[0,59],y_pred[0,60],y_pred[0,61],y_pred[0,62],y_pred[0,63],y_pred[0,64],y_pred[0,65],y_pred[0,66],y_pred[0,67],y_pred[0,68],y_pred[0,69],y_pred[0,70],y_pred[0,71],y_pred[0,72],y_pred[0,73],y_pred[0,74],y_pred[0,75],y_pred[0,76],y_pred[0,77],y_pred[0,78],y_pred[0,79],y_pred[0,80],y_pred[0,81],y_pred[0,82],y_pred[0,83],y_pred[0,84],y_pred[0,85],y_pred[0,86],y_pred[0,87],y_pred[0,88],y_pred[0,89],y_pred[0,90],y_pred[0,91],y_pred[0,92],y_pred[0,93],y_pred[0,94],y_pred[0,95],y_pred[0,96],y_pred[0,97],y_pred[0,98],y_pred[0,99],y_pred[0,100],y_pred[0,101],y_pred[0,102],ribose_5_phosphate_isomerase_reference,y_pred[0,104],y_pred[0,105],y_pred[0,106]])

    y_ref_transpose = np.zeros((1,R))
    for i in range(0,R):
        y_ref_transpose[0,i] = y_ref[i]
       
    y_ref_for_fixed_27_and_31 = y_ref_transpose
    y_actual = plant_output_prediction(U_Candidate5,X_train, Y_train)

    return X_train, Y_train, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref



def initial_processing_6(U_Candidate6):

    R = 107                     # Number of output of the Biological MIMO system

    tr = heli_data()                                          
    tr.Create_batch_train()                                   #Create the batch for training data
    #tr1 = heli_data()
    #tr1.Create_initial_u_candidate()                          #Create the batch for u_candidate

    X_train = tr.inputs                                         #Fetch train inputs
    Y_train = tr.outputs                                         #Fetch train outputs

    
    
    y_pred = SVM_model_predict_output(U_Candidate6)

    #### reference creation initially
    y_ref = np.array([y_pred[0,0], y_pred[0,1],y_pred[0,2],y_pred[0,3],y_pred[0,4],y_pred[0,5],y_pred[0,6],y_pred[0,7],y_pred[0,8],y_pred[0,9],y_pred[0,10],y_pred[0,11],y_pred[0,12],y_pred[0,13],y_pred[0,14],y_pred[0,15],y_pred[0,16],y_pred[0,17],y_pred[0,18],y_pred[0,19],y_pred[0,20],y_pred[0,21],y_pred[0,22],y_pred[0,23],y_pred[0,24],y_pred[0,25], y_pred[0,26] ,y_pred[0,27],y_pred[0,28],y_pred[0,29], y_pred[0,30] ,y_pred[0,31],y_pred[0,32],y_pred[0,33],y_pred[0,34],y_pred[0,35],y_pred[0,36],y_pred[0,37],y_pred[0,38],y_pred[0,39],y_pred[0,40],y_pred[0,41],y_pred[0,42],y_pred[0,43],y_pred[0,44],y_pred[0,45],y_pred[0,46],y_pred[0,47],y_pred[0,48],y_pred[0,49],y_pred[0,50],y_pred[0,51],y_pred[0,52],y_pred[0,53],y_pred[0,54],y_pred[0,55],y_pred[0,56],y_pred[0,57],y_pred[0,58],y_pred[0,59],y_pred[0,60],y_pred[0,61],y_pred[0,62],y_pred[0,63],y_pred[0,64],y_pred[0,65],y_pred[0,66],y_pred[0,67],y_pred[0,68],y_pred[0,69],y_pred[0,70],y_pred[0,71],y_pred[0,72],y_pred[0,73],glucose_6_phosphate_isomerase_reference,y_pred[0,75],y_pred[0,76],y_pred[0,77],y_pred[0,78],y_pred[0,79],y_pred[0,80],y_pred[0,81],y_pred[0,82],y_pred[0,83],y_pred[0,84],y_pred[0,85],y_pred[0,86],y_pred[0,87],y_pred[0,88],y_pred[0,89],y_pred[0,90],y_pred[0,91],y_pred[0,92],y_pred[0,93],y_pred[0,94],y_pred[0,95],y_pred[0,96],y_pred[0,97],y_pred[0,98],y_pred[0,99],y_pred[0,100],y_pred[0,101],y_pred[0,102],y_pred[0,103],y_pred[0,104],y_pred[0,105],y_pred[0,106]])

    y_ref_transpose = np.zeros((1,R))
    for i in range(0,R):
        y_ref_transpose[0,i] = y_ref[i]
       
    y_ref_for_fixed_27_and_31 = y_ref_transpose
    y_actual = plant_output_prediction(U_Candidate6,X_train, Y_train)

    return X_train, Y_train, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref






def countdown(n):
    while n > 0:
        print ('Press 1 to terminate program within ',n,' second\n')
        tm.sleep(1)
        n = n - 1
    if n ==0:
        print('BLAST OFF!')


################  CONSTANT PARTS  #####################

##### METABOLIC REACTION SUBSTRATE CONSTANT km in nM #####

km0	 =	0.05564	
km1	 =	0.4479	
km2	 =	0.05966	
km3	 =	0.052076	
km4	 =	0.4479	
km5	 =	0.052373	
km6	 =	0.4289	
km7	 =	0.054248	
km8	 =	0.4654	
km9	 =	0.053261	
km10	 =	0.07412	
km11	 =	0.052933	
km12	 =	0.4415	
km13	 =	0.09692	
km14	 =	0.41931	
km15	 =	0.051036	
km16	 =	0.4729	
km17	 =	0.13405	
km18	 =	0.4658	
km19	 =	0.053377	
km20	 =	0.3951	
km21	 =	0.073413	
km22	 =	0.42947	
km23	 =	0.13745	
km24	 =	0.49851	
km25	 =	0.054972	
km26	 =	0.45491484	
km27	 =	0.1576	
km28	 =	0.074994	
km29	 =	0.4786	
km30	 =	0.052321	
km31	 =	0.052518	
km32	 =	0.42146	
km33	 =	0.053936	
km34	 =	0.063326	
km35	 =	0.050974	
km36	 =	0.4698	
km37	 =	0.054877	
km38	 =	0.051364	
km39	 =	0.051125	
km40	 =	0.093633	
km41	 =	0.4922	
km42	 =	0.062864	
km43	 =	0.052887	
km44	 =	0.084375	
km45	 =	0.092682	
km46	 =	0.052271	
km47	 =	0.073521	
km48	 =	0.063836	

##### METABOLIC REACTION ENZYME RATE CONATANT K in S**-1  #####

K1	 =	0.09		   # glut1
K2	 =	0.01		   # pgm_1
K3	 =	0.09		   # hk
K4	 =	0.01		   # g6Pase
K5	 =	0.0812		# pgi
K6	 =	0.09		   # pfk1
K7	 =	0.017		   # f16Bpase
K8	 =	0.09		   # pfk2
K9	 =	0.0766		# f26Bpase
K10	 =	0.0826		# ald
K11	 =	0.0899		# tpi
K12	 =	0.076		   # gcld3PD
K13	 =	0.0532		# pglc_kn
K14	 =	0.0755		# pglc_m
K15	 =	0.0727		# enl
K16	 =	0.05		   # pyrk
K17	 =	0.01		   # lacd
K18	 =	0 #0.0809	# pyrd                      ### Perturbation point for warburg effect ####### I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
K19	 =	0 #0.0643	# acyl_cos_synthase         ### Perturbation point for warburg effect ####### I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
K20	 =	0 #0.09		# fa_synthase               ### Perturbation point for warburg effect ####### I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
K21	 =	0 #0.0852	# pyr_crbxylase             ### Perturbation point for warburg effect ####### I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
K22	 =	0 #0.0296	# pep_crbxykinase1          ### Perturbation point for warburg effect ####### I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
K23	 =	0.0818		# cit_synthase
K24	 =	0.0844		# actnase
K25	 =	0.0881		# isocit_deh
K26	 =	0.0848		# KG_deh_cmp
K27	 =	0 #0.09		# succ_coa_synthase             ### Perturbation point for warburg effect ####### I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
K28	 =	0.0757		# succ_deh
K29	 =	0.0766		# frmase
K30	 =	0.0752		# mal_deh
K31	 =	0.0603		# g6P_deh
K32	 =	0.0191		# pglc6
K33	 =	0.0658		# phglc_deh
K34	 =	0.0792		# rbs_5Pis_A
K35	 =	0.069		   # rbls_5P_3ep
K36	 =	0.0892		# trkl
K37	 =	0.0722		# trsdl_1

##### SIGNALLING MOLECULE BINDING RATE CONSTANT KS in per second #####
### Randomly generated between 0.001 to 0.009 
KS1	 =	0.0013	
KS2	 =	0.0059	
KS3	 =	0.0039	
KS4	 =	0.0014	
KS5	 =	0.0049	
KS6	 =	0.0025	
KS7	 =	0.002	
KS8	 =	0.0016	
KS9	 =	0.0022	
KS10	 =	0.0025	
KS11	 =	0.0013	
KS12	 =	0.0061	
KS13	 =	0.0033	
KS14	 =	0.0053	
KS15	 =	0.0066	
KS16	 =	0.005	
KS17	 =	0.0053	
KS18	 =	0.0046	
KS19	 =	0.002	
KS20	 =	0.0049	
KS21	 =	0.0078	
KS22	 =	0.008	
KS23	 =	0.0032	
KS24	 =	0.0027	
KS25	 =	0.0055	
KS26	 =	0.0061	
KS27	 =	0.0043	
KS28	 =	0.0026	
KS29	 =	0.0086	
KS30	 =	0.0017	
KS31	 =	0.0018	
KS32	 =	0.0021	
KS33	 =	0.0023	
KS34	 =	0.006	
KS35	 =	0.0056	
KS36	 =	0.0014	
KS37	 =	0.0084	
KS38	 =	0.0068	
KS39	 =	0.0069	
KS43	 =	0.0015	
KS44	 =	0.0079	
KS45	 =	0.0085	
KS46	 =	0.0089	
KS47	 =	0.0079	
KS48	 =	0.0073	
KS49	 =	0.0051	
KS50	 =	0.0024	
KS52	 =	0.0042	
KS53	 =	0.0021	
KS54	 =	0.0012	

##### GENE BASAL PRODUCTION RATE CONSTANT kbg in per second #####

kbg1	 =	0.0004939		#	glut1
kbg2	 =	0.0004301		#	pgm_1
kbg3	 =	0.0004296		#	hk
kbg4	 =	0.0004333		#	g6Pase
kbg5	 =	0.0004467		#	pgi
kbg6	 =	0.0004648		#	pfk1
kbg7	 =	0.0004025		#	f16Bpase
kbg8	 =	0.0004842		#	pfk2
kbg9	 =	0.0004559		#	f26Bpase
kbg10	 =	0.0004854		#	ald
kbg11	 =	0.0004348		#	tpi
kbg12	 =	0.0004446		#	gcld3PD
kbg13	 =	0.0004054		#	pglc_kn
kbg14	 =	0.0004177		#	pglc_m
kbg15	 =	0.0004663		#	enl
kbg16	 =	0.0004331		#	pyrk
kbg17	 =	0.0004898		#	lacd
kbg18	 =	0.0004118		#	pyrd                          ### Perturbation point for warburg effect... but basal production should not be changed I think
kbg19	 =	0.0004988		#	acyl_cos_synthase             ### Perturbation point for warburg effect... but basal production should not be changed I think
kbg20	 =	0.000454		   #	fa_synthase                   ### Perturbation point for warburg effect... but basal production should not be changed I think
kbg21	 =	0.0004707		#	pyr_crbxylase                 ### Perturbation point for warburg effect... but basal production should not be changed I think
kbg22	 =	0.0004999		#	pep_crbxykinase1              ### Perturbation point for warburg effect... but basal production should not be changed I think
kbg23	 =	0.0004288		#	cit_synthase
kbg24	 =	0.0004415		#	actnase
kbg25	 =	0.0004465		#	isocit_deh
kbg26	 =	0.0004764		#	KG_deh_cmp
kbg27	 =	0.0004818		#	succ_coa_synthase             ### Perturbation point for warburg effect... but basal production should not be changed I think
kbg28	 =	0.00041		   #	succ_deh
kbg29	 =	0.0004178		#	frmase
kbg30	 =	0.000436		   #	mal_deh
kbg31	 =	0.0004057		#	g6P_deh
kbg32	 =	0.0004522		#	pglc6
kbg33	 =	0.0004336		#	phglc_deh
kbg34	 =	0.0004176		#	rbs_5Pis_A
kbg35	 =	0.0004209		#	rbls_5P_3ep
kbg36	 =	0.0004905		#	trkl
kbg37	 =	0.0004675		#	trsdl_1
        
        
##### GENE TRANSCRIPTION FACTORS BINDING RATE CONSTANT kg in per second #####
        
kg1	=	0.0007405	
kg2	=	0.0008736	
kg3	=	0.0006312	
kg4	=	0.0008237	
kg5	=	0.0008209	
kg6	=	0.0007686	
kg7	=	0.0006553	
kg8	=	0.0007792	
kg9	=	0.00069	
kg10	=	0.0006402	
kg11	=	0.0006638	
kg12	=	0.0008685	
kg13	=	0.0006214	
kg14	=	0.0006727	
kg15	=	0.0006161	
kg16	=	0.0007325	
kg17	=	0.000604	
kg18	=	0.0008692	   ### Perturbation point for warburg effect... but production should not be changed I think
kg19	=	0.000659	      ### Perturbation point for warburg effect... but production should not be changed I think                         
kg20	=	0.000628	      ### Perturbation point for warburg effect... but production should not be changed I think
kg21	=	0.0006922	   ### Perturbation point for warburg effect... but production should not be changed I think
kg22	=	0.0007368	   ### Perturbation point for warburg effect... but production should not be changed I think
kg23	=	0.0006305	
kg24	=	0.0008986	
kg25	=	0.0006996	
kg26	=	0.0006892	
kg27	=	0.0006186	   ### Perturbation point for warburg effect... but production should not be changed I think
kg28	=	0.0006895	
kg29	=	0.0006139	
kg30	=	0.0007516	
kg31	=	0.0008284	
kg32	=	0.0007893	
kg33	=	0.000627	
kg34	=	0.0006243	
kg35	=	0.0008332	
kg36	=	0.0008715	
kg37	=	0.0007601	
    
##### GENE DECAY RATE CONSTANT lg in per second #####
    
lg1=	0.0001109		#	glut1
lg2=	0.0001826		#	pgm_1
lg3=	0.0001338		#	hk
lg4=	0.0001294		#	g6Pase
lg5=	0.0001746		#	pgi
lg6=	0.000101		   #	pfk1
lg7=	0.0001048		#	f16Bpase
lg8=	0.0001668		#	pfk2
lg9=	0.0001603		#	f26Bpase
lg10=	0.0001526		#	ald
lg11=	0.000173		   #	tpi
lg12=	0.0001707		#	gcld3PD
lg13=	0.0001781		#	pglc_kn
lg14=	0.0001288		#	pglc_m
lg15=	0.0001693		#	enl
lg16=	0.0001557		#	pyrk
lg17=	0.0001397		#	lacd
lg18=	0.0001062		#	pyrd
lg19=	0.000178		   #	acyl_cos_synthase
lg20=	0.0001338		#	fa_synthase
lg21=	0.0001608		#	pyr_crbxylase
lg22=	0.0001741		#	pep_crbxykinase1
lg23=	0.0001105		#	cit_synthase
lg24=	0.0001128		#	actnase
lg25=	0.000155		   #	isocit_deh
lg26=	0.0001485		#	KG_deh_cmp
lg27=	0.000189		   #	succ_coa_synthase
lg28=	0.0001799		#	succ_deh
lg29=	0.0001734		#	frmase
lg30=	0.0001051		#	mal_deh
lg31=	0.0001073		#	g6P_deh
lg32=	0.0001089		#	pglc6
lg33=	0.0001798		#	phglc_deh
lg34=	0.0001943		#	rbs_5Pis_A
lg35=	0.0001684		#	rbls_5P_3ep
lg36=	0.0001132		#	trkl
lg37=	0.0001723		#	trsdl_1
    
##### Feedback parameters for Metabolic Network ######
#### F49, F50 is not included intentionally
F1	=	0.9            # 0.2435
F2	=	0.9            #0.9293
F3	=	0.9            #0.3500
F4	=	0.9            #0.1966
F5	=	0.9            #0.2511
F6	=	0.9            #0.6160
F7	=	0.9            #0.4733
F8	=	0.9            #0.3517
F9	=	0.9            #0.8308
F10	=	0.9            #0.5853
F11	=	0.9            #0.5497
F12	=	0.9            #0.9172
F13	=	0.9            #0.2858
F14_modified =	0.9   #0.7572
F15	=	0.9            #0.7537
F16	=	0.9            #0.3804
F17	=	0.9            #0.5678
F18	=	0.9            #0.0759
F19	=	0.9            #0.0540
F20	=	0.9            #0.5308
F21	=	0.9            #0.7792
F22	=	0.9            #0.9340
F23	=	0.9            #0.1299
F24	=	0.9            #0.5688
F25	=	0.9            #0.4694
F26	=	0.9            #0.0119
F27	=	0.9            #0.3371
F28	=	0.9            #0.1622
F29	=	0.9            #0.7943
F30	=	0.9            #0.3112
F31	=	0.9            #0.5285
F32	=	0.9            #0.1656
F33	=	0.9            #0.6020
F34	=	0.9            #0.2630
F35	=	0.9            #0.6541
F36	=	0.9            #0.6892
F37	=	0.9            #0.7482
F38	=	0.9            #0.4505
F39	=	0.9            #0.0838
F40	=	0.9            #0.2290
F41	=	0.9            #0.9133
F42	=	0.9            #0.1524
F43	=	0.9            #0.8258
F44	=	0.9            #0.5383
F45	=	0.9            #0.9961
F46	=	0.9            #0.0782
F47	=	0.9            #0.4427
F48	=	0.9            #0.1067
    
F51	=	0.9            #0.7749
F52	=	0.9            #0.8173
F53	=	0.9            #0.8687
F54	=	0.9            #0.0844
F55	=	0.9            #0.0716
    
##### Feedback parameters for Signalling Network ######
#### FS_12 is not included intentionally
F_S1	=	0.9         #0.3998	
F_S2	=	0.9         #0.2599	
F_S3	=	0.9         #0.8001	
F_S4	=	0.9         #0.4314	
F_S5	=	0.9         #0.9106	
F_S6	=	0.9         #0.1818	
F_S7	=	0.9         #0.2638	
F_S8	=	0.9         #0.1455	
F_S9	=	0.9         #0.1361	
F_S10	=	0.9         #0.8693	
F_S11	=	0.9         #0.5797	
F_S13	=	0.9         #0.145	
F_S14	=	0.9         #0.853	
F_S15	=	0.9         #0.6221	
F_S16	=	0.9         #0.351	
F_S17	=	0.9         #0.5132	
F_S18	=	0.9         #0.4018	
F_S19	=	0.9         #0.076	
F_S20	=	0.9         #0.2399	
F_S21	=	0.9         #0.1233	
F_S22	=	0.9         #0.1839	
F_S23	=	0.9         #0.24	
F_S24	=	0.9         #0.4173	
F_S25	=	0.9         #0.0497	
F_S26	=	0.9         #0.9027	
F_S27	=	0.9         #0.9448	
F_S28	=	0.9         #0.4909	
F_S29	=	0.9         #0.4893	
F_S30	=	0.9         #0.3377	
F_S31	=	0.9         #0.9001	
F_S32	=	0.9         #0.3692	
        
        
##### Feedback parameters for Gene Regulatory Network ######
        
FG1	= 0.9             #0.5289
FG2	= 0.9             #0.6944
FG3	= 0.9             #0.2124
FG4	= 0.9             #0.5433
FG5	= 0.9             #0.7025
FG6	= 0.9             #0.9564
FG7	= 0.9             #0.4445
FG8	= 0.9             #0.0854
FG9	= 0.9             #0.0573
FG10 = 0.9            #0.6295
FG11 = 0.9            #0.7962
FG12 = 0.9            #0.6912
FG13 = 0.9            #0.3453


#
# Main driver
# Generate a population and simulate GENERATIONS generations.
#
DNA_SIZE    = 134
POP_SIZE    = 20
GENERATIONS = 5

d_ribose_5p_reference = 0.75
ATP_reference = 0.75
pyruvate_kinase_reference = 0.1
glucose_6_phosphate_dehydrogenase_reference = 0.1
Transketolase_reference = 0.1
ribose_5_phosphate_isomerase_reference = 0.1
glucose_6_phosphate_isomerase_reference = 0.1


Q = 134                     # Number of input of the Biological MIMO system
R = 107                     # Number of output of the Biological MIMO system
num_of_past_input = 19
num_of_input_to_SVM = (Q+R)*num_of_past_input + Q
total_input = (Q+R)*num_of_past_input + Q
number_of_sample = 5000

      
h = 1
Nyq = 1   # prediction horizon
Nur = 1   # control horizon
nr = 19     # Number of past inputs
nz = 19     # Number of past inputs
mq = 19    # Number of past outputs
error = 0.01
error1 = 0.01
mz = 19   # Number of past outputs
max_iter = 500   
u_optimal = np.zeros((1,Q))
U_Candidate2 = np.zeros((1,total_input))
U_Candidate3 = np.zeros((1,total_input))
U_Candidate4 = np.zeros((1,total_input))
U_Candidate5 = np.zeros((1,total_input))
U_Candidate6 = np.zeros((1,total_input))


st = tm.time() 
print("Action\t\t\t\t\tTime")   #Save starting time
print("------\t\t\t\t\t----")
clfs, op_dim = biological_model_load()
print("Biological model loaded...\t\t\t{:.2f}s".format(tm.time()-st))

print("Initial preprocessing begins...\t\t\t{:.2f}s".format(tm.time()-st))


U_of_all_steps = np.zeros((max_iter,total_input))
Y_all_steps = np.zeros((max_iter,R))
Y_actual_all_steps = np.zeros((max_iter,R))
u_optimal_all_steps = np.zeros((max_iter,Q))
y_ref_all_steps = np.zeros((max_iter,R))

All_Population = np.zeros((POP_SIZE,DNA_SIZE,max_iter))
Population_last_to_check = np.zeros((POP_SIZE,DNA_SIZE))

iii = 0


X_train, Y_train, U_Candidate, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref = initial_processing()

U_of_all_steps[0,:] = U_Candidate
Y_all_steps[0,:] = y_pred
Y_actual_all_steps[0,:]=y_actual
u_optimal_all_steps[0,:]=u_optimal
y_ref_all_steps[0,:] = y_ref


kj = 0
for column in range(0,Q):
   u_optimal_all_steps[0,column] = U_Candidate[0,kj]
   kj = kj + 20
   
print("Optimization begins...\t\t\t{:.2f}s".format(tm.time()-st))
############### Optimization Starts Here ############################################
fitness_value_previous = 0

u_optimal_previous = 0
u_optimal_previous_prev = 0
u_optimal_previous_prev2 = 0


ureka = 0
iter = 0
count = 0
flag = 0
iteration = 0
bom_bhole =0
for iteration in range(max_iter):
#    if(iter == 1):
    fitness_value,u_optimal,y_pred, y_pred_new,Population_last_to_check,y_actual,iii,Population_last_to_check = genetic_controller(iter, st, u_optimal, U_Candidate, y_pred, y_actual, y_ref_for_fixed_27_and_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)       
    
    if(iter > 0 and (fitness_value_previous - fitness_value) > error1):
        print("\nBetter fit solution was not found..trying again...!!!\n")
#        iter = iter - 1
        u_optimal = u_optimal_previous_prev
        if(count > 10):
            flag = 1
        else:
            count = count + 1
            flag = 0
    else:
        fitness_value_previous = fitness_value  
        u_optimal_previous_prev2 = u_optimal_previous_prev
        u_optimal_previous_prev = u_optimal_previous
        u_optimal_previous = u_optimal
        count = 0
        flag = 1
    
    if(flag == 1):   
        for ww in range(0,Q):
            pos2 = ww*nz+ww
            U_Candidate[0,pos2+1:pos2+nz] = U_Candidate[0,pos2:pos2+nz-1]        
            U_Candidate[0,pos2] = u_optimal[ww]
        
        for yy in range(0,R):
            s2 = yy*mz
            s3 = Q*nz
            U_Candidate[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate[0,s3+Q+s2:s3+Q+s2+mz-2]
            U_Candidate[0,s3+Q+s2] = y_actual[0,yy]
      
        ## Find actual output from plant
        y_pred = SVM_model_predict_output(U_Candidate)
        y_actual = plant_output_prediction(U_Candidate,X_train, Y_train)    
        y_ref[:] =[y_pred[0,0], y_pred[0,1], y_pred[0,2], y_pred[0,3], y_pred[0,4], y_pred[0,5], y_pred[0,6], y_pred[0,7], y_pred[0,8], y_pred[0,9], y_pred[0,10], y_pred[0,11], y_pred[0,12], y_pred[0,13], y_pred[0,14], y_pred[0,15], y_pred[0,16], y_pred[0,17], y_pred[0,18], y_pred[0,19], y_pred[0,20], y_pred[0,21], y_pred[0,22], y_pred[0,23], y_pred[0,24], y_pred[0,25], y_ref_for_fixed_27_and_31[0,26],y_pred[0,27], y_pred[0,28], y_pred[0,29], y_ref_for_fixed_27_and_31[0,30], y_pred[0,31], y_pred[0,32], y_pred[0,33], y_pred[0,34], y_pred[0,35], y_pred[0,36], y_pred[0,37], y_pred[0,38], y_pred[0,39], y_pred[0,40], y_pred[0,41], y_pred[0,42], y_pred[0,43], y_pred[0,44], y_pred[0,45], y_pred[0,46], y_pred[0,47], y_pred[0,48], y_pred[0,49], y_pred[0,50], y_pred[0,51], y_pred[0,52], y_pred[0,53], y_pred[0,54], y_pred[0,55], y_pred[0,56], y_pred[0,57], y_pred[0,58], y_pred[0,59], y_pred[0,60], y_pred[0,61], y_pred[0,62], y_pred[0,63], y_pred[0,64], y_pred[0,65], y_pred[0,66], y_pred[0,67], y_pred[0,68], y_pred[0,69], y_pred[0,70], y_pred[0,71], y_pred[0,72], y_pred[0,73], y_pred[0,74], y_pred[0,75], y_pred[0,76], y_pred[0,77], y_pred[0,78], y_pred[0,79], y_pred[0,80], y_pred[0,81], y_pred[0,82], y_pred[0,83], y_pred[0,84], y_pred[0,85], y_pred[0,86], y_pred[0,87], y_pred[0,88], y_pred[0,89], y_pred[0,90], y_pred[0,91], y_pred[0,92], y_pred[0,93], y_pred[0,94], y_pred[0,95], y_pred[0,96], y_pred[0,97], y_pred[0,98], y_pred[0,99], y_pred[0,100], y_pred[0,101], y_pred[0,102], y_pred[0,103], y_pred[0,104], y_pred[0,105], y_pred[0,106]]


        U_of_all_steps[iter+1,:] = U_Candidate
        Y_all_steps[iter+1,:] = y_pred
        Y_actual_all_steps[iter+1,:]=y_actual
        u_optimal_all_steps[iter+1,:]=u_optimal
        y_ref_all_steps[iter+1,:] = y_ref
    
        print("Iteration number",(iter+1),"completed with time \t\t\t{:.2f}s\n\n".format(tm.time()-st))
        ## Find sum of absolute difference between actual output and reference output
        #sum_abs_error = 0
        #for i in range(Q):
            #sum_abs_error = sum_abs_error +  abs(y_actual_scaled[0,i] - y_ref_scaled[0,i])
        
        ## Check if the error is less than Threshold     
        #if(sum_abs_error < 0.01):
        if (count > 10):
            print("Interrupt optimizing due to stuck at lower fitness solution...!! \n")
            bom_bhole = 1
            break
        if ((np.absolute(y_ref[26] - y_actual[0,26]) < error) and (np.absolute(y_ref[30] - y_actual[0,30]) < error)):#and (np.absolute(y_ref_scaled[0,0] - y_pred_scaled[0,0]) < error) and (np.absolute(y_ref_scaled[0,1] - y_pred_scaled[0,1]) < error)):# and (np.absolute(y_ref_scaled[0,2] - y_pred_scaled[0,2]) < error) and (np.absolute(y_ref_scaled[0,3] - y_pred_scaled[0,3]) < error) and (np.absolute(y_ref_scaled[0,4] - y_pred_scaled[0,4]) < error)):  # condition ta modify kora holo
            ureka = 1
            print("BOM BHOLE...OPTIMAL SOLUTION FOUND !!!!")
            bom_bhole = 1
            break
        iter = iter + 1


print("Optimization1 ends...\t\t\t{:.2f}s".format(tm.time()-st))
print("Start writing result into excel file...\t\t\t{:.2f}s".format(tm.time()-st))

#write_data(U_of_all_steps,'U_of_all_steps.xlsx')
#print("U_of_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
write_data(Y_all_steps,'Y_all_steps.xlsx')
print("Y_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
write_data(Y_actual_all_steps,'Y_actual_all_steps.xlsx')
print("Y_actual_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
write_data(u_optimal_all_steps,'u_optimal_all_steps.xlsx')
print("u_optimal_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
write_data(y_ref_all_steps,'y_ref_all_steps.xlsx')
print("y_ref_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))


if(bom_bhole == 1):
    print('\nTrying to recover the cancer cell..by deactivating pyruvate kinase\n\n')  


    print("Initial preprocessing begins...\t\t\t{:.2f}s".format(tm.time()-st))



    


    U_of_all_steps2 = np.zeros((max_iter,total_input))
    Y_all_steps2 = np.zeros((max_iter,R))
    Y_actual_all_steps2 = np.zeros((max_iter,R))
    u_optimal_all_steps2 = np.zeros((max_iter,Q))
    y_ref_all_steps2 = np.zeros((max_iter,R))

    All_Population2 = np.zeros((POP_SIZE,DNA_SIZE,max_iter))
    Population_last_to_check2 = np.zeros((POP_SIZE,DNA_SIZE))

    iii = 0


    
    for column in range(0,total_input):
        U_Candidate2 [0,column] = U_of_all_steps[iter+1,column]     #fetch initial U_candidate

                                         
    print("Initial Candidate Input for drug target1 is extracted...\t\t\t{:.2f}s".format(tm.time()-st))
    X_train, Y_train, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref = initial_processing_2(U_Candidate2)

    U_of_all_steps2[0,:] = U_Candidate2
    Y_all_steps2[0,:] = y_pred
    Y_actual_all_steps2[0,:]=y_actual
    u_optimal_all_steps2[0,:]=u_optimal
    y_ref_all_steps2[0,:] = y_ref


    kj = 0
    for column in range(0,Q):
        u_optimal_all_steps2[0,column] = U_Candidate2[0,kj]
        kj = kj + 20

    print("Second Optimization begins...\t\t\t{:.2f}s".format(tm.time()-st))
    ############### Second Optimization Starts Here ############################################
    fitness_value_previous = 0

    u_optimal_previous = 0
    u_optimal_previous_prev = 0
    u_optimal_previous_prev2 = 0

    ureka = 0
    iter2 = 0
    count = 0
    flag = 0
    iteration = 0
    for iteration in range(max_iter):
    #    if(iter == 1):
        fitness_value,u_optimal,y_pred, y_pred_new,Population_last_to_check,y_actual,iii,Population_last_to_check = genetic_controller2(iter2, st, u_optimal, U_Candidate2, y_pred, y_actual, y_ref_for_fixed_27_and_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)       
    
        if(iter2 > 0 and (fitness_value_previous - fitness_value) > error1):
            print("\nBetter fit solution was not found..trying again...!!!\n")
    #        iter = iter - 1
            u_optimal = u_optimal_previous_prev
            if(count > 10):
                flag = 1
            else:
                count = count + 1
                flag = 0
        else:
            fitness_value_previous = fitness_value  
            u_optimal_previous_prev2 = u_optimal_previous_prev
            u_optimal_previous_prev = u_optimal_previous
            u_optimal_previous = u_optimal
            count = 0
            flag = 1
    
        if(flag == 1):   
            for ww in range(0,Q):
                pos2 = ww*nz+ww
                U_Candidate2[0,pos2+1:pos2+nz] = U_Candidate2[0,pos2:pos2+nz-1]        
                U_Candidate2[0,pos2] = u_optimal[ww]
        
            for yy in range(0,R):
                s2 = yy*mz
                s3 = Q*nz
                U_Candidate2[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate2[0,s3+Q+s2:s3+Q+s2+mz-2]
                U_Candidate2[0,s3+Q+s2] = y_actual[0,yy]
      
            ## Find actual output from plant
            y_pred = SVM_model_predict_output(U_Candidate2)
            y_actual = plant_output_prediction(U_Candidate2,X_train, Y_train)    
            y_ref[:] =[y_pred[0,0], y_pred[0,1], y_pred[0,2], y_pred[0,3], y_pred[0,4], y_pred[0,5], y_pred[0,6], y_pred[0,7], y_pred[0,8], y_pred[0,9], y_pred[0,10], y_pred[0,11], y_pred[0,12], y_pred[0,13], y_pred[0,14], y_pred[0,15], y_pred[0,16], y_pred[0,17], y_pred[0,18], y_pred[0,19], y_pred[0,20], y_pred[0,21], y_pred[0,22], y_pred[0,23], y_pred[0,24], y_pred[0,25], y_pred[0,26],y_pred[0,27], y_pred[0,28], y_pred[0,29], y_pred[0,30], y_pred[0,31], y_pred[0,32], y_pred[0,33], y_pred[0,34], y_pred[0,35], y_pred[0,36], y_pred[0,37], y_pred[0,38], y_pred[0,39], y_pred[0,40], y_pred[0,41], y_pred[0,42], y_pred[0,43], y_pred[0,44], y_pred[0,45], y_pred[0,46], y_pred[0,47], y_pred[0,48], y_pred[0,49], y_pred[0,50], y_pred[0,51], y_pred[0,52], y_pred[0,53], y_pred[0,54], y_pred[0,55], y_pred[0,56], y_pred[0,57], y_pred[0,58], y_pred[0,59], y_pred[0,60], y_pred[0,61], y_pred[0,62], y_pred[0,63], y_pred[0,64], y_pred[0,65], y_pred[0,66], y_pred[0,67], y_pred[0,68], y_pred[0,69], y_pred[0,70], y_pred[0,71], y_pred[0,72], y_pred[0,73], y_pred[0,74], y_pred[0,75], y_pred[0,76], y_pred[0,77], y_pred[0,78], y_pred[0,79], y_pred[0,80], y_pred[0,81], y_pred[0,82], y_pred[0,83], y_pred[0,84], y_ref_for_fixed_27_and_31[0,85], y_pred[0,86], y_pred[0,87], y_pred[0,88], y_pred[0,89], y_pred[0,90], y_pred[0,91], y_pred[0,92], y_pred[0,93], y_pred[0,94], y_pred[0,95], y_pred[0,96], y_pred[0,97], y_pred[0,98], y_pred[0,99], y_pred[0,100], y_pred[0,101], y_pred[0,102], y_pred[0,103], y_pred[0,104], y_pred[0,105], y_pred[0,106]]


            U_of_all_steps2[iter2+1,:] = U_Candidate2
            Y_all_steps2[iter2+1,:] = y_pred
            Y_actual_all_steps2[iter2+1,:]=y_actual
            u_optimal_all_steps2[iter2+1,:]=u_optimal
            y_ref_all_steps2[iter2+1,:] = y_ref
    
            print("Iteration number",(iter2+1),"completed with time \t\t\t{:.2f}s\n\n".format(tm.time()-st))
            ## Find sum of absolute difference between actual output and reference output
            #sum_abs_error = 0
            #for i in range(Q):
                #sum_abs_error = sum_abs_error +  abs(y_actual_scaled[0,i] - y_ref_scaled[0,i])
        
             ## Check if the error is less than Threshold     
            #if(sum_abs_error < 0.01):
            if (count > 10):
                print("Interrupt optimizing due to stuck at lower fitness solution...!! \n")
                break
            if ((np.absolute(y_ref[85] - y_actual[0,85]) < error) ):#and (np.absolute(y_ref_scaled[0,0] - y_pred_scaled[0,0]) < error) and (np.absolute(y_ref_scaled[0,1] - y_pred_scaled[0,1]) < error)):# and (np.absolute(y_ref_scaled[0,2] - y_pred_scaled[0,2]) < error) and (np.absolute(y_ref_scaled[0,3] - y_pred_scaled[0,3]) < error) and (np.absolute(y_ref_scaled[0,4] - y_pred_scaled[0,4]) < error)):  # condition ta modify kora holo
                ureka = 1
                print("BOM BHOLE...OPTIMAL SOLUTION FOUND !!!!")
                break
            iter2 = iter2 + 1
    
     
    print("Optimization2 ends...\t\t\t{:.2f}s".format(tm.time()-st))
    print("Start writing result into excel file...\t\t\t{:.2f}s".format(tm.time()-st))

     #write_data(U_of_all_steps,'U_of_all_steps.xlsx')
     #print("U_of_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(Y_all_steps2,'Y_all_steps_drug1.xlsx')
    print("Y_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(Y_actual_all_steps2,'Y_actual_all_steps_drug1.xlsx')
    print("Y_actual_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(u_optimal_all_steps2,'u_optimal_all_steps_drug1.xlsx')
    print("u_optimal_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(y_ref_all_steps2,'y_ref_all_steps_drug1.xlsx')
    print("y_ref_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))


    print('\nTrying to recover the cancer cell..by deactivating Glucose 6 phosphate dehydrogenase\n\n')  


    print("Initial preprocessing begins...\t\t\t{:.2f}s".format(tm.time()-st))


    U_of_all_steps3 = np.zeros((max_iter,total_input))
    Y_all_steps3 = np.zeros((max_iter,R))
    Y_actual_all_steps3 = np.zeros((max_iter,R))
    u_optimal_all_steps3 = np.zeros((max_iter,Q))
    y_ref_all_steps3 = np.zeros((max_iter,R))

    All_Population3 = np.zeros((POP_SIZE,DNA_SIZE,max_iter))
    Population_last_to_check3 = np.zeros((POP_SIZE,DNA_SIZE))

    iii = 0

    
    for column in range(0,total_input):
        U_Candidate3 [0,column] = U_of_all_steps[iter+1,column]     #fetch initial U_candidate    


    print("Initial Candidate Input for drug target2 is extracted...\t\t\t{:.2f}s".format(tm.time()-st))
    X_train, Y_train, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref = initial_processing_3(U_Candidate3)

    U_of_all_steps3[0,:] = U_Candidate3
    Y_all_steps3[0,:] = y_pred
    Y_actual_all_steps3[0,:]=y_actual
    u_optimal_all_steps3[0,:]=u_optimal
    y_ref_all_steps3[0,:] = y_ref


    kj = 0
    for column in range(0,Q):
        u_optimal_all_steps3[0,column] = U_Candidate3[0,kj]
        kj = kj + 20

    print("Third Optimization begins...\t\t\t{:.2f}s".format(tm.time()-st))
    ############### Third Optimization Starts Here ############################################
    fitness_value_previous = 0

    u_optimal_previous = 0
    u_optimal_previous_prev = 0
    u_optimal_previous_prev3 = 0

    ureka = 0
    iter3 = 0
    count = 0
    flag = 0
    iteration = 0
    for iteration in range(max_iter):
    #    if(iter == 1):
        fitness_value,u_optimal,y_pred, y_pred_new,Population_last_to_check,y_actual,iii,Population_last_to_check = genetic_controller3(iter3, st, u_optimal, U_Candidate3, y_pred, y_actual, y_ref_for_fixed_27_and_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)       
    
        if(iter3 > 0 and (fitness_value_previous - fitness_value) > error1):
            print("\nBetter fit solution was not found..trying again...!!!\n")
    #        iter = iter - 1
            u_optimal = u_optimal_previous_prev
            if(count > 10):
                flag = 1
            else:
                count = count + 1
                flag = 0
        else:
            fitness_value_previous = fitness_value  
            u_optimal_previous_prev3 = u_optimal_previous_prev
            u_optimal_previous_prev = u_optimal_previous
            u_optimal_previous = u_optimal
            count = 0
            flag = 1
    
        if(flag == 1):   
            for ww in range(0,Q):
                pos2 = ww*nz+ww
                U_Candidate3[0,pos2+1:pos2+nz] = U_Candidate3[0,pos2:pos2+nz-1]        
                U_Candidate3[0,pos2] = u_optimal[ww]
        
            for yy in range(0,R):
                s2 = yy*mz
                s3 = Q*nz
                U_Candidate3[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate3[0,s3+Q+s2:s3+Q+s2+mz-2]
                U_Candidate3[0,s3+Q+s2] = y_actual[0,yy]
      
            ## Find actual output from plant
            y_pred = SVM_model_predict_output(U_Candidate3)
            y_actual = plant_output_prediction(U_Candidate3,X_train, Y_train)    
            y_ref[:] =[y_pred[0,0], y_pred[0,1], y_pred[0,2], y_pred[0,3], y_pred[0,4], y_pred[0,5], y_pred[0,6], y_pred[0,7], y_pred[0,8], y_pred[0,9], y_pred[0,10], y_pred[0,11], y_pred[0,12], y_pred[0,13], y_pred[0,14], y_pred[0,15], y_pred[0,16], y_pred[0,17], y_pred[0,18], y_pred[0,19], y_pred[0,20], y_pred[0,21], y_pred[0,22], y_pred[0,23], y_pred[0,24], y_pred[0,25], y_pred[0,26],y_pred[0,27], y_pred[0,28], y_pred[0,29], y_pred[0,30], y_pred[0,31], y_pred[0,32], y_pred[0,33], y_pred[0,34], y_pred[0,35], y_pred[0,36], y_pred[0,37], y_pred[0,38], y_pred[0,39], y_pred[0,40], y_pred[0,41], y_pred[0,42], y_pred[0,43], y_pred[0,44], y_pred[0,45], y_pred[0,46], y_pred[0,47], y_pred[0,48], y_pred[0,49], y_pred[0,50], y_pred[0,51], y_pred[0,52], y_pred[0,53], y_pred[0,54], y_pred[0,55], y_pred[0,56], y_pred[0,57], y_pred[0,58], y_pred[0,59], y_pred[0,60], y_pred[0,61], y_pred[0,62], y_pred[0,63], y_pred[0,64], y_pred[0,65], y_pred[0,66], y_pred[0,67], y_pred[0,68], y_pred[0,69], y_pred[0,70], y_pred[0,71], y_pred[0,72], y_pred[0,73], y_pred[0,74], y_pred[0,75], y_pred[0,76], y_pred[0,77], y_pred[0,78], y_pred[0,79], y_pred[0,80], y_pred[0,81], y_pred[0,82], y_pred[0,83], y_pred[0,84], y_pred[0,85], y_pred[0,86], y_pred[0,87], y_pred[0,88], y_pred[0,89], y_pred[0,90], y_pred[0,91], y_pred[0,92], y_pred[0,93], y_pred[0,94], y_pred[0,95], y_pred[0,96], y_pred[0,97], y_pred[0,98], y_pred[0,99], y_ref_for_fixed_27_and_31[0,100], y_pred[0,101], y_pred[0,102], y_pred[0,103], y_pred[0,104], y_pred[0,105], y_pred[0,106]]


            U_of_all_steps3[iter3+1,:] = U_Candidate3
            Y_all_steps3[iter3+1,:] = y_pred
            Y_actual_all_steps3[iter3+1,:]=y_actual
            u_optimal_all_steps3[iter3+1,:]=u_optimal
            y_ref_all_steps3[iter3+1,:] = y_ref
    
            print("Iteration number",(iter3+1),"completed with time \t\t\t{:.2f}s\n\n".format(tm.time()-st))
            ## Find sum of absolute difference between actual output and reference output
            #sum_abs_error = 0
            #for i in range(Q):
                #sum_abs_error = sum_abs_error +  abs(y_actual_scaled[0,i] - y_ref_scaled[0,i])
        
             ## Check if the error is less than Threshold     
            #if(sum_abs_error < 0.01):
            if (count > 10):
                print("Interrupt optimizing due to stuck at lower fitness solution...!! \n")
                break
            if ((np.absolute(y_ref[100] - y_actual[0,100]) < error) ):#and (np.absolute(y_ref_scaled[0,0] - y_pred_scaled[0,0]) < error) and (np.absolute(y_ref_scaled[0,1] - y_pred_scaled[0,1]) < error)):# and (np.absolute(y_ref_scaled[0,2] - y_pred_scaled[0,2]) < error) and (np.absolute(y_ref_scaled[0,3] - y_pred_scaled[0,3]) < error) and (np.absolute(y_ref_scaled[0,4] - y_pred_scaled[0,4]) < error)):  # condition ta modify kora holo
                ureka = 1
                print("BOM BHOLE...OPTIMAL SOLUTION FOUND !!!!")
                break
            iter3 = iter3 + 1
    
     
    print("Optimization3 ends...\t\t\t{:.2f}s".format(tm.time()-st))
    print("Start writing result into excel file...\t\t\t{:.2f}s".format(tm.time()-st))

    #write_data(U_of_all_steps,'U_of_all_steps.xlsx')
    #print("U_of_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(Y_all_steps3,'Y_all_steps_drug2.xlsx')
    print("Y_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(Y_actual_all_steps3,'Y_actual_all_steps_drug2.xlsx')
    print("Y_actual_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(u_optimal_all_steps3,'u_optimal_all_steps_drug2.xlsx')
    print("u_optimal_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(y_ref_all_steps3,'y_ref_all_steps_drug2.xlsx')
    print("y_ref_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))

    


    print('\nTrying to recover the cancer cell..by deactivating Transketolase\n\n')  


    print("Initial preprocessing begins...\t\t\t{:.2f}s".format(tm.time()-st))


    U_of_all_steps4 = np.zeros((max_iter,total_input))
    Y_all_steps4 = np.zeros((max_iter,R))
    Y_actual_all_steps4 = np.zeros((max_iter,R))
    u_optimal_all_steps4 = np.zeros((max_iter,Q))
    y_ref_all_steps4 = np.zeros((max_iter,R))

    All_Population4 = np.zeros((POP_SIZE,DNA_SIZE,max_iter))
    Population_last_to_check4 = np.zeros((POP_SIZE,DNA_SIZE))

    iii = 0

   
    for column in range(0,total_input):
        U_Candidate4 [0,column] = U_of_all_steps[iter+1,column]     #fetch initial U_candidate  


    print("Initial Candidate Input for drug target3 is extracted...\t\t\t{:.2f}s".format(tm.time()-st))
    X_train, Y_train, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref = initial_processing_4(U_Candidate4)

    U_of_all_steps4[0,:] = U_Candidate4
    Y_all_steps4[0,:] = y_pred
    Y_actual_all_steps4[0,:]=y_actual
    u_optimal_all_steps4[0,:]=u_optimal
    y_ref_all_steps4[0,:] = y_ref


    kj = 0
    for column in range(0,Q):
        u_optimal_all_steps4[0,column] = U_Candidate4[0,kj]
        kj = kj + 20

    print("Fourth Optimization begins...\t\t\t{:.2f}s".format(tm.time()-st))
    ############### Fourth Optimization Starts Here ############################################
    fitness_value_previous = 0

    u_optimal_previous = 0
    u_optimal_previous_prev = 0
    u_optimal_previous_prev4 = 0

    ureka = 0
    iter4 = 0
    count = 0
    flag = 0
    iteration = 0
    for iteration in range(max_iter):
    #    if(iter == 1):
        fitness_value,u_optimal,y_pred, y_pred_new,Population_last_to_check,y_actual,iii,Population_last_to_check = genetic_controller4(iter4, st, u_optimal, U_Candidate4, y_pred, y_actual, y_ref_for_fixed_27_and_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)       
    
        if(iter4 > 0 and (fitness_value_previous - fitness_value) > error1):
            print("\nBetter fit solution was not found..trying again...!!!\n")
    #        iter = iter - 1
            u_optimal = u_optimal_previous_prev
            if(count > 10):
                flag = 1
            else:
                count = count + 1
                flag = 0
        else:
            fitness_value_previous = fitness_value  
            u_optimal_previous_prev4 = u_optimal_previous_prev
            u_optimal_previous_prev = u_optimal_previous
            u_optimal_previous = u_optimal
            count = 0
            flag = 1
    
        if(flag == 1):   
            for ww in range(0,Q):
                pos2 = ww*nz+ww
                U_Candidate4[0,pos2+1:pos2+nz] = U_Candidate4[0,pos2:pos2+nz-1]        
                U_Candidate4[0,pos2] = u_optimal[ww]
        
            for yy in range(0,R):
                s2 = yy*mz
                s3 = Q*nz
                U_Candidate4[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate4[0,s3+Q+s2:s3+Q+s2+mz-2]
                U_Candidate4[0,s3+Q+s2] = y_actual[0,yy]
      
            ## Find actual output from plant
            y_pred = SVM_model_predict_output(U_Candidate4)
            y_actual = plant_output_prediction(U_Candidate4,X_train, Y_train)    
            y_ref[:] =[y_pred[0,0], y_pred[0,1], y_pred[0,2], y_pred[0,3], y_pred[0,4], y_pred[0,5], y_pred[0,6], y_pred[0,7], y_pred[0,8], y_pred[0,9], y_pred[0,10], y_pred[0,11], y_pred[0,12], y_pred[0,13], y_pred[0,14], y_pred[0,15], y_pred[0,16], y_pred[0,17], y_pred[0,18], y_pred[0,19], y_pred[0,20], y_pred[0,21], y_pred[0,22], y_pred[0,23], y_pred[0,24], y_pred[0,25], y_pred[0,26],y_pred[0,27], y_pred[0,28], y_pred[0,29], y_pred[0,30], y_pred[0,31], y_pred[0,32], y_pred[0,33], y_pred[0,34], y_pred[0,35], y_pred[0,36], y_pred[0,37], y_pred[0,38], y_pred[0,39], y_pred[0,40], y_pred[0,41], y_pred[0,42], y_pred[0,43], y_pred[0,44], y_pred[0,45], y_pred[0,46], y_pred[0,47], y_pred[0,48], y_pred[0,49], y_pred[0,50], y_pred[0,51], y_pred[0,52], y_pred[0,53], y_pred[0,54], y_pred[0,55], y_pred[0,56], y_pred[0,57], y_pred[0,58], y_pred[0,59], y_pred[0,60], y_pred[0,61], y_pred[0,62], y_pred[0,63], y_pred[0,64], y_pred[0,65], y_pred[0,66], y_pred[0,67], y_pred[0,68], y_pred[0,69], y_pred[0,70], y_pred[0,71], y_pred[0,72], y_pred[0,73], y_pred[0,74], y_pred[0,75], y_pred[0,76], y_pred[0,77], y_pred[0,78], y_pred[0,79], y_pred[0,80], y_pred[0,81], y_pred[0,82], y_pred[0,83], y_pred[0,84], y_pred[0,85], y_pred[0,86], y_pred[0,87], y_pred[0,88], y_pred[0,89], y_pred[0,90], y_pred[0,91], y_pred[0,92], y_pred[0,93], y_pred[0,94], y_pred[0,95], y_pred[0,96], y_pred[0,97], y_pred[0,98], y_pred[0,99], y_pred[0,100], y_pred[0,101], y_pred[0,102], y_pred[0,103], y_pred[0,104], y_ref_for_fixed_27_and_31[0,105], y_pred[0,106]]


            U_of_all_steps4[iter4+1,:] = U_Candidate4
            Y_all_steps4[iter4+1,:] = y_pred
            Y_actual_all_steps4[iter4+1,:]=y_actual
            u_optimal_all_steps4[iter4+1,:]=u_optimal
            y_ref_all_steps4[iter4+1,:] = y_ref
    
            print("Iteration number",(iter4+1),"completed with time \t\t\t{:.2f}s\n\n".format(tm.time()-st))
            ## Find sum of absolute difference between actual output and reference output
            #sum_abs_error = 0
            #for i in range(Q):
                #sum_abs_error = sum_abs_error +  abs(y_actual_scaled[0,i] - y_ref_scaled[0,i])
        
             ## Check if the error is less than Threshold     
            #if(sum_abs_error < 0.01):
            if (count > 10):
                print("Interrupt optimizing due to stuck at lower fitness solution...!! \n")
                break
            if ((np.absolute(y_ref[105] - y_actual[0,105]) < error) ):#and (np.absolute(y_ref_scaled[0,0] - y_pred_scaled[0,0]) < error) and (np.absolute(y_ref_scaled[0,1] - y_pred_scaled[0,1]) < error)):# and (np.absolute(y_ref_scaled[0,2] - y_pred_scaled[0,2]) < error) and (np.absolute(y_ref_scaled[0,3] - y_pred_scaled[0,3]) < error) and (np.absolute(y_ref_scaled[0,4] - y_pred_scaled[0,4]) < error)):  # condition ta modify kora holo
                ureka = 1
                print("BOM BHOLE...OPTIMAL SOLUTION FOUND !!!!")
                break
            iter4 = iter4 + 1
    
     
    print("Optimization4 ends...\t\t\t{:.2f}s".format(tm.time()-st))
    print("Start writing result into excel file...\t\t\t{:.2f}s".format(tm.time()-st))

    #write_data(U_of_all_steps,'U_of_all_steps.xlsx')
    #print("U_of_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(Y_all_steps4,'Y_all_steps_drug3.xlsx')
    print("Y_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(Y_actual_all_steps4,'Y_actual_all_steps_drug3.xlsx')
    print("Y_actual_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(u_optimal_all_steps4,'u_optimal_all_steps_drug3.xlsx')
    print("u_optimal_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(y_ref_all_steps4,'y_ref_all_steps_drug3.xlsx')
    print("y_ref_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))




    print('\nTrying to recover the cancer cell..by deactivating ribose_5_phosphate_isomerase \n\n')  


    print("Initial preprocessing begins...\t\t\t{:.2f}s".format(tm.time()-st))


    U_of_all_steps5 = np.zeros((max_iter,total_input))
    Y_all_steps5 = np.zeros((max_iter,R))
    Y_actual_all_steps5 = np.zeros((max_iter,R))
    u_optimal_all_steps5 = np.zeros((max_iter,Q))
    y_ref_all_steps5 = np.zeros((max_iter,R))

    All_Population5 = np.zeros((POP_SIZE,DNA_SIZE,max_iter))
    Population_last_to_check5 = np.zeros((POP_SIZE,DNA_SIZE))

    iii = 0

   
    for column in range(0,total_input):
        U_Candidate5 [0,column] = U_of_all_steps[iter+1,column]     #fetch initial U_candidate  


    print("Initial Candidate Input for drug target4 is extracted...\t\t\t{:.2f}s".format(tm.time()-st))
    X_train, Y_train, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref = initial_processing_5(U_Candidate5)

    U_of_all_steps5[0,:] = U_Candidate5
    Y_all_steps5[0,:] = y_pred
    Y_actual_all_steps5[0,:]=y_actual
    u_optimal_all_steps5[0,:]=u_optimal
    y_ref_all_steps5[0,:] = y_ref


    kj = 0
    for column in range(0,Q):
        u_optimal_all_steps5[0,column] = U_Candidate5[0,kj]
        kj = kj + 20

    print("Fifth Optimization begins...\t\t\t{:.2f}s".format(tm.time()-st))
    ############### Fifth Optimization Starts Here ############################################
    fitness_value_previous = 0

    u_optimal_previous = 0
    u_optimal_previous_prev = 0
    u_optimal_previous_prev5 = 0

    ureka = 0
    iter5 = 0
    count = 0
    flag = 0
    iteration = 0
    for iteration in range(max_iter):
    #    if(iter == 1):
        fitness_value,u_optimal,y_pred, y_pred_new,Population_last_to_check,y_actual,iii,Population_last_to_check = genetic_controller5(iter5, st, u_optimal, U_Candidate5, y_pred, y_actual, y_ref_for_fixed_27_and_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)       
    
        if(iter5 > 0 and (fitness_value_previous - fitness_value) > error1):
            print("\nBetter fit solution was not found..trying again...!!!\n")
    #        iter = iter - 1
            u_optimal = u_optimal_previous_prev
            if(count > 10):
                flag = 1
            else:
                count = count + 1
                flag = 0
        else:
            fitness_value_previous = fitness_value  
            u_optimal_previous_prev5 = u_optimal_previous_prev
            u_optimal_previous_prev = u_optimal_previous
            u_optimal_previous = u_optimal
            count = 0
            flag = 1
    
        if(flag == 1):   
            for ww in range(0,Q):
                pos2 = ww*nz+ww
                U_Candidate5[0,pos2+1:pos2+nz] = U_Candidate5[0,pos2:pos2+nz-1]        
                U_Candidate5[0,pos2] = u_optimal[ww]
        
            for yy in range(0,R):
                s2 = yy*mz
                s3 = Q*nz
                U_Candidate5[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate5[0,s3+Q+s2:s3+Q+s2+mz-2]
                U_Candidate5[0,s3+Q+s2] = y_actual[0,yy]
      
            ## Find actual output from plant
            y_pred = SVM_model_predict_output(U_Candidate5)
            y_actual = plant_output_prediction(U_Candidate5,X_train, Y_train)    
            y_ref[:] =[y_pred[0,0], y_pred[0,1], y_pred[0,2], y_pred[0,3], y_pred[0,4], y_pred[0,5], y_pred[0,6], y_pred[0,7], y_pred[0,8], y_pred[0,9], y_pred[0,10], y_pred[0,11], y_pred[0,12], y_pred[0,13], y_pred[0,14], y_pred[0,15], y_pred[0,16], y_pred[0,17], y_pred[0,18], y_pred[0,19], y_pred[0,20], y_pred[0,21], y_pred[0,22], y_pred[0,23], y_pred[0,24], y_pred[0,25], y_pred[0,26],y_pred[0,27], y_pred[0,28], y_pred[0,29], y_pred[0,30], y_pred[0,31], y_pred[0,32], y_pred[0,33], y_pred[0,34], y_pred[0,35], y_pred[0,36], y_pred[0,37], y_pred[0,38], y_pred[0,39], y_pred[0,40], y_pred[0,41], y_pred[0,42], y_pred[0,43], y_pred[0,44], y_pred[0,45], y_pred[0,46], y_pred[0,47], y_pred[0,48], y_pred[0,49], y_pred[0,50], y_pred[0,51], y_pred[0,52], y_pred[0,53], y_pred[0,54], y_pred[0,55], y_pred[0,56], y_pred[0,57], y_pred[0,58], y_pred[0,59], y_pred[0,60], y_pred[0,61], y_pred[0,62], y_pred[0,63], y_pred[0,64], y_pred[0,65], y_pred[0,66], y_pred[0,67], y_pred[0,68], y_pred[0,69], y_pred[0,70], y_pred[0,71], y_pred[0,72], y_pred[0,73], y_pred[0,74], y_pred[0,75], y_pred[0,76], y_pred[0,77], y_pred[0,78], y_pred[0,79], y_pred[0,80], y_pred[0,81], y_pred[0,82], y_pred[0,83], y_pred[0,84], y_pred[0,85], y_pred[0,86], y_pred[0,87], y_pred[0,88], y_pred[0,89], y_pred[0,90], y_pred[0,91], y_pred[0,92], y_pred[0,93], y_pred[0,94], y_pred[0,95], y_pred[0,96], y_pred[0,97], y_pred[0,98], y_pred[0,99], y_pred[0,100], y_pred[0,101], y_pred[0,102], y_ref_for_fixed_27_and_31[0,103], y_pred[0,104], y_pred[0,105], y_pred[0,106]]


            U_of_all_steps5[iter5+1,:] = U_Candidate5
            Y_all_steps5[iter5+1,:] = y_pred
            Y_actual_all_steps5[iter5+1,:]=y_actual
            u_optimal_all_steps5[iter5+1,:]=u_optimal
            y_ref_all_steps5[iter5+1,:] = y_ref
    
            print("Iteration number",(iter5+1),"completed with time \t\t\t{:.2f}s\n\n".format(tm.time()-st))
            ## Find sum of absolute difference between actual output and reference output
            #sum_abs_error = 0
            #for i in range(Q):
                #sum_abs_error = sum_abs_error +  abs(y_actual_scaled[0,i] - y_ref_scaled[0,i])
        
             ## Check if the error is less than Threshold     
            #if(sum_abs_error < 0.01):
            if (count > 10):
                print("Interrupt optimizing due to stuck at lower fitness solution...!! \n")
                break
            if ((np.absolute(y_ref[103] - y_actual[0,103]) < error) ):#and (np.absolute(y_ref_scaled[0,0] - y_pred_scaled[0,0]) < error) and (np.absolute(y_ref_scaled[0,1] - y_pred_scaled[0,1]) < error)):# and (np.absolute(y_ref_scaled[0,2] - y_pred_scaled[0,2]) < error) and (np.absolute(y_ref_scaled[0,3] - y_pred_scaled[0,3]) < error) and (np.absolute(y_ref_scaled[0,4] - y_pred_scaled[0,4]) < error)):  # condition ta modify kora holo
                ureka = 1
                print("BOM BHOLE...OPTIMAL SOLUTION FOUND !!!!")
                break
            iter5 = iter5 + 1
    
     
    print("Optimization5 ends...\t\t\t{:.2f}s".format(tm.time()-st))
    print("Start writing result into excel file...\t\t\t{:.2f}s".format(tm.time()-st))

    #write_data(U_of_all_steps,'U_of_all_steps.xlsx')
    #print("U_of_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(Y_all_steps5,'Y_all_steps_drug4.xlsx')
    print("Y_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(Y_actual_all_steps5,'Y_actual_all_steps_drug4.xlsx')
    print("Y_actual_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(u_optimal_all_steps5,'u_optimal_all_steps_drug4.xlsx')
    print("u_optimal_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(y_ref_all_steps5,'y_ref_all_steps_drug4.xlsx')
    print("y_ref_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))




    print('\nTrying to recover the cancer cell..by deactivating phosphoglucoisomerase \n\n')  


    print("Initial preprocessing begins...\t\t\t{:.2f}s".format(tm.time()-st))


    U_of_all_steps6 = np.zeros((max_iter,total_input))
    Y_all_steps6 = np.zeros((max_iter,R))
    Y_actual_all_steps6 = np.zeros((max_iter,R))
    u_optimal_all_steps6 = np.zeros((max_iter,Q))
    y_ref_all_steps6 = np.zeros((max_iter,R))

    All_Population6 = np.zeros((POP_SIZE,DNA_SIZE,max_iter))
    Population_last_to_check6 = np.zeros((POP_SIZE,DNA_SIZE))

    iii = 0

   
    for column in range(0,total_input):
        U_Candidate6 [0,column] = U_of_all_steps[iter+1,column]     #fetch initial U_candidate  


    print("Initial Candidate Input for drug target5 is extracted...\t\t\t{:.2f}s".format(tm.time()-st))
    X_train, Y_train, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref = initial_processing_6(U_Candidate6)

    U_of_all_steps6[0,:] = U_Candidate6
    Y_all_steps6[0,:] = y_pred
    Y_actual_all_steps6[0,:]=y_actual
    u_optimal_all_steps6[0,:]=u_optimal
    y_ref_all_steps6[0,:] = y_ref


    kj = 0
    for column in range(0,Q):
        u_optimal_all_steps6[0,column] = U_Candidate6[0,kj]
        kj = kj + 20

    print("Sixth Optimization begins...\t\t\t{:.2f}s".format(tm.time()-st))
    ############### Sixth Optimization Starts Here ############################################
    fitness_value_previous = 0

    u_optimal_previous = 0
    u_optimal_previous_prev = 0
    u_optimal_previous_prev6 = 0

    ureka = 0
    iter6 = 0
    count = 0
    flag = 0
    iteration = 0
    for iteration in range(max_iter):
    #    if(iter == 1):
        fitness_value,u_optimal,y_pred, y_pred_new,Population_last_to_check,y_actual,iii,Population_last_to_check = genetic_controller6(iter6, st, u_optimal, U_Candidate6, y_pred, y_actual, y_ref_for_fixed_27_and_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)       
    
        if(iter6 > 0 and (fitness_value_previous - fitness_value) > error1):
            print("\nBetter fit solution was not found..trying again...!!!\n")
    #        iter = iter - 1
            u_optimal = u_optimal_previous_prev
            if(count > 10):
                flag = 1
            else:
                count = count + 1
                flag = 0
        else:
            fitness_value_previous = fitness_value  
            u_optimal_previous_prev6 = u_optimal_previous_prev
            u_optimal_previous_prev = u_optimal_previous
            u_optimal_previous = u_optimal
            count = 0
            flag = 1
    
        if(flag == 1):   
            for ww in range(0,Q):
                pos2 = ww*nz+ww
                U_Candidate6[0,pos2+1:pos2+nz] = U_Candidate6[0,pos2:pos2+nz-1]        
                U_Candidate6[0,pos2] = u_optimal[ww]
        
            for yy in range(0,R):
                s2 = yy*mz
                s3 = Q*nz
                U_Candidate6[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate6[0,s3+Q+s2:s3+Q+s2+mz-2]
                U_Candidate6[0,s3+Q+s2] = y_actual[0,yy]
      
            ## Find actual output from plant
            y_pred = SVM_model_predict_output(U_Candidate6)
            y_actual = plant_output_prediction(U_Candidate6,X_train, Y_train)    
            y_ref[:] =[y_pred[0,0], y_pred[0,1], y_pred[0,2], y_pred[0,3], y_pred[0,4], y_pred[0,5], y_pred[0,6], y_pred[0,7], y_pred[0,8], y_pred[0,9], y_pred[0,10], y_pred[0,11], y_pred[0,12], y_pred[0,13], y_pred[0,14], y_pred[0,15], y_pred[0,16], y_pred[0,17], y_pred[0,18], y_pred[0,19], y_pred[0,20], y_pred[0,21], y_pred[0,22], y_pred[0,23], y_pred[0,24], y_pred[0,25], y_pred[0,26],y_pred[0,27], y_pred[0,28], y_pred[0,29], y_pred[0,30], y_pred[0,31], y_pred[0,32], y_pred[0,33], y_pred[0,34], y_pred[0,35], y_pred[0,36], y_pred[0,37], y_pred[0,38], y_pred[0,39], y_pred[0,40], y_pred[0,41], y_pred[0,42], y_pred[0,43], y_pred[0,44], y_pred[0,45], y_pred[0,46], y_pred[0,47], y_pred[0,48], y_pred[0,49], y_pred[0,50], y_pred[0,51], y_pred[0,52], y_pred[0,53], y_pred[0,54], y_pred[0,55], y_pred[0,56], y_pred[0,57], y_pred[0,58], y_pred[0,59], y_pred[0,60], y_pred[0,61], y_pred[0,62], y_pred[0,63], y_pred[0,64], y_pred[0,65], y_pred[0,66], y_pred[0,67], y_pred[0,68], y_pred[0,69], y_pred[0,70], y_pred[0,71], y_pred[0,72], y_pred[0,73], y_ref_for_fixed_27_and_31[0,74], y_pred[0,75], y_pred[0,76], y_pred[0,77], y_pred[0,78], y_pred[0,79], y_pred[0,80], y_pred[0,81], y_pred[0,82], y_pred[0,83], y_pred[0,84], y_pred[0,85], y_pred[0,86], y_pred[0,87], y_pred[0,88], y_pred[0,89], y_pred[0,90], y_pred[0,91], y_pred[0,92], y_pred[0,93], y_pred[0,94], y_pred[0,95], y_pred[0,96], y_pred[0,97], y_pred[0,98], y_pred[0,99], y_pred[0,100], y_pred[0,101], y_pred[0,102], y_pred[0,103], y_pred[0,104], y_pred[0,105], y_pred[0,106]]


            U_of_all_steps6[iter6+1,:] = U_Candidate6
            Y_all_steps6[iter6+1,:] = y_pred
            Y_actual_all_steps6[iter6+1,:]=y_actual
            u_optimal_all_steps6[iter6+1,:]=u_optimal
            y_ref_all_steps6[iter6+1,:] = y_ref
    
            print("Iteration number",(iter6+1),"completed with time \t\t\t{:.2f}s\n\n".format(tm.time()-st))
            ## Find sum of absolute difference between actual output and reference output
            #sum_abs_error = 0
            #for i in range(Q):
                #sum_abs_error = sum_abs_error +  abs(y_actual_scaled[0,i] - y_ref_scaled[0,i])
        
             ## Check if the error is less than Threshold     
            #if(sum_abs_error < 0.01):
            if (count > 10):
                print("Interrupt optimizing due to stuck at lower fitness solution...!! \n")
                break
            if ((np.absolute(y_ref[74] - y_actual[0,74]) < error) ):#and (np.absolute(y_ref_scaled[0,0] - y_pred_scaled[0,0]) < error) and (np.absolute(y_ref_scaled[0,1] - y_pred_scaled[0,1]) < error)):# and (np.absolute(y_ref_scaled[0,2] - y_pred_scaled[0,2]) < error) and (np.absolute(y_ref_scaled[0,3] - y_pred_scaled[0,3]) < error) and (np.absolute(y_ref_scaled[0,4] - y_pred_scaled[0,4]) < error)):  # condition ta modify kora holo
                ureka = 1
                print("BOM BHOLE...OPTIMAL SOLUTION FOUND !!!!")
                break
            iter6 = iter6 + 1
    
     
    print("Optimization6 ends...\t\t\t{:.2f}s".format(tm.time()-st))
    print("Start writing result into excel file...\t\t\t{:.2f}s".format(tm.time()-st))

    #write_data(U_of_all_steps,'U_of_all_steps.xlsx')
    #print("U_of_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(Y_all_steps6,'Y_all_steps_drug5.xlsx')
    print("Y_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(Y_actual_all_steps6,'Y_actual_all_steps_drug5.xlsx')
    print("Y_actual_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(u_optimal_all_steps6,'u_optimal_all_steps_drug5.xlsx')
    print("u_optimal_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(y_ref_all_steps6,'y_ref_all_steps_drug5.xlsx')
    print("y_ref_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))




    print('\nTrying to recover the cancer cell..by activating pyruvate kinase\n\n')  


    print("Initial preprocessing begins...\t\t\t{:.2f}s".format(tm.time()-st))



    

    pyruvate_kinase_reference = 0.95
    U_of_all_steps7 = np.zeros((max_iter,total_input))
    Y_all_steps7 = np.zeros((max_iter,R))
    Y_actual_all_steps7 = np.zeros((max_iter,R))
    u_optimal_all_steps7 = np.zeros((max_iter,Q))
    y_ref_all_steps7 = np.zeros((max_iter,R))

    All_Population7 = np.zeros((POP_SIZE,DNA_SIZE,max_iter))
    Population_last_to_check7 = np.zeros((POP_SIZE,DNA_SIZE))

    iii = 0


    
    for column in range(0,total_input):
        U_Candidate2 [0,column] = U_of_all_steps[iter+1,column]     #fetch initial U_candidate

                                         
    print("Initial Candidate Input for drug target6 is extracted...\t\t\t{:.2f}s".format(tm.time()-st))
    X_train, Y_train, y_ref_for_fixed_27_and_31, y_pred, y_actual, y_ref = initial_processing_2(U_Candidate2)

    U_of_all_steps7[0,:] = U_Candidate2
    Y_all_steps7[0,:] = y_pred
    Y_actual_all_steps7[0,:]=y_actual
    u_optimal_all_steps7[0,:]=u_optimal
    y_ref_all_steps7[0,:] = y_ref


    kj = 0
    for column in range(0,Q):
        u_optimal_all_steps7[0,column] = U_Candidate2[0,kj]
        kj = kj + 20

    print("Seventh Optimization begins...\t\t\t{:.2f}s".format(tm.time()-st))
    ############### Seventh Optimization Starts Here ############################################
    fitness_value_previous = 0

    u_optimal_previous = 0
    u_optimal_previous_prev = 0
    u_optimal_previous_prev2 = 0

    ureka = 0
    iter2 = 0
    count = 0
    flag = 0
    iteration = 0
    for iteration in range(max_iter):
    #    if(iter == 1):
        fitness_value,u_optimal,y_pred, y_pred_new,Population_last_to_check,y_actual,iii,Population_last_to_check = genetic_controller2(iter2, st, u_optimal, U_Candidate2, y_pred, y_actual, y_ref_for_fixed_27_and_31,R,Q,Nyq,mz,nz,Nur,iii,X_train, Y_train)       
    
        if(iter2 > 0 and (fitness_value_previous - fitness_value) > error1):
            print("\nBetter fit solution was not found..trying again...!!!\n")
    #        iter = iter - 1
            u_optimal = u_optimal_previous_prev
            if(count > 10):
                flag = 1
            else:
                count = count + 1
                flag = 0
        else:
            fitness_value_previous = fitness_value  
            u_optimal_previous_prev2 = u_optimal_previous_prev
            u_optimal_previous_prev = u_optimal_previous
            u_optimal_previous = u_optimal
            count = 0
            flag = 1
    
        if(flag == 1):   
            for ww in range(0,Q):
                pos2 = ww*nz+ww
                U_Candidate2[0,pos2+1:pos2+nz] = U_Candidate2[0,pos2:pos2+nz-1]        
                U_Candidate2[0,pos2] = u_optimal[ww]
        
            for yy in range(0,R):
                s2 = yy*mz
                s3 = Q*nz
                U_Candidate2[0,s3+Q+s2+1:s3+Q+s2+mz-1] = U_Candidate2[0,s3+Q+s2:s3+Q+s2+mz-2]
                U_Candidate2[0,s3+Q+s2] = y_actual[0,yy]
      
            ## Find actual output from plant
            y_pred = SVM_model_predict_output(U_Candidate2)
            y_actual = plant_output_prediction(U_Candidate2,X_train, Y_train)    
            y_ref[:] =[y_pred[0,0], y_pred[0,1], y_pred[0,2], y_pred[0,3], y_pred[0,4], y_pred[0,5], y_pred[0,6], y_pred[0,7], y_pred[0,8], y_pred[0,9], y_pred[0,10], y_pred[0,11], y_pred[0,12], y_pred[0,13], y_pred[0,14], y_pred[0,15], y_pred[0,16], y_pred[0,17], y_pred[0,18], y_pred[0,19], y_pred[0,20], y_pred[0,21], y_pred[0,22], y_pred[0,23], y_pred[0,24], y_pred[0,25], y_pred[0,26],y_pred[0,27], y_pred[0,28], y_pred[0,29], y_pred[0,30], y_pred[0,31], y_pred[0,32], y_pred[0,33], y_pred[0,34], y_pred[0,35], y_pred[0,36], y_pred[0,37], y_pred[0,38], y_pred[0,39], y_pred[0,40], y_pred[0,41], y_pred[0,42], y_pred[0,43], y_pred[0,44], y_pred[0,45], y_pred[0,46], y_pred[0,47], y_pred[0,48], y_pred[0,49], y_pred[0,50], y_pred[0,51], y_pred[0,52], y_pred[0,53], y_pred[0,54], y_pred[0,55], y_pred[0,56], y_pred[0,57], y_pred[0,58], y_pred[0,59], y_pred[0,60], y_pred[0,61], y_pred[0,62], y_pred[0,63], y_pred[0,64], y_pred[0,65], y_pred[0,66], y_pred[0,67], y_pred[0,68], y_pred[0,69], y_pred[0,70], y_pred[0,71], y_pred[0,72], y_pred[0,73], y_pred[0,74], y_pred[0,75], y_pred[0,76], y_pred[0,77], y_pred[0,78], y_pred[0,79], y_pred[0,80], y_pred[0,81], y_pred[0,82], y_pred[0,83], y_pred[0,84], y_ref_for_fixed_27_and_31[0,85], y_pred[0,86], y_pred[0,87], y_pred[0,88], y_pred[0,89], y_pred[0,90], y_pred[0,91], y_pred[0,92], y_pred[0,93], y_pred[0,94], y_pred[0,95], y_pred[0,96], y_pred[0,97], y_pred[0,98], y_pred[0,99], y_pred[0,100], y_pred[0,101], y_pred[0,102], y_pred[0,103], y_pred[0,104], y_pred[0,105], y_pred[0,106]]


            U_of_all_steps7[iter2+1,:] = U_Candidate2
            Y_all_steps7[iter2+1,:] = y_pred
            Y_actual_all_steps7[iter2+1,:]=y_actual
            u_optimal_all_steps7[iter2+1,:]=u_optimal
            y_ref_all_steps7[iter2+1,:] = y_ref
    
            print("Iteration number",(iter2+1),"completed with time \t\t\t{:.2f}s\n\n".format(tm.time()-st))
            ## Find sum of absolute difference between actual output and reference output
            #sum_abs_error = 0
            #for i in range(Q):
                #sum_abs_error = sum_abs_error +  abs(y_actual_scaled[0,i] - y_ref_scaled[0,i])
        
             ## Check if the error is less than Threshold     
            #if(sum_abs_error < 0.01):
            if (count > 10):
                print("Interrupt optimizing due to stuck at lower fitness solution...!! \n")
                break
            if ((np.absolute(y_ref[85] - y_actual[0,85]) < error) ):#and (np.absolute(y_ref_scaled[0,0] - y_pred_scaled[0,0]) < error) and (np.absolute(y_ref_scaled[0,1] - y_pred_scaled[0,1]) < error)):# and (np.absolute(y_ref_scaled[0,2] - y_pred_scaled[0,2]) < error) and (np.absolute(y_ref_scaled[0,3] - y_pred_scaled[0,3]) < error) and (np.absolute(y_ref_scaled[0,4] - y_pred_scaled[0,4]) < error)):  # condition ta modify kora holo
                ureka = 1
                print("BOM BHOLE...OPTIMAL SOLUTION FOUND !!!!")
                break
            iter2 = iter2 + 1
    
     
    print("Optimization7 ends...\t\t\t{:.2f}s".format(tm.time()-st))
    print("Start writing result into excel file...\t\t\t{:.2f}s".format(tm.time()-st))

     #write_data(U_of_all_steps,'U_of_all_steps.xlsx')
     #print("U_of_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(Y_all_steps7,'Y_all_steps_drug6.xlsx')
    print("Y_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(Y_actual_all_steps7,'Y_actual_all_steps_drug6.xlsx')
    print("Y_actual_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(u_optimal_all_steps7,'u_optimal_all_steps_drug6.xlsx')
    print("u_optimal_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))
    write_data(y_ref_all_steps7,'y_ref_all_steps_drug6.xlsx')
    print("y_ref_all_steps has written into excel file...\t\t\t{:.2f}s".format(tm.time()-st))














############### Optimization ends Here ############################################

## plot result
#
#import matplotlib.pyplot as plt
#import numpy as np
#
#x_axis = np.zeros((iter+1,1))
#for i in range(iter+1):
#    x_axis[i,0] = i
#
#fig = plt.figure(1, figsize=(10, 10))
#plt.plot(x_axis[:,0], Y_actual_reverse_normalised_all_steps[:, 0], 'r-',x_axis[:,0], y_ref_reverse_normalised_all_steps[:, 0], 'b-',x_axis[:,0], Y_pred_reverse_normalised_all_steps[:, 0], 'k-')
#plt.axis([0, 50, 100, 150])
#plt.title('y_actual VS y_reference VS y_prediction')
#fig.savefig('Final_GA_performance_test_plot.jpg')

