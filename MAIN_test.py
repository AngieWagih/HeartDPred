from sklearn.preprocessing import MinMaxScaler,StandardScaler,KBinsDiscretizer,Normalizer
import pandas as pd
import numpy as np
import pickle
import  sklearn

#load testing data
predict_data = pd.read_csv('FCIS-BIO-1/heart_test.csv')
predict_data=predict_data.drop(columns=['Index'])

# feature and label of train_data
x_predict =  predict_data.iloc[:,0:].values
x_predict = np.asarray(x_predict)
x_predict = x_predict.astype(int)  # convert exponential to integer
#print(x_predict.shape)

#preprocessing
#for KNN
scaler_KNN =StandardScaler()
x_predict_scaled = scaler_KNN.fit_transform(x_predict)

#for LR
LRscaler =MinMaxScaler()
x_predict_LR = scaler_KNN.fit_transform(x_predict)

#loading Models

#Load LRModel
lr_load=pickle.load(open("LRModel.sav",'rb'))
pred_res1=lr_load.predict(x_predict_LR)
print("Predicion of LR Model"+" "+str(pred_res1))


#Load NNModel
NN_load=pickle.load(open("NNModel.sav",'rb'))
pred_res2=NN_load.predict(x_predict_scaled)
print("Predicion of Neural Network Model"+" "+str(pred_res2))


#Load KNNModel
knn_load=pickle.load(open("KNNModel.sav",'rb'))
pred_res3=knn_load.predict(x_predict_scaled)
print("Predicion of KNN Model"+" "+str(pred_res3))

#load RF

RF_load=pickle.load(open("RFModel.sav",'rb'))
pred_res4=knn_load.predict(x_predict_scaled)
print("Predicion of RF Model"+" "+str(pred_res4))


#########################################################################################################################
#PCA
pcaknn=pickle.load(open("pcaknn.sav",'rb'))
pcaknnM=pickle.load(open("pcaknnModel.sav",'rb'))
x_predictpca=pcaknn.transform(x_predict_scaled)
pred_res5=pcaknnM.predict(x_predictpca)
print("Predicion of KNN Model after PCA"+" "+str(pred_res5))



pcarf=pickle.load(open("pcaRF.sav",'rb'))
pcarfM=pickle.load(open("RFPM.sav",'rb'))
x_predictpca1=pcarf.transform(x_predict_LR)
pred_res6=pcarfM.predict(x_predictpca1)
print("Predicion of RF Model after PCA"+" "+str(pred_res6))


pcalr=pickle.load(open("pcalr.sav",'rb'))
pcalrM=pickle.load(open("pcalrM.sav",'rb'))
x_predictpca1=pcalr.transform(x_predict_LR)
pred_res6=pcalrM.predict(x_predictpca1)
print("Predicion of LR Model after PCA"+" "+str(pred_res6))

