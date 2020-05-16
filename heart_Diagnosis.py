import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler,KBinsDiscretizer,Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import time
import pickle
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)

#read train data

train_data = pd.read_csv('FCIS-BIO-1/heart_train.csv')
train_data = train_data.drop(columns=['index'])

# analysis the data
analysis = train_data.describe()
print(analysis)

info=train_data.info()
print(info)

train_data.plot()
plt.title("Relationship Between Features")
plt.show()

sns.countplot(x="target", data=train_data)
plt.title("Show #patients that diagnosised with heart disease and do not")
plt.show()

# feature and label of train_data
x = train_data.drop(columns=['target'])
x_train = np.asarray(x)
# print(x_train)

y = train_data[['target']]
y = np.asarray(y)
y = np.ravel(y, order='C')
# print(y_train)
#spliting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4033, random_state=15)

#read test data
predict_data = pd.read_csv('FCIS-BIO-1/heart_test.csv')
predict_index = predict_data['Index']
predict_data=predict_data.drop(columns=['Index'])
#print(predict_data.head(5))

# feature and label of train_data
x_predict = predict_data.iloc[:,0:].values
x_predict = np.asarray(x_predict)
x_predict = x_predict.astype(int)  # convert exponential to integer

#pre_processing

#for KNN
scaler_KNN =StandardScaler()
x_train_KNN = scaler_KNN.fit_transform(x_train)
x_test_KNN = scaler_KNN.fit_transform(x_test)
x_predict_KNN = scaler_KNN.fit_transform(x_predict)

#for LR
LRscaler =MinMaxScaler()
x_scale=LRscaler.fit_transform(x)
x_train_LR = LRscaler.fit_transform(x_train)
x_test_LR = LRscaler.fit_transform(x_test)
x_predict_LR = LRscaler.fit_transform(x_predict)

#for NN

x_train_NN = scaler_KNN.fit_transform(x_train)
x_test_NN= scaler_KNN.fit_transform(x_test)
x_predict_NN = scaler_KNN.fit_transform(x_predict)

##############################################################################################################################

#build models

#build KNN Model

KNNtraintime1=time.time()
#print("time before train:"+" "+str(traintime1))

#KNN = KNeighborsClassifier()#79.59
#KNN = KNeighborsClassifier(algorithm='brute')#79.59
KNN = KNeighborsClassifier(n_neighbors=10,algorithm='ball_tree',weights='uniform')#84.69%
#KNN = KNeighborsClassifier(n_neighbors=10,algorithm='ball_tree',weights="distance")#83.67%

KNN.fit(x_train_KNN, y_train)
y_predict = KNN.predict(x_test_KNN)
KNNtraintime2=time.time()
#print("time after train:"+" "+str(traintime1))
KNN_trainT1=KNNtraintime2-KNNtraintime1

#get accuracy
KNNvald_accuracy = KNN.score(x_test_KNN, y_test)*100

KNN_Model=pickle.dump(KNN,open("KNNModel.sav",'wb'))
#confusion matrix
KNN_plot=plot_confusion_matrix(KNN,x_test_KNN,y_test)
plt.plot()
plt.title("KNN Confusion Matrix ")
plt.show()

#test and its time
KNNtestime1=time.time()
#print("time before test:"+" "+str(testime1))

y_pred_test = KNN.predict(x_predict_KNN)

KNNtestime2=time.time()
print("KNN predicted Results:")
print(y_pred_test)
#np.savetxt('KNN_sample2.csv',y_pred_test,delimiter=',',fmt='%s')

#print("time after test:"+" "+str(testime2))
KNN_testT1=KNNtestime2-KNNtestime1
print("Accuracy of KNN model: "+" "+str(KNNvald_accuracy))
print("KNN train time:"+" "+str(KNN_trainT1))
print("KNN testing time :"+" "+str(KNN_testT1))

#csv file
res_KNN=pd.DataFrame()
res_KNN['Index']=predict_index
res_KNN['target']=y_pred_test
res_KNN.to_csv('KNN_test.csv',index=False)

#######################################################################################################################
#build LR Model

LRtraintime1=time.time()

#print("time before train:"+" "+str(traintime1))
lgreg = LogisticRegression()
lgreg.fit(x_scale,y)
#lgreg.fit(x_train_LR, y_train)
y_predict = lgreg.predict(x_test_LR)

LRtraintime2=time.time()
#print("time after train:"+" "+str(traintime1))
LRtrainT=LRtraintime2-LRtraintime1

#get accuracy
LRvald_accuracy = lgreg.score(x_test_LR, y_test)*100

LR_Model=pickle.dump(lgreg,open("LRModel.sav",'wb'))
#confusion matrix
LR_plot=plot_confusion_matrix(lgreg,x_test_LR,y_test)
plt.plot()
plt.title("LR Confusion Matrix")
plt.show()


'''lgreg = LogisticRegression(penalty='l1',class_weight='balanced',solver='liblinear',random_state=randam_seed)
lgreg.fit(x_train_scale, y_train)
y_predict = lgreg.predict(x_test_scale)

vald_accuracy = lgreg.score(x_test_scale, y_test)
print(vald_accuracy)'''''#with accuracy=77%


''''lgreg = LogisticRegression(penalty='elasticnet',l1_ratio=1,class_weight='balanced',solver='saga',random_state=randam_seed)
lgreg.fit(x_train_scale, y_train)
y_predict = lgreg.predict(x_test_scale)

vald_accuracy = lgreg.score(x_test_scale, y_test)
print(vald_accuracy)'''''

#test and its time
LRtestime1=time.time()

#print("time before test:"+" "+str(testime1))

y_pred_test = lgreg.predict(x_predict_LR)

LRtestime2=time.time()
print("Logestic Regression predicted Results:")
print(y_pred_test)
#np.savetxt('lR_sample2.csv',y_pred_test,delimiter=',',fmt='%s')


#print("time after test:"+" "+str(testime2))

LRtestT=LRtestime2-LRtestime1
print("Accuracy of Logestic Regression model: "+" "+str(LRvald_accuracy))
print("Logistic Regression train time:"+" "+str(LRtrainT))
print("Logistic Regression testing time :"+" "+str(LRtestT))

#csv file
res_lr=pd.DataFrame()
res_lr['Index']=predict_index
res_lr['target']=y_pred_test
res_lr.to_csv('LR_test.csv',index=False)

#####################################################################################################################
#build neural network

NNtraintime1=time.time()
clf=MLPClassifier()#87%
#clf=MLPClassifier(hidden_layer_sizes=144)#86%
#clf=MLPClassifier(activation='logistic')#82%
#clf=MLPClassifier(activation='logistic',solver='sgd')#51.02%
#clf=MLPClassifier(activation="identity")#83.67
clf.fit(x_train_NN ,y_train)

NNtraintime2=time.time()
clf.predict(x_test_NN )
NNtrainT=NNtraintime2-NNtraintime1

NN_Model=pickle.dump(clf,open("NNModel.sav",'wb'))

NNvald_accuracy = clf.score(x_test_NN , y_test)*100
print("Accuracy of Neural Network  model: "+" "+str(NNvald_accuracy))


NN_plot=plot_confusion_matrix(clf,x_test_NN ,y_test)
plt.plot()
plt.title("NN Confusion Matrix")
plt.show()

NNtestime1=time.time()

pred=clf.predict(x_predict_NN )

NNtestime2=time.time()


NNtesT=NNtestime2-NNtestime1
print("Neural Network predicted Results:")
print(pred)
print("Neural Network train time:"+" "+str(NNtrainT))
print("Neural Networktesting time :"+" "+str(NNtesT))

#csv file
res_NN=pd.DataFrame()
res_NN['Index']=predict_index
res_NN['target']=pred
res_NN.to_csv('NN_test.csv',index=False)

#######################################################################################################################
RFtraintime1=time.time()
RFclf=RandomForestClassifier(max_features='sqrt',criterion='entropy',warm_start=True,random_state=1)
RFclf.fit(x_train_NN ,y_train)
RFtraintime2=time.time()

RFclf.predict(x_test_NN )
RFtraintime=RFtraintime2-RFtraintime1
print("RF Training Time:"+" "+str(RFtraintime))
RFvald_accuracy = RFclf.score(x_test_NN , y_test)*100
print("Accuracy of RF"+" "+str(RFvald_accuracy))

#load RF
RF_Model=pickle.dump(RFclf,open("RFModel.sav",'wb'))

RFtestime1=time.time()
pred=RFclf.predict(x_predict_NN )
print("RF Prediction"+" "+str(pred))
RFtestime2=time.time()

RFtestime=RFtestime2-RFtestime1
print("RF Testing Time:"+" "+str(RFtestime))
plot_confusion_matrix(clf,x_test_NN ,y_test)
plt.plot()
plt.title("Random Forest ClassifierConfution Matrix")
plt.show()

#csv file
res_rf=pd.DataFrame()
res_rf['Index']=predict_index
res_rf['target']=pred
res_rf.to_csv('rf_test.csv',index=False)


#########################################################################################################################
#using PCA for Dimentional Reduction

pca1=PCA(n_components=2,svd_solver='full')
x_train_pca1=pca1.fit_transform(x_train_KNN)
x_test_pca1=pca1.fit_transform(x_test_KNN)
x_predict_pca1=pca1.fit_transform(x_predict_KNN)



#KNN feat.PCA

pca=PCA(n_components=5,svd_solver='full')
x_train_pca=pca.fit_transform(x_train_KNN)
x_test_pca=pca.fit_transform(x_test_KNN)
x_predict_pca=pca.fit_transform(x_predict_KNN)


KNNtraintime_PCA1=time.time()
#print("time before train:"+" "+str(traintime_PCA1))
KNN_pca = KNeighborsClassifier()

KNN_pca.fit(x_train_pca, y_train)
y_predict = KNN_pca.predict(x_test_pca)
pcaknn_file=pickle.dump(KNN_pca,open("pcaknnModel.sav",'wb'))
pca1_file=pickle.dump(pca,open("pcaknn.sav",'wb'))


KNNtraintime_PCA2=time.time()
#print("time after train:"+" "+str(traintime_PCA2))

KNNPCATRT=KNNtraintime_PCA2-KNNtraintime_PCA1


#get accuracy
KNNvald_accuracy2 = KNN_pca.score(x_test_pca, y_test)*100


#confusion matrix
KNN_PCAPlot=plot_confusion_matrix(KNN_pca,x_test_pca,y_test)
plt.plot()
plt.title("KNN feat.PCA Confution Matrix")
plt.show()
KNNtestime_PCA1=time.time()

KNNy_pred_test = KNN_pca.predict(x_predict_pca)

KNNtestime_PCA2=time.time()
print("KNN FEAT.PCA predicted Results:")
print(KNNy_pred_test)
KNNPCATET=KNNtestime_PCA2-KNNtestime_PCA1

print("Accuracy of KNN_PCA model after PCA: "+" "+str(KNNvald_accuracy2))
print("KNN_PCA train time:"+" "+str(KNNPCATRT))
print("KNN_PCA train time:"+" "+str(KNNPCATET))

#csv file
res1=pd.DataFrame()
res1['Index']=predict_index
res1['target']=KNNy_pred_test
res1.to_csv('knnPCA_test.csv',index=False)

######################################################################################################################

#LR Feat. PCA

pca2=PCA(n_components=5,svd_solver='full')
x_train_pca=pca2.fit_transform(x_train_LR)
x_test_pca=pca2.fit_transform(x_test_LR)
x_predict_pca=pca2.fit_transform(x_predict_LR)



LRtraintime_PCA1=time.time()
#print("time before train:"+" "+str(traintime_PCA1))
lr_pca = LogisticRegression()
#lr_pca = LogisticRegression(penalty='l1',class_weight='balanced',solver='liblinear',random_state=randam_seed)
#lgreg_pca = LogisticRegression(penalty='elasticnet',l1_ratio=1,class_weight='balanced',solver='saga',random_state=randam_seed)
lr_pca.fit(x_train_pca, y_train)
y_predict = lr_pca.predict(x_test_pca)

pcalr_file=pickle.dump(lr_pca,open("pcalrM.sav",'wb'))

LRtraintime_PCA2=time.time()
#print("time after train:"+" "+

pca2_file=pickle.dump(pca2,open("pcalr.sav",'wb'))


LRPCATRT=LRtraintime_PCA2-LRtraintime_PCA1


#get accuracy
LRvald_accuracy2 = lr_pca.score(x_test_pca, y_test)*100


#confusion matrix
LR_PCAPlot=plot_confusion_matrix(lr_pca,x_test_pca,y_test)
plt.plot()
plt.title("LR feat.PCA Confution Matrix")
plt.show()

LRtestime_PCA1=time.time()

y_pred_test = lr_pca.predict(x_predict_pca)

LRtestime_PCA2=time.time()

print("KNN FEAT.PCA predicted Results:")
print(y_pred_test)
LRPCAtestT=LRtestime_PCA2-LRtestime_PCA1
print("Accuracy of Logestic Regression model after PCA: "+" "+str(LRvald_accuracy2))
print("Logistic Regression_PCA train time:"+" "+str(LRPCATRT))
print("Logistic Regression_PCA test time:"+" "+str(LRPCAtestT))

#csv
res2=pd.DataFrame()
res2['Index']=predict_index
res2['target']=y_pred_test
res2.to_csv('lrPCA_test.csv',index=False)


#######################################################################################################################
#using PCA
#pca=PCA(n_components=2,svd_solver='full')#77.5
#pca=PCA(n_components=5,svd_solver='full')#73.46
pca3=PCA(n_components=6,svd_solver='full')#77.5

x_train_pca=pca3.fit_transform(x_train_NN)
x_test_pca=pca3.fit_transform(x_test_NN)
x_predict_pca=pca3.fit_transform(x_predict_NN)



traintime_PCA1=time.time()

#print("time before train:"+" "+str(traintime_PCA1))
RF_pca = RandomForestClassifier()
RF_pca.fit(x_train_pca, y_train)

pcaRF_file=pickle.dump(RF_pca,open("RFPM.sav",'wb'))
y_predict = RF_pca.predict(x_test_pca)

pca3_file=pickle.dump(pca3,open("pcaRF.sav",'wb'))


traintime_PCA2=time.time()
#print("time after train:"+" "+str(traintime_PCA2))

RFTrainTPCA=traintime_PCA2-traintime_PCA1



#get accuracy
vald_accuracy2 =RF_pca.score(x_test_pca, y_test)


#confusion matrix
plot_confusion_matrix(RF_pca,x_test_pca,y_test)
plt.plot()
plt.title("RF feat.PCA Confution Matrix")
plt.show()

testime_PCA1=time.time()
y_pred_test = RF_pca.predict(x_predict_pca)
print(y_pred_test)
testime_PCA2=time.time()
RFtesTPCA=testime_PCA2-testime_PCA1
print("Accuracy of RF model after PCA: "+" "+str(vald_accuracy2))
print("RF_PCA train time:"+" "+str(RFTrainTPCA))
print("RF_PCA test time:"+" "+str(RFtesTPCA))
#csv
res3=pd.DataFrame()
res3['Index']=predict_index
res3['target']=y_pred_test
res3.to_csv('RFPCA_test.csv',index=False)

########################################################################################################################
#plot accuracy ,training time &testing time
#Accourding to Kaggle accuracies:
#befrore PCA
accuracies=[93.333,86.666,93.333,80]

models_name=["LR","Neural Network","KNN","RF"]
train=[LRtrainT, NNtrainT,KNN_trainT1,RFtraintime]
test=[LRtestT,NNtesT,KNN_testT1,RFtestime]

#graph accuracies vs models
plt.figure()
acc_models=plt.bar(models_name,accuracies)
plt.title(" accuracies vs models Before PCA")
plt.show()

#graph accuracies vs models
plt.figure()
train_models=plt.bar(models_name,train)
plt.title("Models Vs Training Time Before PCA")
plt.show()

#graph accuracies vs models
plt.figure()
test_models=plt.bar(models_name,test)
plt.title("Models Vs Testing Time Before PCA ")
plt.show()
#####################################################################################################
#after PCA
models_name=["LR_PCA","KNN_PCA","RF_PCA"]
accuracies=[86.66,93.33,90]
train=[LRPCATRT,KNNPCATRT,RFTrainTPCA]
test=[LRPCAtestT,KNNPCATET,RFtesTPCA]

#graph accuracies vs models
plt.figure()
acc_models=plt.bar(models_name,accuracies)
plt.title(" accuracies vs models After PCA")
plt.show()

#graph accuracies vs models
plt.figure()
train_models=plt.bar(models_name,train)
plt.title("Models Vs Training Time After PCA")
plt.show()

#graph accuracies vs models
plt.figure()
test_models=plt.bar(models_name,test)
plt.title("Models Vs Testing Time After PCA ")
plt.show()