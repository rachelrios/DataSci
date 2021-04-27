# -*- coding: utf-8 -*-
"""Homework6.ipynb
Rachel Rios

"""

#Problem 1
# Read in file
##Start of imports##
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn import metrics
import math
pd.set_option('mode.chained_assignment', None) #Supress warning since we will edit views

## End of imports ##
print("Problem 1: Reading File ")
df = pd.read_csv("https://raw.githubusercontent.com/rachelrios/rachelrios.github.io/master/datasets/seeds.csv") #I had to manually republish the file since there were data errors with tabs being off and numbers not being aligned in the matrix
print(df)

# START Problem 1 #
print("Problem 1: Got  R = 0: class L = 1 (negative) and L = 2 (positive)")
print("Problem 1: Extract Subset")
sub_df = df[df["Class"] == 1] #subset 1
sub_df = sub_df.append( df[df["Class"] == 2]) #subset 2
print(sub_df) #print info

'''
About: Stats for confusion matrixs
Params: Takes in actual vs predicted
'''
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    tpfn = 0
    tnfp = 0
    for i in range(len(y_hat)):
        if (y_actual[i]==y_hat[i]==1):
           TP += 1
        if (y_hat[i]==1 and y_actual[i]!=y_hat[i]):
           FP += 1
        if (y_actual[i]==y_hat[i]==0):
           TN += 1
        if (y_hat[i]==0 and y_actual[i]!=y_hat[i]):
           FN += 1
    accuracy = (TP+TN) / (TP+TN+FP+FN)
    tpfn += TP + FN
    tnfp += TN + FP
    try:
        TPR = TP/(TP + FN)
        TNR = TN/(TN + FP)
    except ZeroDivisionError:
        return 0
    print("TP, FP, TN, FN, Accuracy, TPR, TNR")
    return(TP, FP, TN, FN,accuracy,TPR,TNR)

## 1 Define Lambda function for + or - tags
'''
Stats for tags based on class
@params: Data frame
@returns: df with values
'''
def tags(row):
  if row["Class"] == 1: # Return -
    return 0
  if row["Class"] == 2: #Return +
    return 1
  return
print("*********************** START PROBLEM 1 ***********************")
##Problem 1: Kernel Classifier
print('Problem 1: Applying tags via lambda function')
sub_df["Tags"]= sub_df.apply (lambda row: tags(row), axis=1) #Apply lambda function on entire data frame
print("Problem 1: Applied Tags")
print(sub_df) #print new sub_df
# #We will now take the subset of class and tags
# class_df = sub_df[["Class","Tags"]] #Extract Class
#Use random 50/50 splits
X = sub_df[["A","P",	"C", "Length",	"Width" ,"Asymm. Coef","Length Kernel Groove"]].values #Get all values
scaler = StandardScaler() #Define basic Scaler
scaler.fit(X) #Fit X
X = scaler.transform (X) #Transform X
Y = sub_df[['Tags']].values #Get Values
kernels = ['linear','poly','rbf']
for i in kernels:
  info_stats = {"TN":0, "TP":0, "FP":0, "FN":0}
  print(str(i),"Info")
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.50) #Split test 50/50
  y2 = (Y_train.flat) #flatten array for manual confusion matrix
  svm_classifier = svm.SVC(kernel= i ) # Set kernel
  svm_classifier.fit(X_train,Y_train.ravel()) #fit data set
  X_train = scaler.transform(X_train) #transform data set
  X_test = scaler.transform(X_test) #transform testing set
  predicted = svm_classifier.predict(X_test) #calculate predicted
  logi = pd.DataFrame() #create dataframe
  print(perf_measure(Y_test,predicted))
print("----------------------- END PROBLEM 1 -----------------------")

# END Problem 1 #

# START Problem 2#
print("*********************** START PROBLEM 2 ***********************")
#KNN CLASSIFIER
print("Question 2: Pick up any classifier for supervised learning (e.g. kNN, logistic regression, Naive Bayesian, etc")
from sklearn.neighbors import KNeighborsClassifier
print("Chose KNN neighbors with 10 neighbors")
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.50) #Split test 50/50
scaler = StandardScaler() #scale data
scaler.fit(X_train) #train based on scaler
X_train = scaler.transform(X_train) #transform data set
X_test = scaler.transform(X_test) #transform testing set
classifier = KNeighborsClassifier(n_neighbors=10) #Define classifier by the prime array
classifier.fit(X_train, y_train.ravel()) #fit data set
pred = classifier.predict(X_test) #predict vaulues
logi = pd.DataFrame() #create dataframe
print(perf_measure(Y_test,pred))#Print Stats
print("----------------------- END PROBLEM 2 -----------------------")
# END Problem 2#

print("*********************** START PROBLEM 3 ***********************")

#Reset to all 3 class labels
print("Question 3: Take the original dataset with all 3 class labels.")
print("PROBLEM 3.1")
from sklearn.cluster import KMeans
print("3.1 1. for k = 1, 2, . . . , 8 use k-means clustering with random initialization and defaults.")
k  = [1,2,3,4,5,6,7,8]
#Use random 50/50 splits
X = df[["A","P",	"C", "Length",	"Width" ,"Asymm. Coef","Length Kernel Groove"]].values #Get all values
scaler = StandardScaler() #Define basic Scaler
scaler.fit(X) #Fit X
X = scaler.transform (X) #Transform X
Y = df[['Class']].values #Get class values for all three
distort = []
for i in k:
  info_stats = {"TN":0, "TP":0, "FP":0, "FN":0}
  print(str(i),"Info")
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.50) #Split test 50/50
  y2 = (Y_train.flat) #flatten array for manual confusion matrix
  kmeans_classifier = KMeans(n_clusters=i) #iterate through different K's
  predicted = kmeans_classifier.fit_predict(X) #predict y means via x test y means
  centroids = kmeans_classifier.cluster_centers_ #implement random clustering
  logi = pd.DataFrame() #create dataframe
  distort.append(kmeans_classifier.inertia_) #Calculate error
plt.title("Distortion VS K")
plt.xlabel("K")
plt.ylabel("Distortion")
plt.plot(k, distort)
plt.show()
#END 3.1
# PROBLEM 3.2 clustering dataset
print("PROBLEM 3.2 PLOTS RANDOM FEATURES")
df_x = df.drop("Class",axis=1)
X = df_x.sample(n=2,axis='columns') #Radom item
data_top = list(X.columns)
x1 = np.array(df[data_top[0]])
x2 = np.array(df[data_top[1]])
X = X.values
scaler = StandardScaler() #Define basic Scaler
scaler.fit(X) #Fit X
X = scaler.transform (X) #Transform X
Y = df[['Class']].values #Get class values for all three
# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'c']
markers = ['o', 'v', 's']
# KMeans algorithm
K = 3
kmeans_model = KMeans(n_clusters=K).fit(X)
predicted = kmeans_classifier.fit_predict(X) #predict y means via x test y means
print(kmeans_model.cluster_centers_)
centers = np.array(kmeans_model.cluster_centers_)
plt.plot()
plt.title('k means with 2 random features')
for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')
plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')
plt.show()
#End
print("Problem 3.3")
max = int(len(predicted)/3)
cluster1 = predicted[0:max]
cluster2 = predicted[max + 1 :max+max]
cluster3 = predicted[max+max+1:]
cards = {0.0:'Other',1.0:'Kama', 2.0:'Rosa', 3.0:'Canadian',4.0:'Other',5.0:'Other',6.0:'Other',7.0:'Other',8.0:'Other',9.0:'Other'}
print("Top class for cluster 1: ", np.median(cluster1), "with centers at: ",centers[0] )
print("Label =", cards[round(np.median(cluster1),0)] )
print("Top class for cluster 2: ", np.median(cluster2), "with centers at: ",centers[1] )
print("Label =", cards[round(np.median(cluster2),0)] )
print("Top class for cluster 3: ", np.median(cluster3), "with centers at: ",centers[2] )
print("Label =", cards[round(np.median(cluster3),0)] )

print("PROBLEM 3.4")
centers = np.array(kmeans_model.cluster_centers_)
print(centers[0])
center_d= [ (abs(centers[0][0]-centers[0][1])), abs(centers[1][0]-centers[1][1]), abs(centers[2][0]-centers[2][1]) ]
class_assigment = []
for z, l in enumerate(kmeans_model.labels_):
  em = []
  for k in range (0,3):

    for i in range(0,1):
      a = x1[z] - centers[k][i]
      b = x2[z] - centers[k][i+1]
      euc1 = a*a + b*b
      euc = math.sqrt(euc1)
      em.append(euc)
  class_assigment.append(em.index(min(em))+1)
print(class_assigment)
df['CA'] = class_assigment
accuracy = len(df[df['CA'] == df['Class']]) / len(df) * 100
print("Overall Accuracy is:")
print(accuracy)

# clustering dataset
print("Problem 3.5")
sub_df_x = sub_df.drop("Class",axis=1)
x1 = np.array(sub_df[data_top[0]])
x2 = np.array(sub_df[data_top[1]])
X = sub_df.sample(n=2,axis='columns')
data_top = list(X.columns)
X = X.values
scaler = StandardScaler() #Define basic Scaler
scaler.fit(X) #Fit X
X = scaler.transform (X) #Transform X
Y = sub_df[['Class']].values #Get class values for all three
# KMeans algorithm
K = 3
kmeans_model = KMeans(n_clusters=K).fit(X)
predicted = kmeans_classifier.fit_predict(X) #predict y means via x test y means
centers = np.array(kmeans_model.cluster_centers_)
center_d= [ (abs(centers[0][0]-centers[0][1])), abs(centers[1][0]-centers[1][1]), abs(centers[2][0]-centers[2][1]) ]
class_assigment = []
for z, l in enumerate(kmeans_model.labels_):
 em = []
 for k in range (0,3):
   for i in range(0,1):
     a = x1[z] - centers[k][i]
     b = x2[z] - centers[k][i+1]
     euc1 = a*a + b*b
     euc = math.sqrt(euc1)
     em.append(euc)
 class_assigment.append(em.index(min(em))+1)
sub_df['CA'] = class_assigment
accuracy = len(sub_df[sub_df['CA'] == sub_df['Class']]) / len(sub_df) * 100
print("(TP, FP, TN, FN)")
print(perf_measure(Y.flatten(), predicted))
# print("accuracy")
# print(accuracy)
print("----------------------- END PROBLEM 3 -----------------------")
