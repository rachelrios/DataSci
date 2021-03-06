# -*- coding: utf-8 -*-
"""HW4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vYCbQwgEO3WiXjJaw_LDk6A_2J3IJd9L

Question 1:
1. load the data into Pandas dataframe. Extract two dataframes
with the above 4 features: df 0 for surviving patients (DEATH EVENT
= 0) and df 1 for deceased patients (DEATH EVENT = 1)
"""

### Define Libraries and other items ###
import pandas as pd
import seaborn as sns
import numpy as py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sn

#Prevent df warning when duplicated
pd.set_option('mode.chained_assignment', None) #Supress warning since we will edit views

### Problem 1 ###
## 1.1 Load the data frame and add a column for each class
#Call data frame from web
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv')
df = df.filter(['DEATH_EVENT','creatinine_phosphokinase','serum_creatinine','serum_sodium', 'platelets'])
df_0 = df[df['DEATH_EVENT'] == 0]
df_1 = df[df['DEATH_EVENT'] == 1]
print("Problem 1.1  Load the data into dataframe")
print("df 0 for surviving patients (DEATH EVENT = 0) ")
print(df_0)
print("df 1 for deceased patients (DEATH EVENT = 1) ")
print(df_1)

print("Problem 1.2  for each dataset, construct the visual representations of correponding correlation matrices M0 (from df 0) and M1 (from df 1) and save the plots into two separate files")

print("PROBLEM 1.2 M0 FROM DF_0 CORR PLOT")
df_0 = df_0.filter(['creatinine_phosphokinase','serum_creatinine','serum_sodium', 'platelets'])
corrMatrix = df_0.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

print("PROBLEM 1.2 M1 FROM DF_1 CORR PLOT")

df_1 = df_1.filter(['creatinine_phosphokinase','serum_creatinine','serum_sodium', 'platelets'])
corrMatrix = df_1.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()



"""2. Group 2: X: platelets, Y : serum sodium"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Data splits for DF_0
df_truncated0 = df_0.filter(['serum_sodium', 'platelets']) #Remove other columns
X0 = df_truncated0.iloc[:, :-1].values
Y0 = df_truncated0.iloc[:, 1].values
X0_min,X0_max = X0.min(), X0.max()
Y0_min,Y0_max = Y0.min(),Y0.max()
I0 = df_truncated0.index #Extract Indices
X0_train, X0_test, y0_train, y0_test = train_test_split(X0,Y0, test_size=0.50) #Split test 50/50
X0_test_min,X0_test_max = X0_test.min(), X0_test.max()
X0_train_min,X0_train_max = X0_train.min(), X0_train.max()

#Same equation for but M_1
df_truncated1 = df_1.filter(['serum_sodium', 'platelets']) #Remove other columns
X1 = df_truncated1.iloc[:, :-1].values
y1 = df_truncated1.iloc[:, 1].values
X1_min,X1_max = X1.min(), X1.max()
y1_min,y1_max = y1.min(),y1.max()
I0 = df_truncated1.index #Extract Indices
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1, test_size=0.50) #Split test 50/50
X1_test_min,X1_test_max = X1_test.min(), X1_test.max()
X1_train_min,X1_train_max = X1_train.min(), X1_train.max()

def m0_plots_and_anwers(degree):
  degree = degree
  #Flatten dataframes for M0
  X0_train_flat = np.array(X0_train).flatten()
  Y0_train_flat = np.array(y0_train).flatten()
  X0_test_flat = np.array(X0_test).flatten()
  Y0_test_flat = np.array(y0_test).flatten()
  Y0_test_flat = Y0_test_flat[:-1] #Not even split remove 1 item
  #Define Weights and  (a) fit the model on Xtrain
  print("2.a fit the model on Xtrain")
  print("np.polyfit(X1_train_flat,y1_train_flat, degree)")
  weights = np.polyfit(X0_train_flat,Y0_train_flat, degree)
  #(b) print the weights (a, b, . . .)
  print("2.b print the weights (a, b, . . .)")
  print("Weights: ", weights)
  #Set model
  model = np.poly1d(weights)
  #Calculate predicted #(c) compute predicted values using Xtest
  predicted = model(X0_test_flat)
  predicted = predicted[:-1] #Remove last element since it was not an even split

  print("2.c compute predicted values using Xtest")
  print("Predicted Values using Xtest: ", predicted)
  #(d) plot (if possible) predicted and actual values in Xtrain
  print("2.d plot (if possible) predicted and actual values in Xtrain")
  x_points = np.linspace(50,X0_test_max, 500)
  y_points = model(x_points)
  plt.scatter(X0_train_flat, y0_train, color = 'red')
  plt.scatter(X0_train_flat, predicted, color = 'black')
  plt.plot(x_points, y_points, color = 'blue')
  plt.title('Platelets vs Serum Sodium (Train set) Red = Actuals, Black = Predicted')
  plt.xlabel('Platelets')
  plt.ylabel('Serum Sodium')
  plt.show()
  #(e) compute (and print) the corresponding loss function
  rmse = np.sqrt(mean_squared_error(Y0_test_flat, predicted))
  r2 = r2_score(Y0_test_flat, predicted)
  print("2.e compute (and print) the corresponding loss function")
  print("SSE:", r2)
  return 0

def m1_plots_and_anwers(degree):
  degree = degree
  #Flatten dataframes for M1
  X1_train_flat = np.array(X1_train).flatten()
  y1_train_flat = np.array(y1_train).flatten()
  X1_test_flat = np.array(X1_test).flatten()
  y1_test_flat = np.array(y1_test).flatten()
  #Define Weights and  (a) fit the model on Xtrain
  print("2.a fit the model on Xtrain")
  print("np.polyfit(X1_train_flat,y1_train_flat, degree)")
  weights = np.polyfit(X1_train_flat,y1_train_flat, degree)
  #(b) print the weights (a, b, . . .)
  print("2.b print the weights (a, b, . . .)")
  print("Weights: ", weights)
  #Set model
  model = np.poly1d(weights)
  #Calculate predicted #(c) compute predicted values using Xtest
  predicted = model(X1_test_flat)
  print("2.c compute predicted values using Xtest")
  print("Predicted Values using Xtest: ", predicted)
  #(d) plot (if possible) predicted and actual values in Xtrain
  print("2.d plot (if possible) predicted and actual values in Xtrain")
  x_points = np.linspace(50,X1_test_max, 500)
  y_points = model(x_points)
  plt.scatter(X1_train_flat, y1_train, color = 'red')
  plt.scatter(X1_train_flat, predicted, color = 'black')
  plt.plot(x_points, y_points, color = 'blue')
  plt.title('Platelets vs Serum Sodium (Train set) Red = Actuals, Black = Predicted')
  plt.xlabel('Platelets')
  plt.ylabel('Serum Sodium')
  plt.show()
  #(e) compute (and print) the corresponding loss function
  rmse = np.sqrt(mean_squared_error(y1_test_flat, predicted))
  r2 = r2_score(y1_test_flat, predicted)
  print("2.e compute (and print) the corresponding loss function")
  print("SSE:", r2)
  return

def m0_plots_and_anwers_log1(degree):
  degree = degree
  #Flatten dataframes for M1
  X0_train_flat = np.array(X0_train).flatten()
  Y0_train_flat = np.array(y0_train).flatten()
  X0_test_flat = np.array(X0_test).flatten()
  Y0_test_flat = np.array(y0_test).flatten()
  Y0_test_flat = Y0_test_flat[:-1] #Not even split remove 1 item

  #Define Weights and  (a) fit the model on Xtrain
  print("2.a fit the model on Xtrain")
  print("np.polyfit(np.log(X0_train_flat),Y0_train_flat, degree")
  weights = np.polyfit(np.log(X0_train_flat),Y0_train_flat, degree)
  #(b) print the weights (a, b, . . .)
  print("2.b print the weights (a, b, . . .)")
  print("Weights: ", weights)
  #Set model
  model = np.poly1d(weights)
  #Calculate predicted #(c) compute predicted values using Xtest
  predicted = model(np.log(X0_test_flat))
  predicted = predicted[:-1] #Remove last element since it was not an even split
  print("2.c compute predicted values using Xtest")
  print("Predicted Values using Xtest: ", predicted)
  #(d) plot (if possible) predicted and actual values in Xtrain
  print("2.d plot (if possible) predicted and actual values in Xtrain")
  x_points = np.linspace(1,X0_test_max, 500)
  y = weights[0] + np.log(x_points) + weights[1]
  plt.scatter(np.log(X0_train_flat), Y0_train_flat, color = 'red')
  plt.scatter(np.log(X0_train_flat), predicted, color = 'black')
 # plt.plot(np.log(x_points), y, color = 'blue')
  plt.title('Platelets vs Serum Sodium (Train set) Red = Actuals, Black = Predicted')
  plt.xlabel('Platelets')
  plt.ylabel('Serum Sodium')
  plt.xlim(4.6,5)
  plt.show()
  #(e) compute (and print) the corresponding loss function
  rmse = np.sqrt(mean_squared_error(Y0_test_flat, predicted))
  r2 = r2_score(Y0_test_flat, predicted)
  print("2.e compute (and print) the corresponding loss function")
  print("SSE:", r2)
  return

def m1_plots_and_anwers_log1(degree):
  degree = degree
  #Flatten dataframes for M1
  X1_train_flat = np.array(X1_train).flatten()
  y1_train_flat = np.array(y1_train).flatten()
  X1_test_flat = np.array(X1_test).flatten()
  y1_test_flat = np.array(y1_test).flatten()
  #Define Weights and  (a) fit the model on Xtrain
  print("2.a fit the model on Xtrain")
  print("np.polyfit(np.log(X1_train_flat),y1_train_flat, degree")
  weights = np.polyfit(np.log(X1_train_flat),y1_train_flat, degree)
  #(b) print the weights (a, b, . . .)
  print("2.b print the weights (a, b, . . .)")
  print("Weights: ", weights)
  #Set model
  model = np.poly1d(weights)
  #Calculate predicted #(c) compute predicted values using Xtest
  predicted = model(np.log(X1_test_flat))
  print("2.c compute predicted values using Xtest")
  print("Predicted Values using Xtest: ", predicted)
  #(d) plot (if possible) predicted and actual values in Xtrain
  print("2.d plot (if possible) predicted and actual values in Xtrain")
  x_points = np.linspace(1,X1_test_max, 500)
  y = weights[0] + np.log(x_points) + weights[1]
  plt.scatter(np.log(X1_train_flat), y1_train_flat, color = 'red')
  plt.scatter(np.log(X1_train_flat), predicted, color = 'black')
 # plt.plot(np.log(x_points), y, color = 'blue')
  plt.title('Platelets vs Serum Sodium (Train set) Red = Actuals, Black = Predicted')
  plt.xlabel('Platelets')
  plt.ylabel('Serum Sodium')
  plt.xlim(4.6,5)
  plt.show()
  #(e) compute (and print) the corresponding loss function
  rmse = np.sqrt(mean_squared_error(y1_test, predicted))
  r2 = r2_score(y1_test, predicted)
  print("2.e compute (and print) the corresponding loss function")
  print("SSE:", r2)
  return

def m0_plots_and_anwers_log2(degree):
  degree = degree
  #Flatten dataframes for M1
  X0_train_flat = np.array(X0_train).flatten()
  Y0_train_flat = np.array(y0_train).flatten()
  X0_test_flat = np.array(X0_test).flatten()
  Y0_test_flat = np.array(y0_test).flatten()
  Y0_test_flat = Y0_test_flat[:-1] #Not even split remove 1 item

  #Define Weights and  (a) fit the model on Xtrain
  print("2.a fit the model on Xtrain")
  weights = np.polyfit(np.log(X0_train_flat),np.log(Y0_train_flat), degree)
  #(b) print the weights (a, b, . . .)
  print("2.b print the weights (a, b, . . .)")
  print("Weights: ", weights)
  #Set model
  model = np.poly1d(weights)
  #Calculate predicted #(c) compute predicted values using Xtest
  predicted = model(np.log(X0_test_flat))
  predicted = predicted[:-1] #Remove last element since it was not an even split
  print("2.c compute predicted values using Xtest")
  print("Predicted Values using Xtest: ", predicted)
  #(d) plot (if possible) predicted and actual values in Xtrain
  print("2.d plot (if possible) predicted and actual values in Xtrain")
  x_points = np.linspace(1,X0_test_max, 500)
  y = weights[0] + np.log(x_points) + weights[1]
  plt.scatter(np.log(X0_train_flat), np.log(Y0_train_flat), color = 'red')
  plt.scatter(np.log(X0_train_flat), predicted, color = 'black')
 # plt.plot(np.log(x_points), y, color = 'blue')
  plt.title('Platelets vs Serum Sodium (Train set) Red = Actuals, Black = Predicted')
  plt.xlabel('Platelets')
  plt.ylabel('Serum Sodium')
  plt.xlim(4.6,5)
  plt.show()
  #(e) compute (and print) the corresponding loss function
  rmse = np.sqrt(Y0_test_flat, predicted)
  r2 = r2_score(Y0_test_flat, predicted)
  print("2.e compute (and print) the corresponding loss function")
  print("SSE:", r2)
  return

def m1_plots_and_anwers_log2(degree):
  degree = degree
  #Flatten dataframes for M1
  X1_train_flat = np.array(X1_train).flatten()
  y1_train_flat = np.array(y1_train).flatten()
  X1_test_flat = np.array(X1_test).flatten()
  y1_test_flat = np.array(y1_test).flatten()
  #Define Weights and  (a) fit the model on Xtrain
  print("2.a fit the model on Xtrain")
  print("np.polyfit(np.log(X1_train_flat),y1_train_flat, degree")
  weights = np.polyfit(np.log(X1_train_flat),np.log(y1_train_flat), degree)
  #(b) print the weights (a, b, . . .)
  print("2.b print the weights (a, b, . . .)")
  print("Weights: ", weights)
  #Set model
  model = np.poly1d(weights)
  #Calculate predicted #(c) compute predicted values using Xtest
  predicted = model(np.log(X1_test_flat))
  print("2.c compute predicted values using Xtest")
  print("Predicted Values using Xtest: ", predicted)
  #(d) plot (if possible) predicted and actual values in Xtrain
  print("2.d plot (if possible) predicted and actual values in Xtrain")
  x_points = np.linspace(1,X1_test_max, 500)
  y = weights[0] + np.log(x_points) + weights[1]
  plt.scatter(np.log(X1_train_flat), np.log(y1_train_flat), color = 'red')
  plt.scatter(np.log(X1_train_flat), predicted, color = 'black')
 # plt.plot(np.log(x_points), y, color = 'blue')
  plt.title('Platelets vs Serum Sodium (Train set) Red = Actuals, Black = Predicted')
  plt.xlabel('Platelets')
  plt.ylabel('Serum Sodium')
  plt.xlim(4.6,5)
  plt.show()
  #(e) compute (and print) the corresponding loss function
  rmse = np.sqrt(mean_squared_error(y1_test_flat, predicted))
  r2 = r2_score(y1_test_flat, predicted)
  print("2.e compute (and print) the corresponding loss function")
  print("SSE:", r2)
  return

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
print("SURVIVING PATIENTS \n")
for i in range(1,4):
  print("--------------------- MODEL FOR DEGREE",i,"---------------------")
  m0_plots_and_anwers(i)
  print("--------------------------------------------------------------")
print("--------------------- MODEL FOR GLM = a log x + b---------------------")
m0_plots_and_anwers_log1(1)
print("--------------------------------------------------------------")
print("--------------------- MODEL FOR log y = a log x + b---------------------")
m0_plots_and_anwers_log2(1)
print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
print()
print("DECEASED PATIENTS \n")
for i in range(1,4):
  print("--------------------- MODEL FOR DEGREE",i,"---------------------")
  m1_plots_and_anwers(i)  
  print("--------------------------------------------------------------")
print()
print("--------------------- MODEL FOR GLM = a log x + b---------------------")
m1_plots_and_anwers_log1(1)
print("--------------------------------------------------------------")
print("--------------------- MODEL FOR log y = a log x + b---------------------")
m1_plots_and_anwers_log2(1)
print("--------------------------------------------------------------")
