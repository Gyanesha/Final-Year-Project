import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from  sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv("/home/gp/Desktop/Project/Random.csv")
train = 0.6


x = df.iloc[:, [i for i in range(0, 64)]].values    
y = df.iloc[:, 64].values
xtrain, xtest, ytrain, ytest = train_test_split( 
        x, y, test_size = train, random_state = 200) 

print(xtrain)
print(ytrain)

classifier = LogisticRegression(random_state = 100) 
classifier.fit(xtrain, ytrain) 

y_pred = classifier.predict(xtest)
cm = confusion_matrix(ytest, y_pred) 
  
print ("Confusion Matrix : \n", cm) 
print ("Accuracy : ", accuracy_score(ytest, y_pred)) 


