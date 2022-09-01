#Importing the packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plot
import seaborn as sea
import plotly.express as px
import gradio as gr
import warnings
warnings.filterwarnings('ignore')  #never print matching warnings
plot.style.use('fivethirtyeight') #to replicate the styles from FiveThirtyEight.com.
###############
df= pd.read_csv('processed_kidney_disease.csv')
#############
#We are renaming the column name into something that makes sense
df.columns = ['Age','Blood_Pressure','Specific_Gravity','Albumin','Sugar','Red_Blood_Cells','Pus_Cells','Puss_Cell_Clumps','Bacteria',
              'Blood_Gulcose_Random','Blood_Urea','Serum_Creatinine','Sodium','Potassium','Haemoglobin','Packed_Cell_Volume',
              'White_Blood_Cell_Count','Red_Blood_Cell_Count','Hypertension','Diabetes_Mellitus','Coronary_Artery_Disease',
              'Appetite','Peda_Edema','Aanemia', 'clas_s']
###############
print(df.head())
###############
categories_Columns = [col for col in df.columns if df[col].dtype == 'object']
number_columns =  [col for col in df.columns if df[col].dtype != 'object']
################
for col in categories_Columns:
     print(f"{col} has {df[col].nunique()} categories\n")

     ###########
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in categories_Columns:
  df[col] = le.fit_transform(df[col])
############()()()()
ind_col = [col for col in df.columns if col != 'clas_s']
dep_col = 'clas_s'

X = df[ind_col]
y = df[dep_col]


###############()()()()
# splitting data inputs as training and test sets
#training set—a subset to train a model
#test set—a subset to test the trained model.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

##################

#import model object
from sklearn.tree import DecisionTreeClassifier
model =  DecisionTreeClassifier()

#train model on training data
model.fit(X_train, y_train)

#getting model performance on test data
print("Accuracy of DecisionTreeClassifier  is  :", model.score(X_test, y_test))

for col in categories_Columns:
     print(f"{col} has {df[col].nunique()} categories\n")
#########################
##########################################
def Chronic_Kidney_Disease(Age,Blood_Pressure,Specific_Gravity,Albumin,Sugar,Red_Blood_Cells,Pus_Cells,
                           Puss_Cell_Clumps,Bacteria,Blood_Gulcose_Random,Blood_Urea,Serum_Creatinine,
                           Sodium,Potassium,Haemoglobin,Packed_Cell_Volume,White_Blood_Cell_Count,Red_Blood_Cell_Count,
                           Hypertension,Diabetes_Mellitus,Coronary_Artery_Disease, Appetite,Peda_Edema,Aanemia):
                           
                           x = np.array([Age,Blood_Pressure,Specific_Gravity,Albumin,Sugar,Red_Blood_Cells,Pus_Cells,
                           Puss_Cell_Clumps,Bacteria,Blood_Gulcose_Random,Blood_Urea,Serum_Creatinine,
                           Sodium,Potassium,Haemoglobin,Packed_Cell_Volume,White_Blood_Cell_Count,Red_Blood_Cell_Count,
                           Hypertension,Diabetes_Mellitus,Coronary_Artery_Disease, Appetite,Peda_Edema,Aanemia])
                           prediction = model.predict(x.reshape(1, -1))
                           return prediction   

################
outputs = gr.outputs.Textbox()
app = gr.Interface(fn=Chronic_Kidney_Disease, 
inputs= ['number','number','number','number','number','number','number','number','number','number','number','number','number','number','number','number','number','number','number','number','number','number','number','number'], 
outputs=outputs,description="Early Chronic Kidney Diesease Predictor")
#################
app.launch()
app.launch(share=True)
