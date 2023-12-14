#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, 


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sn


# In[93]:


#reading the csv file and checking the dataframe 
df = pd.read_csv(r"C:\Users\errav\Downloads\archive (2)\Credit_card.csv")
df


# Checking for missing values

# In[94]:


df.isna().sum()/len(df)*100


# In[95]:


#removing column type_occupation due to more than 30% missing values
df.drop(['Type_Occupation'],axis=1,inplace = True)


# Handling the missing values by iputing the values with mean , mode and median

# In[96]:


df['Annual_income'].fillna(df.Annual_income.mean(),inplace = True) #Mean imputation
df['Birthday_count'].fillna(df.Birthday_count.mean(),inplace = True) #mean imputation
df 


# In[97]:


df.isna().sum()/len(df)*100


# In[98]:


df['GENDER'].fillna(df['GENDER'].mode()[0],inplace = True) #mode imputation 
df.isna().sum()


# In[99]:


#checking the dataframe 
df


# # Exploratory data analysis

# In[169]:


label_font_dict = {'family':'sans-serif','size':13.5,'color':'brown','style':'italic'}
title_font_dict = {'family':'sans-serif','size':16.5,'color':'Blue','style':'italic'}


# In[100]:


#checking for annual income based on education 
df.groupby('EDUCATION').Annual_income.describe()


# In[ ]:





# In[188]:


fig = sns.countplot(y='Type_Income', hue='EDUCATION', orient = "h"  ,width=1.5 , data=df)

# Adding labels and title
plt.ylabel('Income_Type')
plt.xlabel('Count')
plt.title('Bar Chart of Income_Type and Education')

total = len(df)  # total number of observations
for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/df.shape[0],2))+"%"}', (x + width/2, y + height*1.02), ha='center', fontsize=6.5, rotation=0)

# Adding legend
plt.legend(title='Education')

# Show the plot
plt.show()


# From the above stacked bar chart , we can say that mostly academic degree holders are working in every sector of income , mostly of them are working. so approval of credit card for working persons who are having academics is more easy compared to others

# In[191]:


fig = sns.countplot(y='CHILDREN', hue='EDUCATION', width=2.5, data=df)

# Adding labels and title
plt.ylabel('Children')
plt.xlabel('Count')
plt.title('Bar Chart of Children and Education')

total = len(df)  # total number of observations
for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/df.shape[0],2))+"%"}', (x + width/2, y + height*1.02), ha='center', fontsize=6.5, rotation=0)

# Adding legend
plt.legend(title='Education')

# Show the plot
plt.show()


# from this , we can Most childern are with least educated persons.

# In[177]:


fig = sns.countplot(x='Propert_Owner', hue='GENDER', data=df)

# Adding labels and title
plt.ylabel('Count')
plt.xlabel('Property Owner')
plt.title('Bar Chart of Gender and Property Owner')

total = len(df)  # total number of observations
for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/df.shape[0],2))+"%"}', (x + width/2, y + height*1.02), ha='center', fontsize=8.5, rotation=0)

# Adding legend
plt.legend(title='Education')

# Show the plot
plt.show()


# From this , we can say that males are dominating females in owning a property

# In[176]:


fig = sns.countplot(y='EDUCATION', hue='GENDER', data=df)

# Adding labels and title
plt.ylabel('EDUCATION')
plt.xlabel('GENDER')
plt.title('Bar Chart of Education and Gender')

total = len(df)  # total number of observations
for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/df.shape[0],2))+"%"}', (x + width/2, y + height*1.02), ha='center', fontsize=8.5, rotation=0)

# Adding legend
plt.legend(title='Education')

# Show the plot
plt.show()


# From this , we can say that , Female are dominating the males in academics

# # Importing Label csv file 

# Label: 0 is application approved and 1 is application rejected

# In[134]:


df_labels = pd.read_csv(r"C:\Users\errav\Downloads\archive (2)\Credit_card_label.csv")
df_labels


# Adding labels to dataframe 

# In[135]:


df['label'] = df_labels.label


# In[31]:


df.head()


# In[26]:


df.groupby('GENDER').label.describe()


# In[22]:


plt.scatter(df.Annual_income,df.label)


# In[ ]:





# In[23]:


df.groupby('label').Annual_income.describe()


# In[174]:


fig =sns.countplot(x='label', hue='EDUCATION', data=df)

# Adding labels and title
plt.xlabel('label')
plt.ylabel('Count')
plt.title('Bar Chart of label and Education')

total = len(df)  # total number of observations
for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/df.shape[0],2))+"%"}', (x + width/2, y + height*1.02), ha='center', fontsize=8.5, rotation=0)

# Adding legend
plt.legend(title='Education')

# Show the plot
plt.show()


# From this , we can clearly say that , educated people's credit card approval chances are high and also if you are just secondary educated person and having a good credit history , then you can easily get credit card

# In[175]:


fig = sns.countplot(x='label', hue='Car_Owner', data=df)

# Adding labels and title
plt.xlabel('label')
plt.ylabel('Count')
plt.title('Bar Chart of label and Car Owner')

# Adding legend
plt.legend(title='Car Owner')
total = len(df)  # total number of observations
for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/df.shape[0],2))+"%"}', (x + width/2, y + height*1.02), ha='center', fontsize=8.5, rotation=0)

# Show the plot
plt.show()


# From this bar chart , we can say that , Most number of card owned by people who is having credit card.

# # Data Preprocessing

# In[38]:


from sklearn.preprocessing import LabelEncoder
text_cols = ['GENDER','Car_Owner','Propert_Owner','Type_Income','EDUCATION','Marital_status','Housing_type']
for i in text_cols:
    df[i] = LabelEncoder().fit_transform(df[i])


# In[39]:


df


# In[40]:


from sklearn.preprocessing import StandardScaler


# In[41]:


scaler = StandardScaler()
features = scaler.fit_transform(df.drop(['Ind_ID','label'],axis=1))
features


# In[42]:


target = df.label
target


# In[43]:


features.shape


# In[44]:


target.value_counts()


# # Splitting data and creating ANN model

# In[45]:


from tensorflow.keras.models import Sequential
# Is responsbile for combining the Layers
from tensorflow.keras.layers import InputLayer, Dense
# InputLayer => for Inputs (Optional)
# Dense => Hidden layer (Fully connected Layer)


# In[46]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(features,target,test_size=0.3,random_state=42)


# Data is imbalanced so using SMOTE for balancing the data 

# In[48]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()


# In[49]:


x_train_smote , y_train_smote = smote.fit_resample(X_train, y_train)


# In[50]:


from collections import Counter
print("Before SMOTE :" , Counter(y_train))
print("After SMOTE :" , Counter(y_train_smote))


# In[51]:


#importing useful libraries
from sklearn.datasets import make_classification , make_regression
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input , Dense , Add , Concatenate 
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import plot_model


# Callback for earlystopping

# In[54]:


from tensorflow.keras.callbacks import EarlyStopping
cb_earlystop = EarlyStopping(monitor='val_loss',
                            min_delta=0.01,\
                            patience=5,
                            verbose=1,
                            mode='auto',
                            restore_best_weights=True,
                            start_from_epoch=0)


# # VANILLA ANN Structure

# In[57]:


input_c = Input(shape=(16,), name = 'input1')
dense1 = Dense(30 , activation = 'relu' , name = 'hidden_1')(input_c)
dense2 = Dense(20 , activation = 'relu' , name = 'hidden_2')(input_c)
dense3 = Dense(8 , activation = 'relu' , name  = 'hidden_3')(dense1)
dense4 = Dense(5 , activation = 'relu' , name = 'hidden_4' )(dense2)
concat1 = Concatenate(name= 'concat1')([dense3 , dense4])
divide1 = Dense(3 , activation = 'relu' , name = 'divide1')(concat1)
divide2 = Dense(2 , activation = 'relu' , name = 'divide2')(concat1)
concat2 = Concatenate(name = 'concate')([divide1, concat1])
ouput1  = Dense(1, activation = 'sigmoid' , name = 'Output1')(concat2)
model_c = Model(inputs = input_c, outputs = ouput1)
model_c.compile(optimizer = 'adam' , loss='binary_crossentropy' , metrics= ['accuracy'])


# Fitting the model

# In[60]:


fmodel = model_c.fit(x_train_smote, y_train_smote, epochs=30, validation_data = [X_test , y_test] ,\
                   callbacks=[cb_earlystop], verbose=2)


# Evaluating the model on testing data

# In[68]:


prediction_score = model_c.evaluate(X_test, y_test, verbose=0)

print('Test Loss and Test Accuracy', prediction_score)


# predicting the model on testing data

# In[73]:


y_pred = model_c.predict(X_test, batch_size=1000)


# # Checking for accuracy , f1 score and matrices for model checking

# In[77]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[70]:


def draw_chart(history):
  plt.figure(figsize=(15,8))

  plt.subplot(1,2,1)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Losses vs Epochs')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend(['Train', 'Validation'])

  plt.subplot(1,2,2)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Accuracy vs Epochs')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['Train', 'Validation'])


# In[71]:


draw_chart(fmodel)


# From this , we can say that validation loss and training loss , both are reducing 
# and in Accuracy , train accuracy is better than validation accuracy

# In[79]:


# Accuracy: (TP + TN) / (P + N)
accuracy = accuracy_score(y_test, y_pred.round())
print('Accuracy: %f' % accuracy)
# Precision: TP / (TP + FP)
precision = precision_score(y_test, y_pred.round())
print('Precision of DNN Model: %f' % precision)
# Recall: TP / (TP + FN)
recall = recall_score(y_test, y_pred.round())
print('Recall DNN Model: %f' % recall)
# F1: 2 TP / (2 TP + FP + FN)
f1 = f1_score(y_test, y_pred.round())
print('F1-score DNN Model: %f' % f1)


# # Using keras tuner for checking the best parameteres and best model for this data

# In[80]:


import kerastuner as kt


# In[81]:


# Optimizer Selection, Number of Neurons in Each Layer, number of Layers
def build_model(hp):
    model =Sequential()
    act = hp.Choice('Intial_Act', ['relu','selu','elu','softplus'])
    model.add(Dense(hp.Int("First_Hidden", min_value=1, max_value=20, step=3), input_dim=16, activation=act))
    
    for i in range(hp.Int("Num_Layers", min_value=0, max_value=10, step=1)):
        model.add(Dense(hp.Int("units_"+str(i), min_value=2, max_value=8,step=1), 
                              activation=hp.Choice("Activation_"+str(i),
                                                   values=['relu','selu','elu','softplus'])))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = hp.Choice('Optimizer',values = ['adam','RMSprop','adagrad','sgd'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[82]:


random_tuner = kt.RandomSearch(build_model, 
                              objective='val_accuracy',
                              max_trials=10,
                              directory='diabetes_model',
                              project_name = "diab_tuner_all")


# In[83]:


random_tuner.search(x_train_smote, y_train_smote, epochs=20, validation_data = [X_test , y_test] ,\
                   callbacks=[cb_earlystop], verbose=2)


# In[84]:


random_tuner.search_space_summary()


# In[85]:


random_tuner.get_best_hyperparameters()[0].values


# In[86]:


fmodel = random_tuner.get_best_models(num_models = 1)[0]


# In[87]:


fmodel.summary()


# In[88]:


fmodel.evaluate(X_test , y_test)


# # Conclusion:

# 1. For credit card approval , if you are a academic graduation person , then your credit card rejection chance will be least.
# 2. For credit card approval , you must have working , graduated , good income , and good credit history also.
# 3. Type of income , property owner , gender etc are  not playing a most important role in credit card approval , if you are not having a good credit history .
# 4. The data belongs to outside of india , so  i shocked when i saw that in this present data there is not any approval in the list of academic graduates person .
# 

# # Project Motive

# From the given data I have completed an analysis and created a model for checking the credit card approval by using ANN Structure .
# This model will gave 81 % accuracy on testing .
# 

# # Thank you !

# In[ ]:


feature_importance = best_estimator.feature_importances_

sorted_idx = feature_importance.argsort()[::-1]

colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_idx)))

plt.figure(figsize=(13, 10))
plt.barh(range(X_train.shape[1]), feature_importance[sorted_idx], color=colors)
plt.yticks(range(X_train.shape[1]), X_train.columns[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()

