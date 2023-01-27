#!/usr/bin/env python
# coding: utf-8

# #Random Forest
#  
# Assignment
# 
# 
# About the data: 
# Let’s consider a Company dataset with around 10 variables and 400 records. 
# The attributes are as follows: 
# 
#  Sales -- Unit sales (in thousands) at each location
# 
#  Competitor Price -- Price charged by competitor at each location
# 
#  Income -- Community income level (in thousands of dollars)
# 
#  Advertising -- Local advertising budget for company at each location (in thousands of dollars)
# 
#  Population -- Population size in region (in thousands)
# 
#  Price -- Price company charges for car seats at each site
# 
#  Shelf Location at stores -- A factor with levels Bad, Good and Medium 
# 
# indicating the quality of the shelving location for the car seats at each site
# 
#  Age -- Average age of the local population
# 
#  Education -- Education level at each location
# 
#  Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location
# 
#  US -- A factor with levels No and Yes to indicate whether the store is in the US or not
# 
# 
# The company dataset looks like this: 
#  
# Problem Statement:
# 
# A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
# Approach - A Random Forest can be built with target variable Sales (we will first convert it in categorical variable) & all other variable will be independent in the analysis.  
# 

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder#for train test splitting
from sklearn.model_selection import train_test_split#for decision tree object
from sklearn.tree import DecisionTreeClassifier#for checking testing results
from sklearn.metrics import classification_report, confusion_matrix#for visualizing tree 
from sklearn.tree import plot_tree


# In[76]:


# Pandas is used for data manipulation
import pandas as pd
# Read in data and display first 5 rows
features = pd.read_csv('Company_Data.csv')
features.head(5)


# #EDA
# 

# In[77]:


#getting information of dataset
features.info()


# In[78]:


print('The shape of our features is:', features.shape)


# In[79]:


features.isnull().any()


# In[80]:


# let's plot pair plot to visualise the attributes all at once
sns.pairplot(data=features,hue = 'ShelveLoc')


# In[7]:


#Creating dummy vairables dropping first dummy variable
df=pd.get_dummies(features,columns=['Urban','US'], drop_first=True)


# In[8]:


print(df.head())


# In[9]:


from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[10]:



df['ShelveLoc']=df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})


# In[11]:



print(df.head())


# In[14]:


x=df.iloc[:,1:11]
y=df['Sales']
x


# In[15]:


y


# In[16]:


df['ShelveLoc'].unique()


# In[17]:


df.ShelveLoc.value_counts()


# In[18]:


colnames = list(df.columns)
colnames


# In[19]:


# Descriptive statistics for each column
df.describe()


# In[20]:


df.head()


# In[22]:


# Labels are the values we want to predict
labels = np.array(df['Sales'])
# Remove the labels from the features
# axis 1 refers to the columns
features= df.drop('Sales', axis = 1)
# Saving feature names for later use
feature_list = list(df.columns)
# Convert to numpy array
features = np.array(df)


# In[23]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)                                                                  


# In[24]:


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# #Establish Baseline

# In[25]:


# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('Sales')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))


# In[26]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);


# In[27]:


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# #Determine Performance Metrics

# In[28]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[29]:


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


# In[30]:


# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');


# In[31]:


# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[32]:


# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
important_indices = [feature_list.index('Sales'), feature_list.index('Income')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')


# In[33]:


# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# 
# #Business Problem:

# #Use Random Forest to prepare a model on fraud data treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

# In[34]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn.tree import  DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[35]:


data=pd.read_csv('Fraud_check.csv',encoding='latin1')
data.head()


# In[36]:


data.info()


# In[37]:


data.describe()


# In[38]:


data.isnull().sum()


# In[39]:


df=data.copy()
df.head(2)


# In[40]:


df.insert(6,'tax_class'," ")

for i in range(0,len(df['Taxable.Income']),1):
    if df['Taxable.Income'].iloc[i]<=30000:
        df['tax_class'].iloc[i]='Risky'
    else:
        df['tax_class'].iloc[i]='Good'


# In[41]:


df.head()


# In[42]:


df.tax_class.value_counts()


# In[43]:


df.drop(['Taxable.Income'],axis=1,inplace=True)
df.head()


# In[44]:


#Complete Iris dataset
label_encoder = preprocessing.LabelEncoder()
df['Undergrad']= label_encoder.fit_transform(df['Undergrad'])
df['Marital.Status']= label_encoder.fit_transform(df['Marital.Status'])
df['Urban']= label_encoder.fit_transform(df['Urban'])
df['tax_class']= label_encoder.fit_transform(df['tax_class'])


# In[45]:


df.head()


# In[46]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn import preprocessing


# In[47]:


df.iloc[:,2:4]=StandardScaler().fit_transform(df.iloc[:,2:4])
df


# In[48]:


# Visulazing the distibution of the data for every feature
df.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20));


# In[49]:


df.to_csv('auto.csv')


# In[50]:


plt.figure(figsize=(14,10))
import seaborn as sns
sns.heatmap(df.corr(),annot=True,cmap='terrain')


# In[51]:


pdata=df.copy()
pdata.head(2)


# In[52]:


x=pdata.iloc[:,0:-1]
y=pdata['tax_class']
y.value_counts()


# #Model building using Random forest
# 

# In[53]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
from sklearn.ensemble import RandomForestClassifier 


# In[54]:


modelb=RandomForestClassifier()
modelb.fit(x_train,y_train)
pred=modelb.predict(x_test)
print(np.mean(y_test==pred))


# #Using grid search optimisation

# from sklearn.model_selection import GridSearchCV gmodel=RandomForestClassifier() param_grid=({'n_estimators':[100,200,300,400,500],'criterion':['gini','entropy'],'max_depth':range(2,20),'max_leaf_nodes':range(2,20)}) grid=GridSearchCV(gmodel,param_grid) grid.fit(x,y) print(grid.bestscore) print(grid.bestparams)

# In[55]:


modelb=RandomForestClassifier(criterion='gini',max_depth= 19,max_leaf_nodes=19,n_estimators=1000)
modelb.fit(x,y)
pred=modelb.predict(x)
print(np.mean(y==pred))


# In[56]:


modelb=RandomForestClassifier(criterion='gini',max_depth= 19,max_leaf_nodes=19,n_estimators=1000)
modelb.fit(x_train,y_train)
pred=modelb.predict(x_train)
print(np.mean(y_train==pred))


# #Applying K-Fold cross Validation

# In[57]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold=KFold(n_splits=10)
results=cross_val_score(modelb,x,y,cv=kfold)
print(results.mean())


# #Bagging

# In[58]:


modelb1=RandomForestClassifier(n_estimators=500,criterion='entropy',max_leaf_nodes=19,max_depth=19)
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=modelb1)
results=cross_val_score(model,x,y,cv=kfold)
print(results.mean())


# #Boosting

# In[59]:


from sklearn.ensemble import AdaBoostClassifier
model=AdaBoostClassifier(base_estimator=modelb)
results=cross_val_score(model,x,y,cv=kfold)
print(results.mean())


# Voting classifier

# In[60]:


from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
estimators=[]
model1=LogisticRegression(max_iter=500,solver='liblinear')
estimators.append(('logistic',model1))
model2=DecisionTreeClassifier()
estimators.append(('cart',model2))
model3=SVC()
estimators.append(('svm',model3))
model4=AdaBoostClassifier(base_estimator=modelb1)
estimators.append(('adaboost',model4))
model5=BaggingClassifier(base_estimator=modelb1)
estimators.append(('bagging',model5))
model6=RandomForestClassifier(n_estimators=500,criterion='entropy',max_leaf_nodes=19,max_depth=19)
estimators.append(('RandomForest',model6))

ensemble=VotingClassifier(estimators)
results=cross_val_score(ensemble,x,y,cv=kfold)
print(results)


# In[ ]:


print(results.mean())


# #
# XGBM

# In[61]:


from tqdm.notebook import tqdm, trange
import time
# First XGBoost model for dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[62]:


xgmodel = XGBClassifier()
xgmodel.fit(x_train, y_train)


# In[63]:


# make predictions for test data
y_pred = xgmodel.predict(x_test)
predictions = [round(value) for value in y_pred]


# In[64]:


y_pred1 = xgmodel.predict(x_train)
predictions1 = [round(value) for value in y_pred1]


# In[65]:



# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[66]:


# evaluate predictions
accuracy = accuracy_score(y_train, predictions1)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

