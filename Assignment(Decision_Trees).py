#!/usr/bin/env python
# coding: utf-8

# #Use decision trees to prepare a model on fraud data 
# treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
# 
# Data Description :
# 
# Undergrad : person is under graduated or not
# Marital.Status : marital status of a person
# Taxable.Income : Taxable income is the amount of how much tax an individual owes to the government 
# Work Experience : Work experience of an individual person
# Urban : Whether that person belongs to urban area or not

# In[3]:


#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[4]:


df = pd.read_csv("Fraud_check.csv")
df.head()


# In[5]:



df.tail()


# In[6]:


#Creating dummy vairables for ['Undergrad','Marital.Status','Urban'] dropping first dummy variable
df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)


# In[7]:


#Creating new cols TaxInc and dividing 'Taxable.Income' cols on the basis of [10002,30000,99620] for Risky and Good
df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])


# In[8]:



print(df)


# #Lets assume: taxable_income <= 30000 as “Risky=0” and others are “Good=1”

# In[9]:


#After creation of new col. TaxInc also made its dummies var concating right side of df
df = pd.get_dummies(df,columns = ["TaxInc"],drop_first=True)


# In[10]:


#Viewing buttom 10 observations
df.tail(10)


# In[11]:


# let's plot pair plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data=df, hue = 'TaxInc_Good')


# In[12]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[13]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:,1:])
df_norm.tail(10)


# In[14]:


# Declaring features & target
X = df_norm.drop(['TaxInc_Good'], axis=1)
y = df_norm['TaxInc_Good']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


# Splitting data into train & test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)


# In[17]:


##Converting the Taxable income variable to bucketing. 
df_norm["income"]="<=30000"
df_norm.loc[df["Taxable.Income"]>=30000,"income"]="Good"
df_norm.loc[df["Taxable.Income"]<=30000,"income"]="Risky"


# In[18]:


##Droping the Taxable income variable
df.drop(["Taxable.Income"],axis=1,inplace=True)


# In[19]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass


# In[20]:


##Splitting the data into featuers and labels
features = df.iloc[:,0:5]
labels = df.iloc[:,5]


# In[21]:


## Collecting the column names
colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]
##Splitting the data into train and test


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)


# In[23]:


##Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)


# In[24]:


model.estimators_
model.classes_
model.n_features_
model.n_classes_


# In[25]:


model.n_outputs_


# In[26]:


model.oob_score_
###74.7833%


# In[27]:


##Predictions on train data
prediction = model.predict(x_train)


# In[28]:


##Accuracy
# For accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)
##98.33%


# In[29]:


np.mean(prediction == y_train)
##98.33%


# In[30]:


##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)


# In[31]:


##Prediction on test data
pred_test = model.predict(x_test)


# In[32]:


##Accuracy
acc_test =accuracy_score(y_test,pred_test)
##78.333%


# In[33]:


## In random forest we can plot a Decision tree present in Random forest
from sklearn.tree import export_graphviz
import pydotplus
from six import StringIO


# In[34]:


tree = model.estimators_[5]


# In[35]:


dot_data = StringIO()
export_graphviz(tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity=False)


# In[36]:


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# #Building Decision Tree Classifier using Entropy Criteria

# In[37]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[38]:


from sklearn import tree


# In[39]:


#PLot the decision tree
tree.plot_tree(model);


# In[40]:


colnames = list(df.columns)
colnames


# In[41]:


fn=['population','experience','Undergrad_YES','Marital.Status_Married','Marital.Status_Single','Urban_YES']
cn=['1', '0']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[42]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[43]:


preds


# In[44]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[45]:


# Accuracy 
np.mean(preds==y_test)


# #Building Decision Tree Classifier (CART) using Gini Criteria

# In[46]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[47]:



model_gini.fit(x_train, y_train)


# In[48]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# #Decision Tree Regression Example

# In[49]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[50]:


array = df.values
X = array[:,0:3]
y = array[:,3]


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[52]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[53]:


#Find the accuracy
model.score(X_test,y_test)


# In[53]:





# Decision Tree
#  
# #Assignment
# 
# 
# About the data:
# 
# Let’s consider a Company dataset with around 10 variables and 400 records. 
# 
# The attributes are as follows:
# 
#  Sales -- Unit sales (in thousands) at each location
#  Competitor Price -- Price charged by competitor at each location
#  Income -- Community income level (in thousands of dollars)
#  Advertising -- Local advertising budget for company at each location (in thousands of dollars)
#  Population -- Population size in region (in thousands)
#  Price -- Price company charges for car seats at each site
#  Shelf Location at stores -- A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
#  Age -- Average age of the local population
#  Education -- Education level at each location
#  Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location
#  US -- A factor with levels No and Yes to indicate whether the store is in the US or not
# The company dataset looks like this: 
#  
# Problem Statement:
# 
# A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
# Approach - A decision tree can be built with target variable Sale (we will first convert it in categorical variable) & all other variable will be independent in the analysis.  
# 

# In[54]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report,confusion_matrix,accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[55]:


# Import Dataset
df = pd.read_csv('Company_Data.csv')
df


# Data Exploration

# In[56]:


df.isnull().any()


# In[60]:


df.isnull().sum().sum()


# We have no missing data so all the entries are valid for use.
# 
# Now we can check the column names to get a better understanding of what features we will be basing our classification on.

# #Descriptive Analysis

# In[58]:


df.shape


# In[59]:


df.info()


# In[61]:


df[df.duplicated()].shape


# In[62]:


df[df.duplicated()]


# In[63]:


df.describe()


# Feature Engineering

# In[64]:


df.Sales.describe()


# In[65]:


plt.hist('Sales',data=df)
plt.show()


# As the Sales distribution is not skewed and it is normal distribution we can take mean or median as a threshold to make two or three categories 'Low', 'Medium' and 'High' Sales For two categories lets assume: Sales < 7.5 as “Bad=0” and Sales > 7.5 as “Good=1” and For three categories lets divide the Sales values with Quartiles , less than first quartile as 'Low' , between Second Quartile as 'Medium and above Third Quartile as 'High'

# In[66]:


df.Sales.max()


# In[67]:


16.270000/3


# In[68]:


5.423333333333333*2


# In[69]:


5.423333333333333*3


# In[70]:


# Converting taxable_income <= 30000 as "Risky" and others are "Good"
df1=df.copy()
df1['Sales_cat'] = pd.cut(x = df1['Sales'], bins = [0,5.39,9.32,17], labels=['Low','Medium','High'], right = False)
df1.head()


# In[71]:


df1.Sales_cat.value_counts()


# In[72]:


df1.info()


# In[73]:


categorical_features = df1.describe(include=["object",'category']).columns
categorical_features


# In[74]:


numerical_features = df1.describe(include=["int64","float64"]).columns
numerical_features


# #Data Visualization
# Univariate plots

# In[75]:


numerical_features=[feature for feature in df.columns if df[feature].dtypes != 'O']
for feat in numerical_features:
    skew = df[feat].skew()
    sns.distplot(df[feat], kde= False, label='Skew = %.3f' %(skew), bins=30)
    plt.legend(loc='best')
    plt.show()


# In[76]:


numerical_features


# In[77]:


ot=df.copy() 
fig, axes=plt.subplots(8,1,figsize=(14,12),sharex=False,sharey=False)
sns.boxplot(x='Sales',data=ot,palette='crest',ax=axes[0])
sns.boxplot(x='CompPrice',data=ot,palette='crest',ax=axes[1])
sns.boxplot(x='Income',data=ot,palette='crest',ax=axes[2])
sns.boxplot(x='Advertising',data=ot,palette='crest',ax=axes[3])
sns.boxplot(x='Population',data=ot,palette='crest',ax=axes[4])
sns.boxplot(x='Price',data=ot,palette='crest',ax=axes[5])
sns.boxplot(x='Age',data=ot,palette='crest',ax=axes[6])
sns.boxplot(x='Education',data=ot,palette='crest',ax=axes[7])
plt.tight_layout(pad=2.0)


# In[78]:


#outlier
plt.figure(figsize=(14,6))
sns.boxplot(data=df[numerical_features], orient="h")


# Multivariate Analysis

# In[79]:


plt.figure(figsize=(8,8))
sns.pairplot(df, palette='coolwarm')
plt.show()


# In[80]:


# Having a look at the correlation matrix

fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, fmt='.1g', cmap="viridis", cbar=False, linewidths=0.5, linecolor='black')


# In[81]:


# let's plot pair plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data=df1, hue = 'Sales_cat')


# In[82]:


data_ = df1.copy()
data_.drop('Sales',axis=1, inplace =True)
data_ = pd.get_dummies(data_.iloc[:,:-1])
data_.head()


# Checking for Imbalanced or balanced dataset with regards to the Target

# In[83]:


df1.Sales_cat.value_counts()


# In[84]:


sns.countplot(x='Sales_cat', data=df1, palette = 'viridis', 
              order=df1['Sales_cat'].value_counts().index)
plt.xticks(fontsize = 12)
plt.title('Low Medium or Good for Sales')


# In[85]:


data_ = df1.copy()
data_.drop('Sales',axis=1, inplace =True)
data_ = pd.get_dummies(data_.iloc[:,:-1])
data_.head()


# In[86]:


data_['Sales'] = df1.Sales_cat
data_.head()


# In[87]:


le = LabelEncoder()
le.fit(data_["Sales"])
data_["Sales"]=le.transform(data_["Sales"])
data_.head()


# In[96]:


# split into input (X) and output (y) variables
X = data_.iloc[:, :-1]

y=  data_.Sales


# In[89]:


#Feature importance
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest, chi2


# In[97]:


# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)


# In[98]:


# summarize scores
scores = fit.scores_

features = fit.transform(X)


# In[99]:


score_df = pd.DataFrame(list(zip(scores, X.columns)),
               columns =['Score', 'Feature'])
score_df.sort_values(by="Score", ascending=False, inplace=True)
score_df


# In[100]:


fig, axes = plt.subplots(figsize=(20, 6))
plt.bar([i for i in range(len(scores))],scores)
axes.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
axes.set_xticklabels(X.columns.values)
plt.xticks(rotation = 90, size = 15)
plt.show()


# In[101]:


model_data = data_[['Price', 'Advertising','Population', 'Income', 'Age', 'ShelveLoc_Good', 'ShelveLoc_Bad', 'ShelveLoc_Medium','Sales']]
model_data.head()


# In[102]:


x = model_data.drop('Sales',axis=1)
y = model_data['Sales']


# In[103]:


y.unique()


# In[104]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0,stratify=y)


# In[105]:


print("Shape of X_train: ",x_train.shape)
print("Shape of X_test: ", x_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)


# Building Decision Tree Classifier using Entropy Criteria

# In[106]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[107]:


from sklearn import tree


# In[108]:


#PLot the decision tree
fig,ax = plt.subplots(figsize=(10.,8))
tree.plot_tree(model);


# In[109]:


fn=['Sales','CompPrice','Income','Advertising','Population','Price']
cn=['1', '2', '3']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[110]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category


# In[111]:


preds


# In[112]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[113]:


# Accuracy 
np.mean(preds==y_test)


# Building Decision Tree Classifier (CART) using Gini Criteria

# In[114]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[115]:


model_gini.fit(x_train, y_train)


# In[116]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# Decision Tree Regression Example

# In[117]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[118]:


array = df.values
X = array[:,0:3]
y = array[:,3]


# In[119]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[120]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[121]:


#Find the accuracy
model.score(X_test,y_test)

