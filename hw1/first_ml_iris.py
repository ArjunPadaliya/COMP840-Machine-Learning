
# coding: utf-8

# ## A "Hello World" Example of Machine Learning
# 
# Adopted from [Python Machine Learning - Your First Machine Learning Project in Python Step-By-Step](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)

# In[1]:


# Load libraries
import pandas as pd
import matplotlib.pyplot as plt


# ## Get the data  
# The dateset contains 150 observations of iris flowers. There are four features: four columns of measurements of the flowers in centimeters. The lable is the fifth column:  the species of the flower observed.

# In[2]:


import pandas as pd


# In[3]:


# Load CSV dataset into a pandas dataframe
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(url, names=names)


# ## Understand the data
# 
# To check the dimensions of the data and look at statistical summary of the attribues. Use graphs for visulization. 

# In[4]:


# shape
df.shape


# In[5]:


df.head(20)


# In[6]:


df.describe()


# In[7]:


df.tail(20)


# In[8]:


df.groupby('class').size()


# In[9]:


# use graph to show some feature distribution
df['petal-width'].plot()
plt.show()


# In[10]:


# create box and whisker plots to see the distribution of each feature
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# Or, create a histogram of each feature

# In[11]:


plt.show()


# Use scatter plot of all pairs of attributes to spot structured relationships between input variables.

# In[12]:


pd.plotting.scatter_matrix(df, figsize=[15,15], alpha=0.5)
plt.show()


# ## Prepare training dataset and validation dataset 

# In[13]:


from sklearn import model_selection


# In[14]:


array = df.values  # array is a numpy.ndarray type 


# In[15]:


array.shape


# In[16]:


print(array)


# X is all the input (features). So we slice the array to get all rows and first four colums (and skip the last column which is the class labels)

# In[17]:


X = array[:,0:4]   # 2D array sliding 


# In[18]:


X.shape


# In[19]:


print(X)


# Y is all the outputs (labels). So we slice the array to get all rows from the last colums 

# In[20]:


Y = array[:,4]    # Y is output (labels); one dimensional 


# In[21]:


Y.shape


# In[22]:


print(Y)


# Now, split-out training dataset and validation dataset. 
# 
# we are going to hold back some data that the algorithms will not get to see and we will use this data to get a second and independent idea of how accurate the best model might actually be.

# In[23]:


validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# ## Choose a model

# Import six models from scikit-learn and metrics to evaluate these models. 

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# We use the metric of ‘accuracy‘ to evaluate models. This is a ratio of the number of correctly predicted instances divided by the total number of instances in the dataset. 

# In[35]:


scoring = 'accuracy'


# Now train each of the models. We use 10-fold cross validation to estimate the model accuracy. This will split the training dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.

# In[36]:


models = []
models.append(('LR', LogisticRegression())) # Linear Discriminant Analysis
models.append(('LDA', LinearDiscriminantAnalysis()))  # Linear Discriminant Analysis
models.append(('KNN', KNeighborsClassifier()))  # K-Nearest Neighbors
models.append(('CART', DecisionTreeClassifier())) # Classification and Regression Trees
models.append(('NB', GaussianNB())) # Gaussian Naive Bayes
models.append(('SVC', SVC())) # Support Vector Classification
# evaluate each model in turn
results = []
names = []
seed = 8
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# ## Evalute the model
# 
# Let's select the model that produce the best accuracy. And then see how accurate this model is on our validation set.

# In[37]:


mymodel = SVC()
mymodel.fit(X_train, Y_train)
predictions = mymodel.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# ## Prediction 
# 
# Now let's try our model on new un-seen data

# In[28]:


Xnew = [[7.9, 3.8, 6.4, 2.0],[3.9, 1.8, 0.4, 1.9]]
ynew = mymodel.predict(Xnew)


# In[29]:


print(ynew)

