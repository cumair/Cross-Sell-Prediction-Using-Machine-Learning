#!/usr/bin/env python
# coding: utf-8

# ## Project Notebook
# ### *DSC540 Advanced Machine Learning*
# # Health Insurance Cross Sell Prediction
# ### A measure of health insurance owner interest in vehicle insurance
# ## Analysis Using Machine Learning Techniques
# 
# ### *Umair Chaanda*

# In[1]:


# Import library
from IPython.display import Image
# Load image from local storage
Image(filename = "Cross-selling-Single.jpg")


# # 1. Abstract

# Cross-selling is a frequently employed marketing strategy used by many companies to generate positive revenue flow from an existing customer network. For the purpose of project here, the objective is to assist an insurance company in constructing a predictive model that will forecast the likelihood of current health insurance policy holders enlisting in the company’s vehicle insurance as well. This dedicated classification task, which also entails associated misclassification costs, utilized Gaussian, Decision Tree, Linear Discriminant Analysis (LDA), Rocchio, Random Forest, Ada Boost, Gradient Descent, and Logistic Regression classifiers in an effort to isolate the single most effective forecast prototype. A series of intensive modeling later revealed the base model, the Naive Bayes (Gaussian) classifier, to be the most efficacious of prediction frameworks.
# This process of formulating an optimal model to prognosticate customer interest in vehicle insurance serves immense value to both the company’s sales and strategy divisions. Additionally, it provides a suitable blueprint that can be used to bolster its current business template and supplementary revenue streams.

# # 2. Introduction

# An insurance policy is an arrangement by which a company undertakes to provide a guarantee of compensation for specified loss, damage, illness, or death in return for the payment of a specified premium. A premium is a sum of money that the customer needs to pay regularly to an insurance company for this guarantee.
# 
# For example, you may pay a premium of Rs. 5000 each year for a health insurance cover of Rs. 200,000/- so that if, God forbid, you fall ill and need to be hospitalised in that year, the insurance provider company will bear the cost of hospitalisation etc. for upto Rs. 200,000. Now if you are wondering how can company bear such high hospitalisation cost when it charges a premium of only Rs. 5000/-, that is where the concept of probabilities comes in picture. For example, like you, there may be 100 customers who would be paying a premium of Rs. 5000 every year, but only a few of them (say 2-3) would get hospitalised that year and not everyone. This way everyone shares the risk of everyone else.
# 
# Just like medical insurance, there is vehicle insurance where every year customer needs to pay a premium of certain amount to insurance provider company so that in case of unfortunate accident by the vehicle, the insurance provider company will provide a compensation (called ‘sum assured’) to the customer.
# 
# The data used in this project is publicly-available and acquired from Kaggle. The data was provided by an insurance company that has provided Health Insurance to its customers now they need help in building a best predictive model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company. This is a pure classification task which also includes the misclassification costs.
# 
# The main goal of this project is the implementation of various data mining and machine learning techniques and their applications. The exploration of multiple data analysis tasks on the targeted data, including both supervised knowledge discovery (predictive modeling) as well as unsupervised knowledge discovery for exploratory data analysis. 
# 
# Cross-selling is a frequently employed marketing strategy employed by a multitude of companies. The chief purpose of cross-selling is to generate a positive revenue flow (from existing customers) by selling a variety of product lines. Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimise its business model and revenue.
# 
# To accomplish this goal, some of the important numerical and categorical features are going to be analyzed in relation with the main parameter of interest “Response” (1 : Customer is interested, 0 : Customer is not interested). We are going to experiment and compare with various classifiers provided as part of the scikit-learn machine learning module, as well as with some of its parameters optimization and model evaluation capabilities. In particular, we are going to use the following list of classification algorithms to classify the data: (Naive Bayes (Gaussian), Decision Tree, Linear Discriminant Analysis (LDA), Rocchio, Random Forest, Ada Boost, Gradient Boosting, Logistic Regression). The accuracy of the different models will be compared to select the final model for prediction. In the end, we  will outline our future recommendations and conclusion.
# 
# 

# # 3. Exploratory Data Analysis (EDA)

# ## 3.1 Data Description:

# **id** 
# Unique ID for the customer
# 
# **Gender** 
# Gender of the customer
# 
# **Age** 
# Age of the customer
# 
# **Driving_License** 
# - 0: Customer does not have DL
# - 1: Customer already has DL
# 
# **Region_Code** 
# Unique code for the region of the customer
# 
# **Previously_Insured** 
# - 1: Customer already has Vehicle Insurance
# - 0: Customer doesn't have Vehicle Insurance
# 
# **Vehicle_Age** 
# Age of the Vehicle
# 
# **Vehicle_Damage** 
# - 1: Customer got his/her vehicle damaged in the past. 
# - 0: Customer didn't get his/her vehicle damaged in the past.
# 
# **Annual_Premium** 
# The amount customer needs to pay as premium in the year
# 
# **Policy_Sales_Channel** 
# Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.
# 
# **Vintage** 
# Number of Days, Customer has been associated with the company
# 
# **Response** 
# - 1: Customer is interested
# - 0: Customer is not interested

# The observations present within this dataset include typical consumer-based demographics such as (customer) id, Gender, and Age, as well as atypical domain-specific information such as Driving_License, Region_Code, Previously_Insured, Vehicle_Age, Vehicle_Damage, Annual_Premium, PolicySalesChannel, Vintage, and Response.

# ## 3.2 Import Packages:

# In[2]:


# Your packages imports here
from numpy import *
import numpy as np                                          # Numpy package for carrying out efficient computations
from numpy import linalg as la
import pylab as pl                                          
import pandas as pd                                         # Pandas package for reading and writing spreadsheets
import pdb
import matplotlib.pyplot as plt                             # Matplotlib package for displaying plots
import seaborn as sns                                       # Seaborn package for visualization of data

from sklearn.model_selection import train_test_split        # for splitting data into train and test sets
from sklearn import preprocessing                           # for preprocessing and standardize data

from sklearn import neighbors                               # KNN Classifier
from sklearn.neighbors import KNeighborsClassifier          # KNeighborsClassifier
from sklearn import tree                                    # Decision tree Classifier
from sklearn.tree import DecisionTreeClassifier, plot_tree  # DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # Linear Discriminant Analysis (LDA)
from sklearn import naive_bayes                             # Naive bayes classifier
from sklearn.neighbors import NearestCentroid               # Rocchio Classifier
from sklearn.ensemble import RandomForestClassifier         # Random Forest Classifier
from sklearn.ensemble import AdaBoostClassifier             # Ada Boost Classifier
from sklearn.ensemble import GradientBoostingClassifier     # Gradient Boosting Classifier
from sklearn.ensemble import BaggingClassifier              # Bagging Classifier
from sklearn.svm import SVC                                 # SVC Classifier (Support Vector Machines)
from sklearn.linear_model import LogisticRegression         # Logistic Regression Classifier

from sklearn import metrics                                 # Measure model performance
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve, recall_score, balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report           # for classification report
from sklearn.metrics import confusion_matrix                # for confustion matrix

from sklearn.model_selection import cross_val_score, KFold  # k-fold cross validation

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from scipy.stats import sem

from sklearn.model_selection import GridSearchCV            # To optimize model parameters
from sklearn.model_selection import RandomizedSearchCV      # To optimize model parameters


# In[3]:


import sklearn
sklearn.__version__


# In[4]:


pd.set_option('display.max_columns', 100)


# In[5]:


import warnings
warnings.filterwarnings('ignore')                           # ignore displaying the warnings


# In[6]:


get_ipython().run_line_magic('pylab', 'inline')


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## 3.3 Read Data File
# The data used in this project is publicly-available and acquired from Kaggle and can be found at https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction. The data was provided by an insurance company that has provided Health Insurance to its customers. It contains information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), Policy (Premium, sourcing channel) etc. The data consists of two sets, (i) train and (ii) test, where the target variable ‘Response’ was intentionally removed from the test set by the company.

# ### 3.3.1 Train Data
# For the train dataset there are 12 columns and 381109 rows.

# In[8]:


train = pd.read_csv("train.csv")
                                       
print('Rows,', 'Columns')
print(train.shape)              # print number of rows and columns in data
train.head()                    # see the first five rows in data


# ### 3.3.2 Test Data
# For the test dataset there are 11 columns and 127037 rows.

# In[9]:


test = pd.read_csv("test.csv")
                                       
print('Rows,', 'Columns')
print(test.shape)               # print number of rows and columns in data
test.head()                     # see the first five rows in data


# In[10]:


# Set id column as the index

train = train.set_index("id")
test = test.set_index("id")


# ## 3.4 Data Cleaning and Preprocessing
# There are several data cleaning, data exploration and preprocessing steps will be performed as part of the project.

# ### 3.4.1 General information about the data and Examine and handle the missing data

# In[11]:


# General info about the train data
train.info()


# In[12]:


# General info about the test data
test.info()


# In[13]:


# display general information about the train data set
# good way to check if any features have null values and the levels of categorical variables
pd.DataFrame({'Dtype':train.dtypes,
              'Levels':[train[x].unique() for x in train.columns],
              'Null_Count':train.isna().sum(),
              'Number_Unique_Values':train.nunique()
             })


# In[14]:


# display general information about the test data set
# good way to check if any features have null values and the levels of categorical variables
pd.DataFrame({'Dtype':test.dtypes,
              'Levels':[test[x].unique() for x in test.columns],
              'Null_Count':test.isna().sum(),
              'Number_Unique_Values':test.nunique()
             })


# After loading the dataset into a python Dataframe, our initial step was to take a comprehensive scope of the data, examining qualities that include variable name, type, and count, as well as the presence of missing values, unique values, and the categorical levels for each feature. Among the variables, it was found that there were four (4) **categorical ordinal** variables: id, Region_Code, Vehicle_Age, and Policy_Sales_Channel; five (5) **categorical
# binary** variables: Gender, Driving_License, Previously_Insured, Vehicle_Damage, and Response; two (2) **numeric discrete** variables: Age and Vintage; and one (1) **numeric continuous** variable: Annual_Premium. Specific definitions of each variable, regardless of type, are provided in part 3 (above).
# - The dataset is sufficiently large and complex. 
# - The dataset contains both numeric and categorical columns.
# - There are also no missing values in any of the variables as we can see that the Null_Count column shows 0.
# - There are total 381109 observations in training data. 
# - There are total 127037 observations in test data.
# - This is a 67% and 33% split from the original data set.
# - There are total 11 variables in the training data set.
# - There are total 10 variables in the test data set.
# - Test data is purposly missing target variable 'Response' because this is what we need to predict.

# ### 3.4.2 Plot missing variables list

# In[15]:


# conda install -c conda-forge/label/gcc7 missingno


# In[16]:


# train
import missingno as msno
msno.matrix(train)
plt.show()


# - This plot also displays that there are no missing values in any of the variables.

# In[17]:


# test
import missingno as msno
msno.matrix(test)
plt.show()


# ### 3.4.3 Convert Some Variables Types
# In many situations, we need to convert variables from one type into another. Type conversion is a method of changing features from one data type to another.

# In[18]:


# convert these variables from float to int64
float_list = ['Region_Code', 'Annual_Premium', 'Policy_Sales_Channel']
train[float_list] = train[float_list].astype(int64)
test[float_list] = test[float_list].astype(int64)


# ### 3.4.4 Data Transformation: 

# - Gender: Male: 1 | Female: 0
# - Vehicle_Damage: Yes: 1 | No: 0
# - Vehicle_Age: > 2 Years: 3 | 1-2 Year: 2 | < 1 Year: 1

# In[19]:


# Transforming string data into numeric data on train dataset
train["Gender"] = train["Gender"].map(lambda s: 1 if (s == "Male") else 0)
train["Vehicle_Damage"] = train["Vehicle_Damage"].map(lambda s: 1 if (s == "Yes") else 0)
train["Vehicle_Age"] = train["Vehicle_Age"].map(lambda s: 3 if (s == "> 2 Years") else (2 if (s == "1-2 Year") else 1))

# Transforming string data into numeric data on test dataset
test["Gender"] = test["Gender"].map(lambda s: 1 if (s == "Male") else 0)
test["Vehicle_Damage"] = test["Vehicle_Damage"].map(lambda s: 1 if (s == "Yes") else 0)
test["Vehicle_Age"] = test["Vehicle_Age"].map(lambda s: 3 if (s == "> 2 Years") else (2 if (s == "1-2 Year") else 1))


# ### 3.4.5 General Information on the Cleaned dataset:

# In[20]:


# this shows the number of features in the data
print(train.shape)
print(test.shape)


# In[21]:


# display general information about the cleaned train data set
# good way to check if any features have null values and the levels of categorical variables
pd.DataFrame({'Dtype':train.dtypes,
              'Levels':[train[x].unique() for x in train.columns],
              'Null_Count':train.isna().sum(),
              'Number_Unique_Values':train.nunique()
             })


# In[22]:


# display general information about the cleaned test data set
# good way to check if any features have null values and the levels of categorical variables
pd.DataFrame({'Dtype':test.dtypes,
              'Levels':[test[x].unique() for x in test.columns],
              'Null_Count':test.isna().sum(),
              'Number_Unique_Values':test.nunique()
             })


# ### 3.4.6 Basic Statistics of Features:
# Let's explore the general characteristics of the data as a whole: examine the means, standard deviations, and other statistics associated with the numerical attributes.

# In[23]:


# summary of the distribution of continuous variables
# provides basic statistics for numerical variables
train.describe(include="all").T


# Above is the display of basic statistics of features. This lets us explore the general characteristics of the data as a whole: examine the means, standard deviations, and other statistics associated with the numerical attributes.
# - There seems to be some variables which have larger values compared to other variables so we might need to rescale the numeric features. We can use scikit learn preprocessing package for Min-Max Normalization to transform the values of all numeric attributes in the table onto the range 0.0-1.0. Then we fit the MinMaxScaler on the training data first and then transform the training and test data using this scaler.
# - Variable "Annual_Premium" has the largest value of 540165.

# In[24]:


# summary of the distribution of continuous variables
# provides basic statistics for numerical variables
test.describe(include="all").T


# ## 3.5 Exploration / Visualization of Data

# 
# ### 3.5.1 Distribution of all categorical features in the data
# Show the distributions of values associated with categorical attributes using SEABORN package and/or plotting capabilities of Pandas to generate bar charts showing the distribution of categories for each attribute).

# In[25]:


# extract names of the categorical features in the data
categorical = ['Gender', 'Driving_License', 'Previously_Insured', 
               'Vehicle_Age', 'Vehicle_Damage', 'Response']


# In[26]:


fig, ax = plt.subplots(2, 3, figsize=(20, 10))
fig.subplots_adjust(hspace=0.3, wspace=0.4)

for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(train[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# From the plot of each categorical feature, we can observe that there are some features that distribute evenly, yet, several columns are skewed data as well.
# 
# *   In this dataset, Gender, Previously Insured and Vehicle Damage columns are evenly distributed.
# *   The Driving License column, as we see the percentage of each value below, it shows skew severely. Among this column, 99.8% of the rows are 1, which means 99.8% of the cases have driving license.

# In[27]:


train['Driving_License'].value_counts(normalize = True)


# *   As we look at the Vehicle Age column, only 4.2% of cases' that the vehicle age are above 2 years.

# In[28]:


train['Vehicle_Age'].value_counts(normalize = True)


# *   The predict value Response is also skew variables. There are 87.7% of the rows that shows 0 in this feature, which means the response of the customers are not interested in the insurance.

# In[29]:


train['Response'].value_counts(normalize = True)


# ### 3.5.2 Cross Tabulated View using Barplots 
# #### Comparing categorical attributes with respect to target attribute (Response)
# 
# Perform a cross-tabulation of all categorical attributes with the target attribute (Response). This requires the aggregation of the occurrences of each Response value (0 or 1) separately for each value of the other attributes. Here, we are using SEABORN visualization package to create bar charts graphs to visualize of the relationships between these sets of variables.

# In[30]:


fig, ax = plt.subplots(2, 3, figsize=(20, 10))
fig.subplots_adjust(hspace=0.3, wspace=0.4)

for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(train['Response'], hue=variable, data=train, ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# Following are some of the key points observed from these visualizations:
# 

# 
# 
# *   Gender v.s. Response: Both genders have even distribution on the response.
# *   Driving License v.s. Response: Even though the driving license is skewed data, customers who have driving license's distribution on response is even. However, as we see the percentage in the data frame below, for customers who do not have driver license their response are not interested in the insurance.

# In[31]:


rd = train.groupby(['Response', 'Driving_License'])
result = rd.ngroup().value_counts(normalize=True,sort=False)
result.index = rd.groups.keys()

rd.size().to_frame('Size').assign(Percentage=result.mul(100).round(2)).reset_index()


# 
# *   Previously Insured v.s. Response: For customers who have insurance previously, they usually are not interested in the insurance. In this case, customers who repond interested in insurance usually do not have insurance previously.
# *   Vehicle Age v.s. Response: In this plot, we can observe that people who are interested in insurance have a large percentage of their cars that are about 1-2 years old.
# *   Vehicle Damage v.s. Response: The relationship between whether customers have vehicle damager or not and the response. It shows that most customers who are interested in insurance have vehicle damage before.
# 
# 

# ### 3.5.3 Correlations Analysis
# Lets perform basic correlation analysis among the attributes. The following Complete Correlation Matrix shows any significant positive or negative correlations among pairs of attributes. 
# 
# - A correlation matrix is a table showing correlation coefficients between sets of variables.
# - Correlation analysis is a very important step in pre-processing because it helps in identifying multicollinearity and building components in Principle Components Analysis (PCA).

# In[32]:


# Compute the correlation matrix
corr = train.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Following are the few of the variables that seem Strong Positively Correlated:
# - Vehicle_Age and Age
# - Vehicle_Damage and Age
# - Vehicle_Damage and Vehicle_Age
# - Vehicle_Damage and Response

# Following are few of the variables that seems to have moderate negative correlation:
# - Vehicle_Damage and Previously_Insured
# - Policy_Sales_Channel and Age
# - Policy_Sales_Channel and Vehicle_Age

# ### 3.5.4 Proportion of Target Variable (Response) Classes

# In[33]:


x = train['Response']
plt.pie(x.value_counts(), labels = x.unique(),explode = [0,0.2],
autopct = lambda x: str(round(x, 2)) + '%')
plt.title(x.name)
plt.legend(loc='upper right')
plt.show()


# The proportion of Response classes in train data set:
# - 1: Customer is interested = 87.74% 
# - 0: Customer is not interested = 12.26%

# A challenge we will face is an imbalance dataset. There is more of class 1 then there is class 2. So there will be more pre-processing required in order to prepare our dataset for modeling.
# 
# Some things we may have to do include:
# 
# - Undersampling the majority class and oversampling the minority class in order to rebalance our dataset. Deleting some of the majority class in undersampling can remove some valuable information. 
# 
# - Another challenge is to choose a metric to consider the imbalance dataset. For example, we would not be able to choose accuracy as that is not a good measure when dealing with an imbalance dataset.

# ### 3.5.5 Pairplot
# 
# Let's visualize the attributes using pairplot function from SEABORN package:
# 
# A pairplot plots a pairwise relationships in a dataset which shows the Correlation between numerical attributes and it also displays the distributions of all attributes. 

# In[34]:


features_pair = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel']


# In[35]:


sns.pairplot(train[features_pair], corner=True)


# ### 3.5.6 Violin Plots:

# Let's see some of the strong correlated variables' relationship by using the seaborn package:

# *  In the correlation plot we can see variables Age and previously Insured shows strong positive correlation. By using the violin plot we can see the probability density of age.

# In[36]:


sns.catplot(x="Previously_Insured", y="Age", kind = "violin", hue = "Response",
            palette="Blues", data=train, split = True)


# *  Vehicle Damage and Age also shows strong positively correlated in the correlation plot. Therefore, violin plot was used to see the probability density of age by whether the customer damaged their vechile before.

# In[37]:


sns.catplot(x="Vehicle_Damage", y="Age", kind = "violin", 
            hue = "Response",palette="Blues", data=train, split = True)


# *  In the correlation plot, the variables Age and Policy Sales Channel present moderate negatively correlated. Hence, the kde plot is used to see the distribution of these two variables.

# In[38]:


sns.catplot(x="Vehicle_Age", y="Policy_Sales_Channel", kind = "violin",
            hue = "Response",palette="Blues", data=train, split = True)


# # 4. Research Questions

# ## 4.1 Research Question 1: 
# What other variables or indicators that can lead to an increased insurance premium?
# 
# - **Possible analysis**: First we wil do some EDA. Then will will bucket the premiums into four groups ranging from low, low-med, med-high, and high. Next we will build a Random Forest Model in order to examine variable importance.

# ### 4.1.1 Handling outliers and Skeweness:

# Some algorithms works well with normally distributed data. Therefore, we must remove skewness of variable(s). There are methods like log, square root or inverse of the values to remove skewness.

# In[39]:


#pip install -U seaborn


# In[40]:


df = train.copy()
age_viz = ['Annual_Premium']


# In[41]:


plt.figure(figsize = (20.7, 10))

for i in range(0, len(age_viz)):
    plt.subplot(2, 1, i + 1)
    sns.histplot(
        x = df[age_viz[i]],
        kde = True
    )

plt.tight_layout()


# In[42]:


# skewness along the index axis 
train[features_pair].skew(axis = 0, skipna = True)


# In[43]:


plt.hist(train['Annual_Premium'], bins=25)
plt.title("Histogram - Annual_Premium", fontsize=15)
plt.show()


# - The annual premium data is skewed and needs to have outliers removed before having a more normal distribution.

# In[44]:


plt.figure(figsize = (20.7, 10))


plt.subplot(2, 1, 2)
sns.boxplot(
    x = df['Annual_Premium']
)

plt.tight_layout()


# - As we can see there are alot of outliers in the annual_premium that skews the data

# In[45]:


# We will remove outliers base on quartiles 
Q1 = df['Annual_Premium'].quantile(0.25)
Q3 = df['Annual_Premium'].quantile(0.75)
IQR = Q3 - Q1
low_limit = Q1 - (1.5 * IQR)
high_limit = Q3 + (1.5 * IQR)

print(low_limit)
print(high_limit)


# In[46]:


filtered_entries = ((df['Annual_Premium'] >= low_limit) & (df['Annual_Premium'] <= high_limit))
df = df[filtered_entries].reset_index(drop = True)


# In[47]:


before_remove = df.shape[0]
before_remove 
after_remove = df.shape[0]
after_remove


# In[48]:


print('Outlier removed:', 381109-370789)


# In[49]:


plt.figure(figsize = (20.7, 10))


plt.subplot(2, 1, 2)
sns.boxplot(
    x = df['Annual_Premium']
)

plt.tight_layout()


# In[50]:


df['Annual_Premium'].value_counts().head()


# In[51]:


df['Annual_Premium'].plot.hist()


# #### Our next step will be breaking the groups into low,low-med,med-high, and high annual premium group.

# In[52]:


low = df['Annual_Premium'].quantile(0.33)
med = df['Annual_Premium'].quantile(0.66)
high =df['Annual_Premium'].quantile(0.99)

print(low)
print(med)
print(high)


# In[53]:


def Annual_Premium_code(x):
    if x['Annual_Premium'] < 27169:
        Annual_Premium = 'low'
    elif x['Annual_Premium'] < 36212:
        Annual_Premium = 'low-med'
    elif x['Annual_Premium'] < 72963:
        Annual_Premium = 'med-high'
    else:
        Annual_Premium = 'high'
    return Annual_Premium


# In[54]:


df['Annual_Premium_Group'] = df.apply(lambda x: Annual_Premium_code(x), axis = 1)


# In[55]:


df.head()


# In[56]:


df['Annual_Premium_Group'].value_counts()


# - The rows were successfully segregated into the different premium groups. 

# In[57]:


# WE will now drop Annual_Premium in order to run our models and not worry about multicollinarity
newdf = df.drop(columns=["Annual_Premium"])


# In[58]:


# Now we will have to change our categorical variables into numbers
def Annual_Premium_code(x):
    if x['Annual_Premium'] == 'low':
        Annual_Premium = 1
    elif x['Annual_Premium'] == 'low-med':
        Annual_Premium = 2
    elif x['Annual_Premium'] == 'med-high':
        Annual_Premium = 3
    else:
        Annual_Premium = 4
    return Annual_Premium


# In[59]:


df['Annual_Premium_Group'] = df.apply(lambda x: Annual_Premium_code(x), axis = 1)


# In[60]:


df.head()


# In[61]:


#Now we are ready to run a random forest to see if we can predict the Annual Premium Group

X = df.drop(['Annual_Premium_Group'], axis=1)
y = df['Annual_Premium_Group']


# In[62]:


X.shape


# In[63]:


y.shape


# In[64]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=555)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=555)


# In[65]:


from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100)
reg.fit(X_train, y_train)
train_pred = reg.predict(X_train)
val_pred = reg.predict(X_val)


# In[66]:


print(reg.feature_importances_)
#plot graph of feature importances for better visualization
feat_importances = pd.Series(reg.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# **Conclusion**: 
# From the feature importance and correlation maps. There seems to be no other strongly correlated features that could predict annual premium.

# ## 4.2 Research Question 2: 
# Is vehicle damage related to other attributes such as region, vehicle age, and gender?
# 
# - **Possible analysis**: By checking the plot or regression individually between the vehicle damage attribute and other variables.

# In[67]:


df.corr()


# Positively Correlated:
# Vehicle_Damage and Age,
# Vehicle_Damage and Vehicle_Age,
# Vehicle_Damage and Response,
# Negative Correlated:
# Vehicle_Damage and Insurance
# 

# Next I will run a linear regression

# In[68]:


X = df.drop(['Vehicle_Damage'], axis=1)
y = df['Vehicle_Damage']


# In[69]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[70]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[71]:


reg = LinearRegression().fit(X, y)
reg.fit(X_train, y_train)


# In[72]:


#creating the predictions for training and val set
train_pred = reg.predict(X_train)
val_pred = reg.predict(X_val)


# In[73]:


#Finding the R2 of training data and val data
print("The R2 of the training data is",r2_score(y_train, train_pred))
print("The R2 of the validation data is",r2_score(y_val, val_pred))


# In[74]:


#Finding the RMSE of training and validation set
print("The RMSE of the training data is",mean_squared_error(y_train, train_pred))
print("The RMSE of the validation data is",mean_squared_error(y_val, val_pred))


# **Findings**: Vehicle damage goes up with the age of the vehicle, age of the person, and whether they are more likely to respond yes to health insurance. It also seems like those who are previously insured were less likely to have vehicle damage. All these variables are interesting finds and we know that people who are older and have older cars are more at risk of having vehicle damage. Also those who are interested in the cross insurance coverage were more likely to have damaged vehicle. While those who were previously insured are a less risky group in terms of damaging their cars.

# ## 4.3 Research Question 3: 
# Can we predict customer loyalty to the company?
# 
# - **Possible analysis**: Looking at how long customers have been with the company by using the vintage variable, can we make up the demographic profile such as gender, age, or vehicle age. This is something important for companies to predict churn and what group of people are best to target for loyal paying customers.

# In[75]:


df = train.copy()
age_viz = ['Vintage']


# In[76]:


plt.figure(figsize = (20.7, 10))

for i in range(0, len(age_viz)):
    plt.subplot(2, 1, i + 1)
    sns.histplot(
        x = df[age_viz[i]],
        kde = True
    )

plt.tight_layout()


# In[77]:


df['Vintage'].value_counts().head()


# In[78]:


df['Vintage'].describe()


# In[79]:


def vintage_month(x):
    to_month = int(round(x['Vintage'] / 30, 0))
    return to_month


# In[80]:


# convert into months for easy understanding
df['Vintage_Month'] = df.apply(lambda x: vintage_month(x), axis = 1)
df.head()


# In[81]:


df['Vintage_Month'].value_counts()


# **Findings about customer loyalty:**
# - Most common length of relationship span from 2,4,6,8 months as the most common. 
# - While 1,3,5,9,7, less common. With super loyal customers at 10 months being the least commmon of all. 
# - An intersting insight is that there are 40,920 customers that has been with them for 8 months but only 19507 customers that stay for 10 months. 

# ## 4.4 Research Question 4: 
# What type of people have insurance and who are the people who don’t have insurance?
# 
# - **Possible analysis**: In order to examine the difference between those who have insurance and those who don't have insurance. We will do EDA and run a machine learning algorithm such as a decision tree to predict those who have insurance and those who don’t have insurance.

# I will run a correlation chart

# In[82]:


df.corr()


# First I will run a decision tree model to find if it possible to predict those who have insurance from those who dont.

# In[83]:


X = df.drop(['Previously_Insured'], axis=1)
y = df['Previously_Insured']


# In[84]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)


# In[85]:



from sklearn import tree
dt = tree.DecisionTreeClassifier(criterion='gini')
dt = dt.fit(X_train, y_train)


# In[86]:


#Used to measure decision tree performance
from sklearn import metrics

def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    y_pred = clf.predict(X)   
    if show_accuracy:
         print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")
    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y, y_pred),"\n")
      
    if show_confussion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y, y_pred),"\n")


# In[87]:



from sklearn import metrics
measure_performance(X_test, y_test, dt, show_confussion_matrix=True, show_classification_report=False)


# In[88]:


yesinsurancedf = df[df.Previously_Insured != 0]
noinsureancedf=df[df.Previously_Insured != 1]


# In[89]:


yesinsurancedf.describe().T


# In[90]:


yesinsurancedf.shape


# In[91]:


noinsureancedf.describe().T


# In[92]:


noinsureancedf.shape


# From this dataset 206,481 people were not previously insured. 174,628 people were previously insured

# In[93]:


df = yesinsurancedf
age18_25 = df.Age[(df.Age <= 25) & (df.Age >= 18)]
age26_35 = df.Age[(df.Age <= 35) & (df.Age >= 26)]
age36_45 = df.Age[(df.Age <= 45) & (df.Age >= 36)]
age46_55 = df.Age[(df.Age <= 55) & (df.Age >= 46)]
age55above = df.Age[df.Age >= 56]

x = ["18-25","26-35","36-45","46-55","55+"]
y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=x, y=y, palette="rocket")
plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()


# In[94]:


df = noinsureancedf


# In[95]:


age18_25 = df.Age[(df.Age <= 25) & (df.Age >= 18)]
age26_35 = df.Age[(df.Age <= 35) & (df.Age >= 26)]
age36_45 = df.Age[(df.Age <= 45) & (df.Age >= 36)]
age46_55 = df.Age[(df.Age <= 55) & (df.Age >= 46)]
age55above = df.Age[df.Age >= 56]

x = ["18-25","26-35","36-45","46-55","55+"]
y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=x, y=y, palette="rocket")
plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()


# Some findings are that those who were not previously insured had a higher mean age, higher mean vehicle age, and reported higher mean vehicle damage. Also from the correlation plot there was a negative correlation between those who were previously insured and vehicle damage, suggesting those who have insurance are less likely to have damage cars.

# # 5. Machine Learning

# Cross-selling is a frequently employed marketing strategy employed by a multitude of companies. The chief purpose of cross-selling is to generate a positive revenue flow (from existing customers) by selling a variety of product lines. 
# 
# Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimise its business model and revenue.
# 
# To accomplish this goal, some of the important numerical and categorical features are going to be analyzed in relation with the main parameter of interest “Response” (1 : Customer is interested, 0 : Customer is not interested). 
# 
# We are going to experiment and compare with various classifiers provided as part of the scikit-learn machine learning module, as well as with some of its parameters optimization and model evaluation capabilities. In particular, we are going to use the following list of classification algorithms to classify the data:
# 
# - Naive Bayes (Gaussian) classifier
# - Decision Tree Classifier
# - Linear Discriminant Analysis (LDA) Classifier
# - Rocchio Classifier
# - Random Forest Classifier
# - Ada Boost Classifier
# - Gradient Boosting Classifier
# - Logistic Regression Classifier
# 
# 
# Using RandomizedSearchCV from Scikit-learn, evaluate predictive models on 3-fold cross validation. We will experiment with RandomizedSearchCV for parameters optimization to see if we can improve accuracy (we will not provide the details of all of our experimentation, but will provide a short discussion on what parameters worked best as well as our final results).
# 
# 
# We are using RandomizedSearchCV because it is a lot faster compared to GridSearchCV and allows us to explore the parameter space more systematically and lets us select the best tuning parameters (aka "hyperparameters"). RandomizedSearchCV allows us to define a grid of parameters that are searched using K-fold cross-validation.
# 
# 
# Finally, the accuracy of the different models will be compared to select the final model for prediction and the overview of model performances will be presented in a tabulated form.

# ## 5.1 Prepare Data for Machine Learning

# ### 5.1.1 Convert / Transform Categorical Variables
# 
# Let's convert the selected variables into the Standard Spreadsheet format (i.e., convert categorical attributes into numeric by creating dummy variables).
# 
# 
# This requires converting each categorical attribute into multiple binary ("dummy") attributes (one for each values of the categorical attribute) and assigning binary values corresponding to the presence or not presence of the attribute value in the original record). The numeric attributes should remain unchanged.
# 
# 
# > The same transformation needs to be aplied on every dataset. 
# 
# > The easiest way would be to merge train, val, and test sets, and to split after the transformation.

# In[96]:


# list of cat features to transform
cat_feats = ['Gender', 'Driving_License', 'Previously_Insured', 
             'Vehicle_Age', 'Vehicle_Damage']


# #### Merge Train and Test Data Sets

# In[97]:


# make copies of original data sets
train_copy = train.copy()
test_copy = test.copy()


# In[98]:


# concate
df = pd.concat([train_copy.assign(ind="train"), test_copy.assign(ind="test")])


# **Now you can use `pd.get_dummies(df,columns=cat_feats)` to create a fixed larger dataframe that has new feature columns with dummy variables.**

# In[99]:


df = pd.get_dummies(df, columns=cat_feats)


# #### Split both Data sets train and test after Transformation

# In[100]:


test, train = df[df["ind"].eq("test")], df[df["ind"].eq("train")]


# In[101]:


# drop the column 'Response' and ind' from test set
test.drop(['Response', 'ind'], axis=1, inplace=True)


# In[102]:


# drop the column 'ind' from train set
train.drop(['ind'], axis=1, inplace=True)


# In[103]:


# convert Response back to int64 from float64
train['Response'] = train['Response'].astype(int64)


# In[104]:


train.head()


# In[105]:


test.head()


# ### 5.1.2 Let's separate the target attribute ("Response") and the attributes used for model training
# The next step after cleaning and exploration of variables is to separate the target attribute ("Response") and the attributes used for model training. Let's create a separate data frame which contains the records without target attribute. Then pull the target attribute and store it separately.

# In[106]:


y = train.Response
X = train.drop(['Response'], axis=1)


# In[107]:


y.head()


# In[108]:


X.head()


# In[ ]:





# ### 5.1.3 Split the data into train and validation sets
# 
# - Use sklearn's tran_test_split() module of "sklearn.cross_validation" function to create the 80%-20% stratified split). Note that the same split should also be performed on the target attribute).
# - Create a 20%-80% stratified split of the data. Set aside the 20% validation portion; the 80% training data partition will be used for model building, cross-validation etc.
# - Use random_state = 22 to create consistent and repeatable train-test splits.

# In[109]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.20, random_state=22, shuffle=True)


# ### 5.1.4 Proportion of Target Variable Classes

# In[110]:


# Creating temporary variables to see the proportion of target variable as Percentage wise in bar plot
temp_y_train = y_train.value_counts()/y_train.count()
temp_y_val = y_val.value_counts()/y_val.count()


# In[111]:


# Using Barplot to Compare different categories of target variable
fig1 = plt.figure(figsize=(15,4))                           # size of the figure

# train
sub1 = fig1.add_subplot(121)                                # 1 row, 2 columns, 1st position
sub1.set_xlabel('Response', size=15)                       # label for x-axis
sub1.set_ylabel('Percentage', size=15)                      # label for y-axis
sub1.set_title("Response Variable Distribution - Train", 
               size=15)                                     # title of the plot
temp_y_train.plot(kind='bar', grid = True);                 # using temp_y_train variable for plotting

# test
sub1 = fig1.add_subplot(122)                                # 1 row, 2 columns, 2nd position
sub1.set_xlabel('Response', size=15)                       # label for x-axis
sub1.set_ylabel('Percentage', size=15)                      # label for y-axis
sub1.set_title("Response Variable Distribution - Validation", 
               size=15)                                     # title of the plot
temp_y_val.plot(kind='bar', grid = True);                  # using temp_y_test variable for plotting


# In[112]:


print(temp_y_train)
print(temp_y_val)


# The proportion of Response classes in train and validation sets after splitting are approximately:
# 
# - 0: Customer is not interested = 88%
# - 1: Customer is interested = 12%

# **Importance of Stratified Sampling**: In statistics, stratified sampling is a method of sampling from a population which can be partitioned into subpopulations. Stratified sampling offers several advantages over simple random sampling.
# - A stratified sample can provide greater precision than a simple random sample of the same size.
# - Because it provides greater precision, a stratified sample often requires a smaller sample, which saves money.

# ### 5.1.5 Standardization / Normalization of Numerical Data:
# 
# We are performing min-max normalization to rescale numeric features. Let's use scikit learn preprocessing package for Min-Max Normalization to transform the values of all numeric attributes in the table onto the range 0.0-1.0. Fit the MinMaxScaler on the training data first and then transform the train, validation, and test data using this scaler.
# 
# While transforming the numerical features, it is best to avoid normalizing dummy variables. Even if the numerical values do not change, the underlying data type will be changed to float. Once converted to float, the variables are no longer treated as mutually exclusive binary values. It is not detrimental to KNN, but it can make the models lose the benefit of using one-hot encoding for many other ML algorithms (especially neural networks).

# In[113]:


numeric_columns = ['Age', 'Annual_Premium', 'Vintage']

# fit the MinMaxScaler on the training data
min_max_scaler = preprocessing.MinMaxScaler().fit(X_train[numeric_columns])

X_train[numeric_columns] = min_max_scaler.transform(X_train[numeric_columns]) # transform the training data using above scaler
X_val[numeric_columns] = min_max_scaler.transform(X_val[numeric_columns])    # transform the val data using above scaler
test[numeric_columns] = min_max_scaler.transform(test[numeric_columns])    # transform the test data using above scaler


# In[114]:


# Let's look at the normalized train data
np.set_printoptions(precision=2, linewidth=80, suppress=True)
print(X_train.shape)
print(y_train.shape)
X_train[numeric_columns]


# In[115]:


# Let's look at the normalized validation data
np.set_printoptions(precision=2, linewidth=80, suppress=True)
print(X_val.shape)
print(y_val.shape)
X_val[numeric_columns]


# In[116]:


# Let's look at the normalized test data
np.set_printoptions(precision=2, linewidth=80, suppress=True)
print(test.shape)
test[numeric_columns]


# ## 5.2 Predictive Modeling and Model Evaluation Using Classification Techniques 

# In this section, we are going to perform following tasks for all of the classifiers mentioned above:
# 
# - Initiate the classifier from the sklearn library 
# - Fit the model
# - Make predictions
# - Calculate training and validation **ROC_AUC** score of the model
# - Create and visualize confusion matrix
# - Using classification_report method in `sklearn.metrics` get the following metrics on the validation set:
#      - Recall (Sensitivity)
#      - Specificity
#      - Precision
#      - Balanced Accuracy
#      - F1 Score
# - Using RandomizedSearchCV, perform cross-validation and hyper-parameter tunning:
#     - Define the lists of parameters to tune and apply cross validation to find the best value.  
#     - Perform RandomizedSearchCV where you check for combinations of these hyper-parameters.
#         - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# - Compare the performance of all models.
# - Choose the best model out of all models based on performance:
#     - Using the classification_report method in `sklearn.metrics` get the following metrics on the testing set:
#          - Recall (Sensitivity)
#          - Specificity
#          - Precision
#          - Balanced Accuracy
#          - F1 Score
# - Choose a winner model.
#     - Make predictions on the hold-out test set whose outcome variables are not known (data we imported as: test).
#     - At the end it will output the prediction of the labels.
#     - Save the predictions in a csv file.
#     - There will be a single column in the csv file. The column header will be 'predictions'

# ### 5.2.1 Define Functions:

# In[117]:


# A dictionary to store all CLASSIFICATION models performance statistics:
dict_clf = {
    'Model':[], 
    'Train ROC_AUC':[], 
    'Test ROC_AUC':[],
    'Precision':[],
    'Recall':[],
    'F1 Score':[],
    'Support':[],
}


# #### A Versatile Function that:
# - Generates and fit the models
# - Predicts on train data
# - Predicts on validation / test data
# - Print Confusion Matrix
# - Print Classification Report
# - Print and stores performance statistics

# In[118]:


def fit_predict_score(name, clf, X_train, X_val, y_train, y_val):
    
    clf.fit(X_train, np.ravel(y_train))       # fit the model
    predicted_train = clf.predict(X_train)    # make predictions on train data
    predicted_test = clf.predict(X_val)       # make predictions on test data

    # Confusion Matrix
    confusion_matrix =  pd.crosstab(index=np.ravel(y_val), columns=predicted_test.ravel(), rownames=['Expected'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, square=False, fmt='', cbar=False)
    
    # ROC_AUC Scores
    score_train = np.round(metrics.roc_auc_score(y_train , predicted_train),3)
    score_test = np.round(metrics.roc_auc_score(y_val , predicted_test),3)
       
    # model performance statistics
    precision, recall, f1_score, support = (precision_recall_fscore_support(y_val, predicted_test, average='weighted'))
    
    plt.title(name + " " + str(score_test), fontsize = 15)
    plt.show()
    
    # Classification Report
    print (metrics.classification_report(y_val, predicted_test))  
    print()
    print("Train ROC_AUC: {0:.3f}".format(score_train))      
    print("Test ROC_AUC: {0:.3f}".format(score_test),"\n")     
    
    dict_clf["Model"].append(name)
    dict_clf["Train ROC_AUC"].append(score_train)
    dict_clf["Test ROC_AUC"].append(score_test)
    dict_clf["Precision"].append(precision)
    dict_clf["Recall"].append(recall)
    dict_clf["F1 Score"].append(f1_score)
    dict_clf["Support"].append(support)


# ### 5.2.2 Hyperparameters Tunning:
# 
# It is the problem of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a parameter whose value is used to control the learning process. 
# 
# One traditional and popular way to perform hyperparameter tuning is by using an Exhaustive Grid Search from Scikit learn. This method tries every possible combination of each set of hyper-parameters. Using this method, we can find the best set of values in the parameter search space. This usually uses more computational power and takes a long time to run since this method needs to try every combination in the grid size.

# #### A versatile function to perform RandomizedSearchCV on different models:
# We are going to use the RandomizedSearchCV to explore the parameter space more systematically. Select the best tuning parameters (aka "hyperparameters"). Randomized Grid Search allows us to define a grid of parameters that will be searched using K-fold cross-validation.
# 
# - Perform cross validation and hyper-parameter tuning using RandomizedSearchCV
# - Print wall time of the best model
# - Print best parameters
# - Print and store best cv score
# - Print and store best training score
# - Print and store best validation score

# #### Cross Validation:
# 
# Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. Use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.

# Scoring Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

# In[119]:


def grid_search_function(m, modelName, params, xtrain, ytrain, xtest, ytest, cv=3):
    'Performs RandomizedSearchCV and gets different statistics'
    
    gs = RandomizedSearchCV(m, params, scoring='roc_auc', cv=3, verbose=1, random_state=101, n_iter=10, n_jobs=-1)              
       
    get_ipython().run_line_magic('time', '_ = gs.fit(xtrain, ytrain)')
    print()
    print(gs.best_params_)                             
    print()
    
    fit_predict_score(modelName, gs, xtrain, xtest, ytrain, ytest)


# ### 5.2.3 Naive Bayes (Gaussian) Classifier - Base Model:
# The Naïve Bayes Classifier belongs to the family of probability classifier, using Bayesian theorem. The reason why it is called ‘Naïve’ because it requires rigid independence assumption between input variables. Therefore, it is more proper to call Simple Bayes or Independence Bayes. This algorithm has been studied extensively since 1960s. Simple though it is, Naïve Bayes Classifier remains one of popular methods to solve text categorization problem, the problem of judging documents as belonging to one category or the other, such as email spam detection.
# 
# Bayes’s theorem plays a critical role in probabilistic learning and classification.
# - Uses prior probability of each class given no information about an item
# - Classification produces a posterior probability distribution over the possible classes given a description of an item
# - The models are incremental in the sense that each training example can incrementally increase or decrease the probability that a hypothesis is correct. Prior knowledge can be combined with observed data.

# In[120]:


nbclf = naive_bayes.GaussianNB()
fit_predict_score("NaiveBayes BaseModel", nbclf, X_train, X_val, y_train, y_val)


# ### 5.2.4 Naive Bayes (Gaussian) Classifier - Parameters Tunning:

# In[121]:


nbclf = naive_bayes.GaussianNB()
print(nbclf.get_params())


# In[122]:


# define the parameter values that should be searched
parameters = {
    'priors': [None],
    'var_smoothing': [1e-09],
}

# perform grid search using grid_search_function
grid_search_function(nbclf, 'NaiveBayes RSCV', parameters, X_train, y_train, X_val, y_val, cv=3)


# ### 5.2.5 Decision Tree Classifier - BaseModel:
# 
# Decision Tree algorithm belongs to the family of supervised learning algorithms. In Decision Trees, for predicting a class label for a record we start from the root of the tree. We compare the values of the root attribute with the record’s attribute. On the basis of comparison, we follow the branch corresponding to that value and jump to the next node.

# In[123]:


dt = tree.DecisionTreeClassifier()
fit_predict_score("DecisionTree BaseModel", dt, X_train, X_val, y_train, y_val)


# ### 5.2.6 Decision Tree Classifier - Parameters Tunning:

# In[124]:


dt = tree.DecisionTreeClassifier()
print(dt.get_params())


# In[125]:


# define the parameter values that should be searched
parameters = {
    'criterion': ['gini', 'entropy'],
    'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
    'max_depth': np.linspace(1, 10, 10, endpoint=True), 
    'min_samples_leaf': range(1,6),
}

# perform grid search using grid_search_function
grid_search_function(dt, 'DecisionTree RSCV', parameters, X_train, y_train, X_val, y_val, cv=3)


# ### 5.2.7 Linear Discriminant Analysis (LDA) Classifier - Base Model:
# Linear discriminant analysis, normal discriminant analysis, or discriminant function analysis is a generalization of Fisher's linear discriminant, a method used in statistics and other fields, to find a linear combination of features that characterizes or separates two or more classes of objects or events. Wikipedia

# In[126]:


ldclf = LinearDiscriminantAnalysis()
fit_predict_score("LinearDiscriminant BaseModel", ldclf, X_train, X_val, y_train, y_val)


# ### 5.2.8 Linear Discriminant Analysis (LDA) Classifier - Parameters Tunning:

# In[127]:


ldclf = LinearDiscriminantAnalysis()
print(ldclf.get_params())


# In[128]:


# define the parameter values that should be searched
parameters = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': arange(0, 0.1, 0.001)    
}

grid_search_function(ldclf, 'LinearDiscriminant RSCV', parameters, X_train, y_train, X_val, y_val, cv=3)


# ### 5.2.9 Rocchio Classifier - BaseModel:
# The Rocchio algorithm is based on a method of relevance feedback found in information retrieval systems which stemmed from the SMART Information Retrieval System which was developed 1960-1964. Like many other retrieval systems, the Rocchio feedback approach was developed using the Vector Space Model. Wikipedia

# In[129]:


roclf = NearestCentroid()
fit_predict_score("Rocchio BaseModel", roclf, X_train, X_val, y_train, y_val)


# ### 5.2.10 Random Forest Classifier (an example of bagging) - BaseModel:
# - Each classifier in the ensemble is a decision tree classifier and is generated using a random selection of attributes at each node to determine the split.
# - During classification, each tree votes and the most popular class is returned.
# - Comparable in accuracy to Adaboost, but more robust to errors and outliers.
# - Insensitive to the number of attributes selected for consideration at each split, and faster than boosting.

# In[130]:


rf = RandomForestClassifier()
fit_predict_score("RandomForest BaseModel", rf, X_train, X_val, y_train, y_val)


# ### 5.2.11 Random Forest Classifier - Parameters Tunning:

# In[131]:


rf = RandomForestClassifier()
print(rf.get_params())


# In[132]:


# define the parameter values that should be searched
parameters = {
    'n_estimators': range(5, 101, 5),
    'criterion': ['gini', 'entropy'],
    'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True)
}

grid_search_function(rf, 'RandomForest RSCV', parameters, X_train, y_train, X_val, y_val, cv=3)


# ### 5.2.12 Ada Boost Classifier - BaseModel:
# 
# AdaBoost, short for “Adaptive Boosting”, is the first practical boosting algorithm proposed by Freund and Schapire in 1996. It focuses on classification problems and aims to convert a set of weak classifiers into a strong one.

# In[133]:


ab = AdaBoostClassifier()
fit_predict_score("AdaBoost BaseModel", ab, X_train, X_val, y_train, y_val)


# ### 5.2.13 Ada Boost Classifier - Parameters Tunning:

# In[134]:


ab = AdaBoostClassifier()
print(ab.get_params())


# In[135]:


# define the parameter values that should be searched
parameters = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 1.8, 2.0],
    'n_estimators': range(5, 51, 5),
}

grid_search_function(ab, 'AdaBoost RSCV', parameters, X_train, y_train, X_val, y_val, cv=3)


# ### 5.2.14 Gradient Boosting Classifier - BaseModel:
# The following is an example of using Gradient Boosted Decision Trees. GBDT is a generalization of boosting to arbitrary differentiable loss functions. GBDT is an accurate and effective procedure that can be used for both regression and classification.
# 
# Analogy: Consult several doctors, based on a combination of weighted diagnoses—weight assigned based on the previous diagnosis accuracy
# 
# 
# How boosting works?
# - Weights are assigned to each training tuple
# - A series of k classifiers is iteratively learned
# - After a classifier Mi is learned, the weights are updated to allow the subsequent classifier, Mi+1 , to pay more attention to the training tuples that were misclassified by Mi 
# - The final M* combines the votes of each individual classifier, where the weight of each classifier's vote is a function of its accuracy
# 
# 
# Boosting algorithm can be extended for numeric prediction. Compared to bagging: Boosting tends to have greater accuracy, but it also risks overfitting the model to misclassified data

# In[136]:


gb = GradientBoostingClassifier()
fit_predict_score("GradientBoost BaseModel", gb, X_train, X_val, y_train, y_val)


# ### 5.2.15 Gradient Boosting Classifier - Parameters Tunning:

# In[137]:


gb = GradientBoostingClassifier()
print(gb.get_params())


# In[138]:


# define the parameter values that should be searched
parameters = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 1.8, 2.0],
    'n_estimators': range(5, 51, 5),
    'random_state': [0],
}

grid_search_function(gb, 'GradientBoost RSCV', parameters, X_train, y_train, X_val, y_val, cv=3)


# ### 5.2.16 Logistic Regression Classifier - BaseModel:
# Logistic regression is a classification algorithm, used when the value of the target variable is categorical in nature. Logistic regression is most commonly used when the data in question has binary output, so when it belongs to one class or another, or is either a 0 or 1.

# In[139]:


lr = LogisticRegression()
fit_predict_score("LogisticRegression BaseModel", lr, X_train, X_val, y_train, y_val)


# ### 5.2.17 Logistic Regression Classifier - Parameters Tunning:

# In[140]:


lr = LogisticRegression()
print(lr.get_params())


# In[141]:


# define the parameter values that should be searched
parameters = {
    'l1_ratio': np.linspace(0.0001, 100, 200),
    'penalty': ['l1', 'l2', 'elasticnet'],
}

grid_search_function(lr, 'LogisticRegression RSCV', parameters, X_train, y_train, X_val, y_val, cv=3)


# # 6. Results and Conclusion

# ### 6.1 Performance of Different Models:

# In[142]:


# create a dataframe using a dictionary
models_matrix = pd.DataFrame(dict_clf)

# set Model column as index of the dataframe
models_matrix = models_matrix.set_index('Model')

models_matrix['ROC_AUC Difference'] = models_matrix['Train ROC_AUC'] - models_matrix['Test ROC_AUC']

# Sort the dataframe by Test_Accuracy
models_matrix.sort_values(by=['Test ROC_AUC'], ascending=False)


# In above table, the model performance statistics of the different models were compared to select the final model for prediction. The Naive Bayes (Gaussian) Classifier - Base Model was selected based on its performance. It had the highest Test ROC_AUC score of around 78% and there is zero difference/gap between Test ROC_AUC and Training ROC_AUC.

# ### 6.2 Winner Model:

# - Choose a winner model.
# - Make predictions on the hold-out test set whose outcome variables are not known (data we imported as: `test`).
# - Write the code below to make predictions with this model. At the end it should output the prediction of the labels.
# - Save the predictions in a csv file
# - There will be a two columns in the csv file 'id' and 'Response'.

# In[143]:


# hold-out set test
test.head()


# #### The winner model is Naive Bayes (Gaussian) Classifier - Base Model:

# In[144]:


# initiate the winner model
final_model = naive_bayes.GaussianNB()

# fit the model
final_model = final_model.fit(X_train, np.ravel(y_train))

# Make predictions on the hold-out test set
pred = final_model.predict(test)

pred


# ### 6.3 Submission

# In[145]:


test_ids = test.index

# create a dataframe with id and predictions column
submission = pd.DataFrame(data = {'id': test_ids, 'Response': pred})

# Save the predictions in a csv file
submission.to_csv('predicted_labels.csv', index = False)

submission.head()


# ### 6.4 Findings:

# To answer the question (What variables can lead to an increase in insurance premium?), we Looked at correlation between other variables and insurance premium variable. We also built a Random Forest model to predict Insurance Premium, however we did not find any strong variables that can be big predictors of insurance premium.
# 
# Research question 2 indicated that the Vehicle Damage goes up with: The age of the vehicle; and Age of the person. Whereas the Vehicle Damage goes down with: Those who are previously insured. Another useful finding was that people who are older and have older cars are more at risk of having vehicle damage.
# 
# To predict whether health insurance policyholders will also be interested in purchasing vehicle insurance from the same, we built multiple classification algorithms. The Naive Bayes (Gaussian) Classifier - Base Model was selected based on its performance. It had the highest Test ROC_AUC score of around 78% and there is zero difference/gap between Test ROC_AUC and test (hold-out) sets. This model can predict the customers who are interested in vehicle insurance which will ultimately help the company to plan its communication strategy and increase the revenue.
# 
# If we had more time, we could build more models after under-sampling the majority class or oversampling the minority class in order to rebalance our dataset.

# # 7. References

# - Machine Learning in Action, by Peter Harrington, Manning Publications, 2012 https://www.manning.com/books/machine-learning-in-action
# 
# - Wikipedia: https://www.wikipedia.org/
# 
# - Towards Data Science: https://towardsdatascience.com/
# 
# - Scikit-Learn: https://scikit-learn.org/
# 
# - Python for Data Analysis Book: https://wesmckinney.com/pages/book.html
# 
# - KDnuggets: https://www.kdnuggets.com/
# 
# - https://www.researchgate.net/profile/P-Pintelas/publication/228084509_Handling_imbalanced_datasets_A_review/links/0c960517fefa59fa6b000000/Handling-imbalanced-datasets-A-review.pdf
# 
# - https://www.ijrter.com/papers/volume-3/issue-4/a-review-on-imbalanced-data-handling-using-undersampling-and-oversampling-technique.pdf
# 
# - https://link.springer.com/chapter/10.1007/978-981-4585-18-7_2

# # 8. Project Files:

# **o Jupyter Notebook Single File:**
#     - dsc540_finalprojectnotebook.ipynb
# **o HTML Output of Jupyter Notebook Single File:**
#     - dsc540_finalprojectnotebook.html
# **o Data Files:**
#     - train.csv
#     - test.csv
# **o Data Files Kaggle Link:**
#     - https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction
# **o Final Predicted Labels on Test Data:**
#     - predicted_labels.csv
# **o Pictures:**
#     - Cross-selling-Single.jpg
# **o Other Files:**
#     - ReadMe.docx

# ## Thank you!
