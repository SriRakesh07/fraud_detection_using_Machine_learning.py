#!/usr/bin/env python
# coding: utf-8

# # Classifying Fraudulent and Valid Transactions

# ## Goals of This Project.
# 1. Classify fraudulent transactions as accurately as possible.
# 2. Learn how to explore data, pre-process it, apply multiple popular machine learning methods and diagnose their performances (or lack thereof)

# In[1]:


# loading needed methods
import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.cm as cm
from random import seed,sample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, roc_curve, auc,precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier


# In[2]:


# loading data

data = pd.read_csv("/Users/srirakeshnagasai/Downloads/onlinefraud 2.csv")


# ## Exploratory Data Analysis
# <a id='EDA'></a>

# ###  Summary of Dataset
# <a id='SummData'></a>

# In[3]:


data.describe()


# In[4]:


data.shape


# In[5]:


data.head(7)


# ### Looking at Account Types 
# <a id='AcctType'></a>

# In[6]:


# adding feature type1
data_new = data.copy() 
data_new["type1"] = np.nan 

# filling feature column
data_new.loc[data.nameOrig.str.contains('C') & data.nameDest.str.contains('C'),"type1"] = "CC" 
data_new.loc[data.nameOrig.str.contains('C') & data.nameDest.str.contains('M'),"type1"] = "CM"
data_new.loc[data.nameOrig.str.contains('M') & data.nameDest.str.contains('C'),"type1"] = "MC"
data_new.loc[data.nameOrig.str.contains('M') & data.nameDest.str.contains('M'),"type1"] = "MM"

    


# In[7]:


# Subsetting data into observations with fraud and valid transactions:
fraud = data_new[data_new["isFraud"] == 1]
valid = data_new[data_new["isFraud"] == 0]


# In[8]:


# seeing the counts of transactions by type1 (CC,CM,MC,MM)
print("Fraud transactions by type1: \n",fraud.type1.value_counts())
print("\n Valid transactions by type1: \n",valid.type1.value_counts())


# In[9]:


# getting rid of type1 column.

fraud = fraud.drop('type1', 1)
valid = valid.drop('type1',1)
data_new = data_new.drop('type1',1)


# ### Looking at Transaction Types
# <a id='TransType'></a>

# In[10]:


# seeing the counts of transactions by type
print("Fraud transactions by type: \n",fraud.type.value_counts())
print("\n Valid transactions by type: \n",valid.type.value_counts())


# In[11]:


# Subsetting data according to the conclusion above
# I don't have to subset for the fraud dataset because all of their transaction types are either TRANSFER or CASH_OUT

valid = valid[(valid["type"] == "CASH_OUT")| (valid["type"] == "TRANSFER")]
data_new = data_new[(data_new["type"] == "CASH_OUT") | (data_new["type"] == "TRANSFER")]


# ### Looking balances before and after the transaction
# <a id='bal'></a>

# Most, if not all, of the observations have errors in calculating the balances before and after the transaction. 

# In[12]:


wrong_orig_bal = sum(data["oldbalanceOrg"] - data["amount"] != data["newbalanceOrig"])
wrong_dest_bal = sum(data["newbalanceDest"] + data["amount"] != data["newbalanceDest"])
print("Percentage of observations with balance errors in the account giving money: ", 100*round(wrong_orig_bal/len(data),2))
print("Percentage of observations with balance errors in the account receiving money: ", 100*round(wrong_dest_bal/len(data),2))


# In[13]:


## Calculating some quantities to justify or reject some assumptions

# flatten the subsetted dataframe of floats into an array of floats
relevant_cols = data[["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]].values.flatten()
# number of observations with negative numbers
num_neg_amt = sum(n < 0 for n in relevant_cols)
# number of observations where the amount given is greater than the amount that is in the giver's account
num_amt_oldgiver = sum(data["amount"] > data["oldbalanceOrg"]) 
# number of observations where the amount received is greater than the amount that is in the receiver's account
num_amt_newreceiver = sum(data["amount"] > data["newbalanceDest"]) 

print("number of observations with negative numbers: ", num_neg_amt)
print("number of observations where the amount given is greater than the amount that is in the giver's account: "
      , num_amt_oldgiver)
print("number of observations where the amount received is greater than the amount that is in the receiver's account: "
      , num_amt_newreceiver)


# With these calculations, hypotheses 2 and 3 have been rejected. 

# In[14]:


# counting number of observations where oldbalanceOrg - amount != newbalanceOrig or newbalanceDest + amount != newbalanceDest
# Essentially, I am counting the number of observations where the effects of the transactions are not properly reflected
# the balances of account sending money and the account receiving money.

num_wrong_bal = (data["oldbalanceOrg"] - data["amount"] != data["newbalanceOrig"]) | (data["newbalanceDest"] + data["amount"] != data["newbalanceDest"])
print("Percentage of observations with balance errors: ", 100*round(sum(num_wrong_bal)/len(data),2))


# In fact, **all** observations have balance contain errors.
# 
# Since I don't know why these errors are caused, I cannot replace them. 
# 
# I also don't want to get rid of the variables oldbalanceOrg, newbalanceOrig, newbalanceDest, oldbalanceDest since they might be important in identifying fraudulent transactions from valid transactions.
# 
# So for now, I will them be.
# 
# However, do these errors differ between fraudulent and valid transactions?

# In[15]:


# adding features errorBalanceOrg, errorBalanceDest
data_new["errorBalanceOrg"] = data_new.newbalanceOrig + data_new.amount - data_new.oldbalanceOrg
data_new["errorBalanceDest"] = data_new.oldbalanceDest + data_new.amount - data_new.newbalanceDest

# Subsetting data into observations with fraud and valid transactions:
fraud = data_new[data_new["isFraud"] == 1]
valid = data_new[data_new["isFraud"] == 0]


# In[16]:


print("Summary statistics of errorBalanceOrg for fraudulent transactions: \n",fraud["errorBalanceOrg"].describe())
print("\n Summary statistics of errorBalanceOrg for valid transactions: \n",valid["errorBalanceOrg"].describe())


# From the summary statistics on the errorBalanceOrg, it seems that a large proportion of the data have an error of 0 or close to zero. This is indicated by the fact that the most negative error is -7.450581e-09 or $-7.450581 x 10^{-9}$ which is very small and close to 0, and the 3rd quartile is 0 (that is, about 75% of the data is between -7.450581e-09 and 0). However, there are some large errors, the largest error being 10,000,000.
# 
# On the other hand, for valid transactions, a large proportion of the data have large errors. For instance,
# about 75% of the data haver errors exceeding 52,613.43 (the first quartile). The largest error is 92,445,520.

# In[17]:


print("Summary statistics of errorBalanceDest for fraudulent transactions: \n",fraud["errorBalanceDest"].describe())
print("\n Summary statistics of errorBalanceDest for valid transactions: \n",valid["errorBalanceDest"].describe())


# From the summary statistics of the errorBalanceDest variable, the errors are huge in both directions (both fraudulent and valid transactions have large positive and negative errors in the accounts where money has been moved to.)
# 
# Let's see what the differences look like when I plot errorBalanceOrg and errorBalanceDest together.

# In[18]:


errors = ["errorBalanceOrg", "errorBalanceDest"]
ax = plt.subplot()

fplot = fraud.plot(x="errorBalanceOrg",y="errorBalanceDest",color="red",kind="scatter",ax=ax,label="Fraudulent transactions")
vplot = valid.plot(x="errorBalanceOrg",y="errorBalanceDest",color="green",kind="scatter",                   alpha=0.01,ax=ax,label="Valid transactions")
plt.title("errorBalanceOrg vs errorBalanceDest")
plt.show()


# It seems that many fraudulent transactions that are found in the top right corner where errorBalanceDest > 0, whereas transactions occur much more often when the errorBalanceDest <= 0. 

# In[19]:


print("Proportion of fraudulent transactions with errorBalanceDest > 0: ", len(fraud[fraud.errorBalanceDest > 0])/len(fraud))
print("Proportion of valid transactions with errorBalanceDest > 0: ", len(valid[valid.errorBalanceDest > 0])/len(valid))
print("Proportion of fraudulent transactions with errorBalanceOrg > 0: ", len(fraud[fraud.errorBalanceOrg > 0])/len(fraud))
print("Proportion of valid transactions with errorBalanceOrg > 0: ", len(valid[valid.errorBalanceOrg > 0])/len(valid))


# ### Conclusion: 
# 
# The spread of errors in both the balanceOrg and balanceDest variables are large, however valid transactions are much more likely to have an errorBalanceOrg > 0.
# 
# Similarly, fraudulent transactions are much more likely to have errorBalanceDest > 0 than valid transactions.
# 
# In addition, only valid transactions have errorBalanceDest > 10,000,000
# 
# These distinctions and probably more, make errorBalanceDest and errorBalanceOrg potentially effective features. 

# ### Another Look at Transaction Types and Account Names
# <a id='AcctandTrans'></a>

# In[20]:


print("Fraud transactions by type: \n",fraud.type.value_counts())


# Clearly, fraudulent transactions exclusively involved cashouts and transfers

# In[21]:


pd.DataFrame.head(data_new,13)


# However, in this sample the account that the money to transferred to tends to not be the account used to make the cashout. 
# 
# 
# Let's test this statement programmatically.

# In[22]:


# separating transfers and cashouts for fraud accounts

fraud_transfer = fraud[fraud["type"] == "TRANSFER"]
fraud_cashout = fraud[fraud["type"] == "CASH_OUT"]

# checking if the recipient account of a fraudulent transfer was used as a sending account for cashing out 
fraud_transfer.nameDest.isin(fraud_cashout.nameOrig).any()


# ### Conclusion:
# 
# Thus in this dataset, for fraudulent transactions, the account that received funds during a transfer was not used at all for cashing out.
# 
# If that is the case, there seems to be no use for nameOrig or nameDest since there seems to be no restrictions on which accounts cashout from fraudulent transactions.
# 
# Thus, I am omitting the nameOrig and nameDest columns from analysis.

# In[23]:


# getting rid of nameOrig and nameDest column.
names = ["nameOrig","nameDest"]
fraud = fraud.drop(names, 1)
valid = valid.drop(names,1)
data_new = data_new.drop(names,1)


# ### Looking at Flagged Transactions
# <a id='Flag'></a>

# From the overview, the variable isFlaggedFraud is described as transactions that were flagged as fraud.
# 
# To be flagged as fraud, the transaction would have to be fraudulent and involve a transfer of more than 200, 000 units in a specified currency.
# 
# With that in mind, I have some questions. 

# In[24]:


# how many observations were flagged as Fraud?
flagged = data_new[data_new["isFlaggedFraud"] == 1]
flagged_correctly = sum(flagged["isFraud"] == 1)
flagged_wrongly = len(flagged) - flagged_correctly
total = flagged_correctly + flagged_wrongly
print(flagged_correctly," observations were flagged correctly and ", flagged_wrongly,       " observations were flagged wrongly for a total of ", total, " flagged observations.")

# how many observations where the transaction is fraudulent, the transaction is a transfer and the amount is greater 
# than 200, 000 are in the dataset
should_be_flagged = fraud[(fraud["amount"] > 200000) & (fraud["type"] == "TRANSFER")]
print("number of observations that should be flagged: ",len(should_be_flagged))


# ### Conclusion: 
# 
# In a modified dataset with more than 2 million observations, a variable that brings attention to only 16 observations is insignificant.
# 
# Furthermore, the number of transactions that should have been flagged far exceeds the number of observations that were actually flagged.
# 
# In addition, I am trying to develop a new fraud detection screen that does not depend on a pre-existing fraud detection scheme.
# 
# For that reason, I am omitting the isFlaggedFraud column from the analysis.

# In[25]:


# dropping isFlaggedFraud column from the fraud,valid, and new_data datasets

fraud = fraud.drop("isFlaggedFraud",1)
valid = valid.drop("isFlaggedFraud",1)
data_new = data_new.drop("isFlaggedFraud",1)


# ### Looking at Time
# <a id='Time'></a>

# In[26]:


# Time patterns

bins = 50

valid.hist(column="step",color="green",bins=bins)
plt.xlabel("1 hour time step")
plt.ylabel("# of transactions")
plt.title("# of valid transactions over time")

fraud.hist(column ="step",color="red",bins=bins)
plt.xlabel("1 hour time step")
plt.ylabel("# of transactions")
plt.title("# of fraud transactions over time")

plt.tight_layout()
plt.show()


# There are stark difference between the *step* data between valid and fraud transactions.
# 
# 1. A large proportion of valid transactions occur between around the 0th and 60th timestep as well as the 110th and 410th time-steps.
# 2. The frequency at which fraudulent transactions occur does not seem to change much over time.
# 
# However the visualizations showcase the number of transactions for each time step over the course of a month.
# 
# Let's see what the patterns look like over any particular, day of the week or hour of the day.
# 

# In[27]:


# getting hours and days of the week
num_days = 7
num_hours = 24
fraud_days = fraud.step % num_days
fraud_hours = fraud.step % num_hours
valid_days = valid.step % num_days
valid_hours = valid.step % num_hours

# plotting scatterplot of the days of the week, identifying the fraudulent transactions (red) from the valid transactions (green) 
plt.subplot(1, 2, 1)
fraud_days.hist(bins=num_days,color="red")
plt.title('Fraud transactions by Day')
plt.xlabel('Day of the Week')
plt.ylabel("# of transactions")

plt.subplot(1,2,2)
valid_days.hist(bins=num_days,color="green")
plt.title('Valid transactions by Day')
plt.xlabel('Day of the Week')
plt.ylabel("# of transactions")

plt.tight_layout()
plt.show()


# Note: With respect to days, day 0 does not necessarily mean the first day of the week, Sunday. 
# 
# E.g If day 0 is Wednesday, then day 1 is Thursday, day 2 is Friday and so on...
# 
# From the graphs above, there is little evidence to suggest that fraudulent transactions occur at particular days of the week.
# 
# Much like valid transactions, fraudulent transactions seem to occur uniformally for each day of the week.
# 
# Thus I won't make a feature showing what day of the week that the transaction occured.

# In[28]:


plt.subplot(1, 2, 1)
fraud_hours.hist(bins=num_hours, color="red")
plt.title('Fraud transactions by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel("# of transactions")


plt.subplot(1, 2, 2)
valid_hours.hist(bins=num_hours, color="green")
plt.title('Valid transactions by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel("# of transactions")

plt.tight_layout()
plt.show()


# ### Conclusion:
# 
# Note: With respect to days, hour 0 does not necessarily mean 1am in the morning. 
# 
# E.g If hour 0 is 9am, then hour 1 is 10 am, hour 2 is 11am and so on...
# 
# From the graphs above, there is strong evidence to suggest that from hour 0 to hour 9 (inclusive) valid transactions very seldom occur. On the other hand, fraudulent transactions still occur at similar rates to any hour of the day outside of hours 0 to 9 (inclusive).
# 
# In response to this, I will create another feature HourOfDay, which is the step column with each number taken to modulo 24.

# In[29]:


dataset1 = data_new.copy()


# adding feature HourOfDay to Dataset1 
dataset1["HourOfDay"] = np.nan # initializing feature column
dataset1.HourOfDay = data_new.step % 24


print("Head of dataset1: \n", pd.DataFrame.head(dataset1))


# ### Looking at Amounts Moved in Transactions
# <a id='Amt'></a>

# In[30]:


# Seeing summary statistics of the data

print("Summary statistics on the amounts moved in fraudulent transactions: \n",pd.DataFrame.describe(fraud.amount),"\n")
print("Summary statistics on the amounts moved in valid transactions: \n", pd.DataFrame.describe(valid.amount),"\n")


# It seems that during fraudulent transactions, the amount moved is capped at 10 million currency units.
# 
# Whereas for valid transactions, the amount moved is capped at about 92.4 million currency units.
# 
# when plotting time-steps against amount moved we get this plot...
# 

# In[31]:


# plotting overlayed step vs amount scatter plots

alpha = 0.3
fig,ax = plt.subplots()
valid.plot.scatter(x="step",y="amount",color="green",alpha=alpha,ax=ax,label="Valid Transactions")
fraud.plot.scatter(x="step",y="amount",color="red",alpha=alpha,ax=ax, label="Fraudulent Transactions")

plt.title("1 hour timestep vs amount")
plt.xlabel("1 hour time-step")
plt.ylabel("amount moved in transaction")
plt.legend(loc="upper right")

# plotting a horizontal line to show where valid transactions behave very differently from fraud transactions

plt.axhline(y=10000000)
plt.show()


print("Proportion of transactions where the amount moved is greater than 10 million: ",       len(data_new[data_new.amount > 10000000])/len(data_new))


# ### Conclusion:
# 
# Only valid transaction involved amounts larger than 10,000,000, however these transactions make up less than 0.01% of the relevant data.
# 
# When the amounts moved is less than 10,000,000 there doesn't seem to be a large difference fraudulent and valid transactions.
# 
# I leave the variable amount as is without creating a feature out of it.

# In[32]:


# finalizing dataset
dataset = dataset1.copy() # unchanged dataset1


# ## Pre-processing Data
# <a id='Preprocess'></a>

# ### Handling Categorical Variables
# <a id='Cat'></a>
# 
# Note that many algorithms require that all elements used in the computation are numbers.
# 
# For that reason, the categorical variables encoded as string must be encoded as numbers. Since there is no "order"/"hierarchy" in the type variable, the method I will use to numerically encode categorical variables is called 1 hot encoding.
# 
# One-Hot encoding involves creating indicator variables for each category in a categorical variable.
# 
# If an observation is part of a particular category (e.g. the transaction type is CASH_OUT), the indicator variable associated with the category would be 1. If it isn't part of a particular category, then the indicator variable associated with that category would be 0.
# 

# In[33]:


# getting one-hot encoding of the 'type' variable

dataset = pd.get_dummies(dataset,prefix=['type'])


# In[34]:


pd.DataFrame.head(dataset)


# ## Splitting and Standardizing Data.
# <a id='Split'></a>
# Similarly, many, if not all, machine learning algorithms perform better when the data is standardized/normalized (when all values are between 0 and 1 inclusive).
# 
# We will do this to standardize the data without standardizing the target variable isFraud.
# 
# Additionally, we will also split the data up into training sets and testing sets. A common split is to separate 80% of the data as the training set and the rest as the testing set. However we will rely on the "default" split which is 75% of the data is used as the training set, 25% is used as the testing set.
# 

# In[35]:


# Setting random_state and seed so that the training/testing splits and model results are reproducible
RandomState = 42
seed(21)


# 42 is used often due to Hitchhiker's Guide to the Galaxy, I will use a number that a far smaller group may understand.
# Not that the actual number doesn't matter and is only used to make sure results are reproducible.
# creating training and testing sets
X = dataset.drop("isFraud",1)
y = dataset.isFraud
X_train, X_test, y_train, y_test = train_test_split(X, y)
    
# Normalizing data so that all variables follow the same scale (0 to 1)
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Model Selection
# <a id='Models'></a>

# ### Model 1: Artificial Neural Networks
# <a id='Model-1'></a>
# 

# In[ ]:


ncols = len(X.columns)
hidden_layers = (ncols,ncols,ncols)
max_iter = 1000
MLP = MLPClassifier(hidden_layer_sizes=hidden_layers,max_iter=1000,random_state=RandomState)

# training model
MLP.fit(X_train,y_train)
    
# evaluating model on how it performs on balanced datasets
predictionsMLP = MLP.predict(X_test)
CM_MLP = confusion_matrix(y_test,predictionsMLP)
CR_MLP = classification_report(y_test,predictionsMLP)
fprMLP, recallMLP, thresholdsMLP = roc_curve(y_test, predictionsMLP)
AUC_MLP = auc(fprMLP, recallMLP)
    
resultsMLP = {"Confusion Matrix":CM_MLP,"Classification Report":CR_MLP,"Area Under Curve":AUC_MLP}


# In[37]:


# showing results from Multilayered perceptrons developed from each dataset
for measure in resultsMLP:
    print(measure,": \n",resultsMLP[measure])


# In the context of fraud detection the performance of the Neural Network isn't terrible, but it isn't great either. The loss is performance is very likely due to the phenomenon that Neural Networks perform worse when the data is imbalanced. When data is imbalanced, Neural Networks and many other models trained on the data tend to be very biased towards the *majority class*. In our case, the majority class are valid transactions.  
# 
# This model will be the benchmark that I will compare other individual models against.
# 
# The next few models will be generated from methods that are well-known for handling imbalanced data effectively.

# ### Model 2: Random Forest.
# <a id='Model-2'></a>
# 
# 

# In[ ]:


# Train model
parametersRF = {'n_estimators':15,'oob_score':True,'class_weight': "balanced",'n_jobs':-1,                 'random_state':RandomState}
RF = RandomForestClassifier(**parametersRF)
fitted_vals = RF.fit(X_train, y_train)
 
# Predict on testing set
predictionsRF = RF.predict(X_test)
 
     
# Evaluating model
CM_RF = confusion_matrix(y_test,predictionsRF)
CR_RF = classification_report(y_test,predictionsRF)
fprRF, recallRF, thresholdsRF = roc_curve(y_test, predictionsRF)
AUC_RF = auc(fprRF, recallRF)

resultsRF = {"Confusion Matrix":CM_RF,"Classification Report":CR_RF,"Area Under Curve":AUC_RF}


# In[ ]:


# showing results from Random Forest

for measure in resultsRF:
    print(measure,": \n",resultsRF[measure])


# As expected, the Random Forest performs much better than the Neural Networks. Instead of crowning this model as the best model, let's try another model known for performing well in imbalanced datasets.

# ### Model 3: e**X**treme **G**radient **B**oosting Trees (or XGB trees for short)
# <a id='Model-3'></a>

# In[ ]:


# Train model
weights = (y == 0).sum() / (1.0 * (y == 1).sum()) # for unbalanced datasets, these weights are recommended
parametersXGB = {'max_depth':3,'scale_pos_weight': weights,'n_jobs':-1,                 'random_state':RandomState,'learning_rate':0.1}
XGB = XGBClassifier(**parametersXGB)
    
fitted_vals = XGB.fit(X_train, y_train)
 
# Predict on testing set
predictionsXGB = XGB.predict(X_test)
 
     
# Evaluating model
CM_XGB = confusion_matrix(y_test,predictionsXGB)
CR_XGB = classification_report(y_test,predictionsXGB)
fprXGB, recallXGB, thresholds_XGB = roc_curve(y_test, predictionsXGB)
AUC_XGB = auc(fprXGB, recallXGB)
resultsXGB = {"Confusion Matrix":CM_XGB,"Classification Report":CR_XGB,"Area Under Curve":AUC_XGB}


# In[ ]:


# showing results from Extreme Gradient Boosting
for measure in resultsXGB:
    print(measure,": \n",resultsXGB[measure],"\n")


# ### Comparing Performances
# <a id='Visuals'></a>

# In[ ]:


print("Number of valid transactions labelled as fraudulent by Random Forest: \n", CM_RF[0,1])
print("Number of valid transactions labelled as fraudulent by XGB trees: \n", CM_XGB[0,1])


# On the basis on limiting the amount of valid transaction labelled as fraudulent, the Random Forest performed better.

# In[ ]:


print("Number of fraud transactions labelled as valid by Random Forest: \n", CM_RF[1,0])
print("Number of fraud transactions labelled as valid by XGB trees: \n", CM_XGB[1,0])


# On the basis on limiting the amount of fraudulent transactions labelled as fraudulent, the XGB trees performed better.

# Based purely on the results on the confusion matrix, the better model is decided by which model incurs the lowest costs.
# 
# If the combined cost of mislabelling over 100 more valid transactions as fraudulent exceeds the cost of mislabelling a few more fraudulent transactions as valid then the random forest would be a better model.
# 
# Otherwise, the Extreme Gradient Boosted model would be superior.
# 
# Some of the other metrics tracked (precision, recall, f1-score which are found in the classification report) will convey the same information that is offered by the confusion matrix.
# 
# So what if we compared the classification reports?

# In[ ]:


print("Note: scores in the same vertical level as 0 are scores for valid transactions. \n       Scores in the same vertical level as 1 are scores for fraudulent transactions. \n")
print("Classification Report of Random Forest: \n", CR_RF)
print("Classification Report of XGB trees: \n", CR_XGB)


# While the recall scores for both models are identical, the Random Forest performed slightly better in terms of their precision score for fraudulent transactions. 
# 
# This means that there are considerably less false positives (identifying valid transactions as fraudulent) in the Random Forest than in the XGB model. This makes sense given what we've seen in their confusion matrices (a few valid transactions labelled as fraudulent by Random Forest, compared to over 100 by the XGB model). 
# 
# Based on the classification report, the Random Forest is superior.

# What about AUC's (area under the curve)?
# 
# The only reason why I computed AUC's is because it is a popularly used metric to measure performance in kaggle competitions. 
# The curve in Area Under **Curve** is a plot of the true positive rates (in our case, the proportion of valid transactions labelled as valid) against the false positive rate (in our case, the proportion of fraudulent transactions labelled as valid). The curve is also known as the Receiver Operating Characteristic Curve or ROC.
# 
# The ideal AUC is then 1 (all transactions predicted as valid are actually valid). 
# 
# 

# In[ ]:


print("\nReceiver Operating Characteristic Curves for Random Forests and Extreme Gradient Boosted Trees: \n")
plt.subplot(1, 2, 1)
plt.plot(fprRF, recallRF, color='purple', label='ROC curve (area = %0.2f)' % AUC_RF)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([-0.01, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC (Random Forest)')
plt.legend(loc="lower right")


plt.subplot(1, 2, 2)
plt.plot(fprRF, recallRF, color='green', label='ROC curve (area = %0.2f)' % AUC_RF)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([-0.01, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC (XGB)')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

print("\nAUC of Random Forest: \n", AUC_RF)
print("\nAUC of XGB trees: \n", AUC_XGB)


# While the AUC of the Extreme Gradient Boosted Trees is slightly greater than the AUC of the Random Forests. For all practical purposes, they are essentially the same.

# Overall, I would deem the Random Forest as the superior choice because I think the cost of resolving over a 100 valid transactions labelled as fraudulent would exceed the cost of resolving a handful more fraudulent transactions that have been passed off as valid.
# 
# With that said, this may not true in actual companies. The best decision would be to consult people with experience dealing with mislabelled transactions.

# Black-box/Non-parametric methods (like Neural Networks, Random Forests, Extreme Gradient Boosted Trees) are known to not be interpretable (due to having a lack of an equation to interpret from).
# 
# Nonetheless, let's take a look at what features ended up being the most important in classifying transactions.

# In[ ]:


x = np.arange(ncols)

# getting importances of features
importances = RF.feature_importances_

# getting the indices of the most important feature to least important
sort_ind = np.argsort(importances)[::-1]
plt.figure(figsize=(18,7))
plt.bar(x, importances[sort_ind])
plt.xticks(x,tuple(X.columns.values[sort_ind]))
plt.title("Important Features: Greatest to Least")
plt.show()


# It seems that errorBalanceOrg ended by the most important feature by far for classifying transactions followed by oldBalanceOrg and newBalanceOrig.  

# ## Final Remarks
# <a id='Final'></a>

# From the exploring the dataset, we uncovered patterns that allowed us to construct important features and discard useless ones. 
# 
# We applied a few popular machine learning algorithms and saw that methods that involved generating multiple decisions trees and pooling their results together performed better than Multi-Layered Perceptrons (a type of Neural Network). While it may seem that Neural Networks are unsuitable for unbalanced datasets, in some occasions, techniques such as SMOTE, oversampling and undersampling can resolve such issues.  
# 
# Finally, between the Random Forests and the Extreme Gradient Boosting, practical considerations were used to decide that the Random Forests were a better model.
# 
# In particular, Random Forests were better because we thought that the cost of dealing with over a 100 wrongly labeled valid transactions is more expensive than the cost of dealing with a few additional fraudulent transactions.
# 
# Working with this dataset was a lot of work and a lot of fun. I learned a lot about exploring data and when to apply/not apply certain methods.
# 
# Thanks for reading! . 
