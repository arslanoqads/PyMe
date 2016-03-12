import pandas as pd
import numpy as np
import pylab as pl
from sklearn.neighbors import KNeighborsRegressor



# Initialize KNN
# We will impute data using the KNN method with 3 neighbours
income_imputer = KNeighborsRegressor(n_neighbors=3)

# Path of the file
path='C:\\Users\\Arslan Qadri\\Google Drive\\Sem1\\Multivariate\\Project\\others\\'
# Load Dataset into a dataframe
df = pd.read_csv(path+"training.csv")


# Create Train/Test sets
is_test = np.random.uniform(0, 1, len(df)) > 0.75 # generate random values
train = df[is_test==False] # training set
test = df[is_test==True] # test set

# The below tests are seperated on the basis of whether they contain NULLs
MI = train[train.MonthlyIncome.isnull()==False]  #non null
MI_null = train[train.MonthlyIncome.isnull()==True] #nulls

# Decide columns to pick  
cols = ['NumberRealEstateLoansOrLines', 'NumberOfOpenCreditLinesAndLoans']

# Start imputation process
# Using the selected columns as features, and Monthlyincome as target, fit KNN
income_imputer.fit(MI[cols], MI.MonthlyIncome)


# Predict NULL values in the MI_NULL set
new_values = income_imputer.predict(MI_null[cols])


# Impute new values
MI_null['MonthlyIncome'] = new_values

# Join MI and MI_NULL datasets
train = MI.append(MI_null)

# Using same logic impute in the test set.
test['monthly_income_imputed'] = income_imputer.predict(test[cols])