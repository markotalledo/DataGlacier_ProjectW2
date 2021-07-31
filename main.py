# %%
# EXPLORATORY DATA ANALYSIS
# Basics

from math import sqrt
from scipy import stats
from scipy.stats import kurtosis
import scipy.stats as scy
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import dtale
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from tabulate import tabulate
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Regression modelling

# Visualization

# Metrics

# Splitting

# Stats & Math

# %%
# Showing determined number of columns
pd.options.display.max_columns = 50
pd.options.display.max_rows = 1000

# %%

# DATA

# Importing dataset to work with them
cab_data = pd.read_csv('data/in/Cab_Data.csv')
city_data = pd.read_csv('data/in/City.csv')
customerID = pd.read_csv('data/in/Customer_ID.csv')
transacID = pd.read_csv('data/in/Transaction_ID.csv')


# %%

# Applying merge instance to join tables on Customer ID
df1 = transacID.merge(customerID, on='Customer ID')
df1

# %%
# Applying merge instance to join tables on City column
df2 = cab_data.merge(city_data, on='City')
df2

# %%

# Merging the last output tables on Transaction ID
# This time NAN's will appear because those trip's data is not available
data = df1.merge(df2, on="Transaction ID", how='outer')

# I had merge on Transaction ID because each Customer has many
# Transactions

# There are 80706 entries with NAN data, we must seek for imput them
# or delete them, but we will se the best way to approach it

# %%

# We can see that every column has the same number of null data
# It's not caused by the columns, instead, it is caused by
# Many customers has not available information, that's why we will try
# To get the nulls per each customer
data.isnull().sum()


# %%

# Creating a dataframe that displays transaction count per each
# customer ID

val_customer = pd.DataFrame(data['Customer ID'].value_counts())
val_customer.reset_index(level=0, inplace=True)
val_customer = val_customer.rename({'index': 'Customer ID',
                                    'Customer ID': 'count'},
                                   axis=1)
val_customer


# %%

# Creating a dataframe that shows the count of the transactions
# with NAN data for each customer


nan_customer = pd.DataFrame(
    data.loc[(data['Company'].isnull()), ['Customer ID']].value_counts())
nan_customer.reset_index(level=0, inplace=True)
nan_customer = nan_customer.rename(
    {'index': 'Customer ID', 0: 'nan_count'}, axis=1)
nan_customer


# %%

# Merging last two tables in order to compare quantities

val_customer = val_customer.merge(nan_customer, on="Customer ID")


# %%
# Looking for the percentage of NAN entries in each customer
val_customer['nan_per'] = round(
    (val_customer['nan_count']/val_customer['count'])*100, 2)
val_customer = val_customer.sort_values(by='nan_per', ascending=True)
val_customer

# %%

# We'll assume that the information of every customer that
# has 25% NAN data or more is not useful because
# It's not a loyal customer for our company
# So We'll proceed to clean that data

dim_v1 = data.shape
dim_v1

# First we will get a list of those clients
droplist = val_customer.loc[val_customer['nan_per'] >= 25, ['Customer ID']]


# %%
data.drop(data['Customer ID'] == droplist)


# %%%

val_customer['nan_count'].sum()

# %%
!pip install dtale
# %%

data.isnull().sum()

# %%
data.groupby(by='Customer ID')['Price Charged'].mean()
# %%
