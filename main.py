# %%
# EXPLORATORY DATA ANALYSIS

# libraries

import xlrd
import matplotlib.colors as colors
import random
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
droplist = val_customer.loc[val_customer['nan_per'] >= 50, ['Customer ID']]

# %%

# Then we iterate for filtering the dataframe
for c in droplist['Customer ID']:
    data = data[data['Customer ID'] != c]
print(data.shape)

# %%
var = df1.shape[0] - data.shape[0]

data.isnull().sum()

# With this evidence, we can see that most of the NaN customer data
# is not useful. Also, the tiny part of NaN data is too short and not
# significant enough to be imputed.

# %%
# That's why it's better to go back
# few steps and merging df1 and df1 with inner joined

data = df1.merge(df2, on="Transaction ID", how='inner')
data.shape

# %%

# That decision made us lost this percentage of the total amount of data
dataloss = var*100/df1.shape[0]
dataloss
# 18.34% of loss, it is not that bad at all


# %%
# Let's Watch a quicker look to our dataset

d = dtale.show(data)
d.open_browser()

# %%

# FUNCTIONS

# TABLE FUNCTION FOR CATEGORY VARIABLES


def tab(data, col1, col2="Freq", norm=False, marg=True, marg_name='Total'):
    # Warning
    if ((col2 != "Freq") & (col2 not in data.columns)) | (col1 not in data.columns):
        print('There is at least one column that not belongs to the dataframe')
        return
    # Absolute Frequency table
    if col2 == "Freq":
        if norm == False:
            tab = pd.crosstab(index=data[col1], columns=col2, normalize=False,
                              margins=marg, margins_name='Total')
    # Relative Frequency
        else:
            tab = pd.crosstab(index=data[col1], columns=col2, normalize=True,
                              margins=marg, margins_name='Total').round(4).apply(lambda r: r*100, axis=1)
    # Relative frequencies with 2 variables
    elif (col2 != "Freq") & (norm == True):
        tab = pd.crosstab(index=data[col1], columns=data[col2], normalize=True,
                          margins=marg, margins_name='Total').round(4).apply(lambda r: r*100, axis=1)
    # Absolute frequencies for 2 variables
    else:
        tab = pd.crosstab(index=data[col1], columns=data[col2], normalize=False,
                          margins=marg, margins_name='Total')

    return tab


# COUNTPLOT FUNCTION
colors_list = list(colors._colors_full_map.values())


def cntplot(x, data, title):
    sns.set_style('whitegrid')
    sns.set_context('notebook')
    sns.catplot(x=x,
                data=data,
                kind='count',
                palette=random.choices(colors_list, k=2),
                edgecolor='black',
                linewidth=1.25)
    plt.title(title, fontsize=16, weight='bold')


# PIE CHART FUNCTION
def pieplot(data, catcol, title):
    tab(data, catcol,
        norm=True, marg=False).plot(kind='pie',
                                    autopct='%.2f',
                                    colors=random.choices(colors_list, k=2),
                                    subplots=True,
                                    wedgeprops={'linewidth': 1.5,
                                                'edgecolor': 'black'})
    plt.title(title, fontsize=16, weight='bold')


# %%
# DATA MANIPULATION & Visualization
# %%
# Transaction ID
# We have to change its data type to string type
data['Transaction ID'] = data['Transaction ID'].astype('string')


# %%
# Customer ID
# We have to change its data type to string type
data['Customer ID'] = data['Customer ID'].astype('string')
data['Customer ID']

# %%
# Payment mode
data['Payment_Mode'].describe()
data['Payment_Mode'].value_counts()
tab(data, 'Payment_Mode', norm=True)


# %%
pieplot(data, 'Payment_Mode', title='Payment Method Pie Chart')
cntplot(data['Payment_Mode'], data, title='Payment Method Countplot')


# %%
# Gender
data['Gender'].describe()
data['Gender'].value_counts()
tab(data, 'Gender', norm=False)

# %%
pieplot(data, 'Gender', title='Gender Pie Chart')
cntplot(data['Gender'], data, title='Gender Countplot')


# %%

# AGE

# Distribution of the variable
data['Age'].hist(bins=15)

# Distribution of the logarithm transformation
np.log(data['Age']).hist(bins=15)


# %%
sns.boxplot(x='Age', data=data, linewidth=1.5, orient="h", palette="Set2")
sns.swarmplot(x="Age", data=data, color=".25", alpha=0.05)


# %%

data_num = data.select_dtypes(include=["number"])
data_num.corr()
sns.heatmap(data_num.corr(), annot=True)


# %%

data['Date of Travel'].apply(lambda s: xlrd.xldate.xldate_as_datetime(s, 0))


# %%

data['Date of Travel']


# %%%

val_customer['nan_count'].sum()

# %%
!pip install dtale
# %%

data.isnull().sum()

# %%
data.groupby(by='Customer ID')['Price Charged'].mean()
# %%
