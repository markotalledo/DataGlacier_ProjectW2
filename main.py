# %%
# EXPLORATORY DATA ANALYSIS

# %%
# libraries

import fbprophet
from pylab import rcParams
import calendar
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
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

sns.set_style('whitegrid')
sns.set_context('notebook')

# %%

# DATA

# Importing dataset to work with them

path = 'data/in/'
cab_data = pd.read_csv(path + 'Cab_Data.csv')
city_data = pd.read_csv(path + 'City.csv')
customerID = pd.read_csv(path + 'Customer_ID.csv')
transacID = pd.read_csv(path + 'Transaction_ID.csv')


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
# has 50% NAN data or more is not useful because
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


# DISPERSION MEASURES FOR NUMERIC VARIABLES
def disper_table(data, col):
    table = [['std_%s' % (col[0:2]), 'var_%s' % (col[0:2]), 'cv_%s' % (col[0:2]), 'skew_%s' % (col[0:2]),
              'kurt_%s' % (col[0:2])],
             [data[col].std(ddof=0), data[col].var(ddof=0),
             stats.variation(data[col]), stats.skew(data[col]), kurtosis(data[col], fisher=True)]]

    return pd.DataFrame(table, columns=table[0]).drop([0])


# OUTLIER REMOVER
def outlier_remover(data, cuantcol, cualcol):
    for c in data[cuantcol].unique():
        Q1 = data.loc[data[cualcol] == c, [cuantcol]].quantile(0.25)
        Q3 = data.loc[data[cualcol] == c, [cuantcol]].quantile(0.75)
        IQR = Q3 - Q1
        LL = float(Q1 - 1.5*IQR)
        RL = float(Q3 + 1.5*IQR)
        cond1 = (data[cualcol] == c) & (data[cuantcol] >= LL)
        cond2 = (data[cualcol] == c) & (data[cuantcol] <= RL)
        if c == data[cualcol].unique()[0]:
            data_out = data.loc[cond1 & cond2, :]
        else:
            dummy = data.loc[cond1 & cond2, :]
            data_out = pd.concat([data_out, dummy])
        print(data_out.shape)


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

disper_table(data, 'Age')

# %%
# Distribution of the variable
data['Age'].hist(bins=15)

# Distribution of the logarithm transformation
np.log(data['Age']).hist(bins=15)


# %%
sns.boxplot(x='Age', data=data, linewidth=1.5, orient="h", palette="Set2")
sns.swarmplot(x="Age", data=data.sample(frac=.01), color=".25", alpha=0.05)


# %%

# INCOME

# Got renamed
data.rename(columns={'Income (USD/Month)': 'Income'}, inplace=True)

# %%

disper_table(data, 'Income')


# %%
# Distribution of the variable
data['Income'].hist(bins=15)

# Distribution of the logarithm transformation
np.log(data['Income']).hist(bins=15)

# %%
sns.boxplot(x='Income', data=data, linewidth=1.5, orient="h",
            palette=random.choices(colors_list, k=1))
sns.swarmplot(x="Income", data=data.sample(frac=.01), color=".25", alpha=0.05)


# %%
datalog = data.copy()
datalog['log_income'] = np.log(data['Income'])
sns.boxplot(x='log_income', data=datalog, linewidth=1.5,
            orient="h", palette=random.choices(colors_list, k=1))
sns.swarmplot(x="log_income", data=datalog.sample(
    frac=.01), color=".25", alpha=0.05)

# %%
disper_table(datalog, 'log_income')
disper_table(data, 'Income')


# %%

# COMPANY

data['Company'].describe()
data['Company'].value_counts()
tab(data, 'Company', norm=True)

# %%
pieplot(data, 'Company', title='Company Pie Chart')
cntplot(data['Company'], data, title='Company Countplot')


# %%

# CITY

data['City'].describe()
data['City'].value_counts()
tab(data, 'City', norm=True)

# %%

order = data['City'].value_counts().sort_values(ascending=False)
order.shape


# %%
sns.set_context('notebook')
sns.catplot(x=data['City'], data=data,
            kind='count', order=order.index,
            palette="viridis")
plt.xticks(rotation=90)
plt.show()


# %%

# DATE OF TRAVEL
data['Date of Travel'] = data['Date of Travel'].apply(
    lambda s: xlrd.xldate.xldate_as_datetime(s, 0))

# %%

g = sns.relplot(x=data['Date of Travel'],
                y=data['Price Charged'],
                data=data,
                kind='line')
g.fig.set_size_inches(15, 15)

# %%
# KM TRAVELLED

disper_table(data, 'KM Travelled')


# %%
# Distribution of the variable
data['Income'].hist(bins=30)


# %%
sns.boxplot(x='KM Travelled', data=data, linewidth=1.5, orient="h",
            palette=random.choices(colors_list, k=1))
sns.swarmplot(x='KM Travelled', data=data.sample(
    frac=.01), color=".25", alpha=0.05)


# %%
# PRICE CHARGED

disper_table(data, 'Price Charged')


# %%
data['Price Charged'].hist(bins=30)

# %%

sns.boxplot(x='Price Charged', data=data, linewidth=1.5, orient="h",
            palette=random.choices(colors_list, k=1))
sns.swarmplot(x='Price Charged', data=data.sample(
    frac=.01), color=".25", alpha=0.05)


# %%

# There are many outliers in the 'Price Charged' Variable, it is better to
# drop them in order to clean our data. Interquartile Range method is going to
# be used

Q1 = data['Price Charged'].quantile(0.25)
Q3 = data['Price Charged'].quantile(0.75)

IQR = Q3 - Q1

data = data[(data['Price Charged'] >= (Q1 - 1.5*IQR)) &
            (data['Price Charged'] <= (Q3 + 1.5*IQR))]

# %%
data.shape

# %%
# COST OF THE TRIP
data['Cost of Trip'].hist(bins=30)

# %%
sns.boxplot(x='Cost of Trip', data=data, linewidth=1.5, orient="h",
            palette=random.choices(colors_list, k=1))
sns.swarmplot(x='Cost of Trip', data=data.sample(
    frac=.01), color=".25", alpha=0.05)


# %%

data['Population'] = data['Population'].str.replace(',', '').astype(float)


# %%

data['Population']

# %%

data['Users'] = data['Users'].str.replace(',', '').astype(float)

# %%
data['Users']

data.groupby(by=['City', 'Company'])[['Users']].count()


# %%

data['Profit'] = data['Price Charged'] - data['Cost of Trip']
data['Profit']

# %%

h = sns.relplot(x=data['Date of Travel'],
                y=data['Profit'],
                data=data,
                kind='line')
h.fig.set_size_inches(30, 10)


# %%

# KPI CREATION

data['cost_per_km'] = data['Cost of Trip'] / data['KM Travelled']
data['cost_per_km']


# %%
cost_medians = data.groupby('City')['cost_per_km'].median()

# %%
cost_medians['ATLANTA GA']


# %%
# STANDARIZING COST PER KM FOR EACH CITY

for cit in data['City'].unique():
    data.loc[data['City'] == cit, ['cost_per_km']] = cost_medians[cit]
print(data)

# %%

# CREATING KPI
data['profit_per_km'] = data['Profit'] / data['KM Travelled']
data['profit_per_km']

# %%

# REMOVING OUTLIER
Q1 = data['profit_per_km'].quantile(0.25)
Q3 = data['profit_per_km'].quantile(0.75)

IQR = Q3 - Q1
IQR

data = data[(data['profit_per_km'] >= (Q1 - 1.5*IQR)) &
            (data['profit_per_km'] <= (Q3 + 1.5*IQR))]


# %%

# REMOVING OUTLIERS
Q1 = data['Profit'].quantile(0.25)
Q3 = data['Profit'].quantile(0.75)

IQR = Q3 - Q1
IQR

data = data[(data['Profit'] >= (Q1 - 1.5*IQR)) &
            (data['Profit'] <= (Q3 + 1.5*IQR))]


# %%


# REMOVING OUTLIERS PER CITY
for c in data['City'].unique():
    Q1 = data.loc[data['City'] == c, ['Profit']].quantile(0.25)
    Q3 = data.loc[data['City'] == c, ['Profit']].quantile(0.75)
    IQR = Q3 - Q1
    LL = float(Q1 - 1.5*IQR)
    RL = float(Q3 + 1.5*IQR)
    cond1 = (data['City'] == c) & (data['Profit'] >= LL)
    cond2 = (data['City'] == c) & (data['Profit'] <= RL)
    if c == data['City'].unique()[0]:
        outdata = data.loc[cond1 & cond2, :]
    else:
        dummy = data.loc[cond1 & cond2, :]
        outdata = pd.concat([outdata, dummy])
    print(outdata.shape)


# %%

# BOXPLOT OF PROFIT DISTRIBUTION PER CITY

order = outdata.groupby(by=["City"])["Profit"].median(
).iloc[::-1].sort_values(ascending=False).index
order
sns.boxplot(x='City', y='Profit', data=outdata, order=order)
plt.xticks(rotation=90)
plt.show()


# %%

# REMOVING OUTLIERS PER CITY

for c in outdata['City'].unique():
    Q1 = outdata.loc[outdata['City'] == c, ['profit_per_km']].quantile(0.25)
    Q3 = outdata.loc[outdata['City'] == c, ['profit_per_km']].quantile(0.75)
    IQR = Q3 - Q1
    LL = float(Q1 - 1.5*IQR)
    RL = float(Q3 + 1.5*IQR)
    cond1 = (outdata['City'] == c) & (outdata['profit_per_km'] >= LL)
    cond2 = (outdata['City'] == c) & (outdata['profit_per_km'] <= RL)
    if c == outdata['City'].unique()[0]:
        data_out = outdata.loc[cond1 & cond2, :]
    else:
        dummy = outdata.loc[cond1 & cond2, :]
        data_out = pd.concat([data_out, dummy])
    print(data_out.shape)


# %%
order = data_out.groupby(by=["City"])["profit_per_km"].median(
).iloc[::-1].sort_values(ascending=False).index
order
sns.boxplot(x='City', y='profit_per_km', data=data_out, order=order)
plt.xticks(rotation=90)
plt.show()


# %%


sns.boxplot(x='City', y='cost_per_km', data=data_out, linewidth=1.5)
plt.xticks(rotation=90)


# %%

# VARIABLES COMPARISON


# NUMERIC VS NUMERIC


# WE SAW THAT SOME VARIABLES HAVE STRONG CORRELATIONS BETWEEN THEM
# MAYBE IT WOULD BE BETTER TO SEE GRAPHICALLY THE INTERACTION OF THOSE
# VARIABLES


# For modelling erase: profit, price charged, users, population, cost of trip

# %%
# KM
for var in ['Cost of Trip', 'Profit']:
    sns.relplot(x='KM Travelled', y=var, data=data_out,
                kind='scatter')


# %%

# PRICE
for var in ['Cost of Trip']:
    sns.relplot(x='Price Charged', y=var, data=data_out,
                kind='scatter')

# %%
sns.relplot(x='KM Travelled', y='Cost of Trip', data=data_out,
            kind='scatter', hue='Company', row='Gender', col='Payment_Mode',
            palette=['Pink', 'Yellow'], alpha=0.005)


# %%
sns.relplot(x='Cost of Trip', y='Price Charged', data=data_out,
            kind='scatter', hue='Gender', row='Company', col='Payment_Mode',
            palette=random.choices(colors_list, k=2), alpha=0.005)


# PAYMENT MODE
# GENDER
# Company
#
# %%

sns.relplot(x='cost_per_km', y='profit_per_km', data=data_out,
            kind='scatter', hue='Gender', row='Company', col='Payment_Mode',
            palette=random.choices(colors_list, k=2), alpha=0.005)


# %%

# cost per km / city

sns.catplot(x='cost_per_km', y='City', data=data_out,
            kind='point', palette='viridis',
            order=data_out.groupby('City')['cost_per_km'].mean().sort_values(ascending=True).index)
plt.xticks(rotation=90)

# %%
data_out


# %%

data_num = data_out.select_dtypes(include=["number"])
data_num.corr()
sns.heatmap(data_num.corr(), annot=True)


# %%

# TIME DATA
# Crear nueva variable de estaciones
# Ver dias festivos de USA

data_num = data_num.merge(
    data_out[['Transaction ID', 'Customer ID', 'Date of Travel']], on=data_num.index, how='inner')

# %%
data_num = data_num.drop(columns=['key_0'], axis=1)

# %%
numtimedata = data_num.copy()
numtimedata = numtimedata.set_index('Date of Travel')
numtimedata


# %%


# %%

numtimedata.plot(subplots=True, figsize=(10, 12))

# %%

df_month = numtimedata.resample("M").mean()
fig, ax = plt.subplots(figsize=(10, 6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.bar(df_month['2016':].index, df_month.loc['2016':,
       "Profit"], width=25, align='center')

# %%

numtimedata


# %%

numtimedata['Month'] = pd.DatetimeIndex(numtimedata.index).month
numtimedata['Year'] = pd.DatetimeIndex(numtimedata.index).year
# %%

numtimedata.drop(columns=['Transaction ID_y'], inplace=True)

# %%
numtimedata.rename(
    columns={'Transaction ID_x': 'Transaction ID'}, inplace=True)
numtimedata

# %% WE MUST APPLYT THIS ANALYSIS ON THE CUSTOMERS THAT LIVES IN THE TOP CITIES
rfmdata = numtimedata.groupby(by=['Customer ID', 'Year']).count()[
    'Transaction ID'].sort_values(ascending=False)
TPYdata = pd.DataFrame(rfmdata)
TPYdata.rename(columns={'Transaction ID': 'TPY'}, inplace=True)
TPYdata.sort_values(['Year', 'TPY'], ascending=[False, False])


# %%
TPYdata.reset_index(level=0, inplace=True)
TPYdata
# %%
TPYdata.loc[(TPYdata['Year'] == 2018) & (TPYdata['TPY'] >= 10), :]

# %%

fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
for name, ax in zip(['Profit', 'profit_per_km', 'KM Travelled', 'Cost of Trip'], axes):
    sns.boxplot(data=numtimedata, x='Month', y=name, ax=ax)
    ax.set_ylabel("")
    ax.set_title(name)
    if ax != axes[-1]:
        ax.set_xlabel('')


# %%
df_month['Profit'].plot(figsize=(8, 6))

# %%
numtimedata.groupby(['Year', 'Month'])[
    'KM Travelled'].mean().plot(figsize=(8, 6))

# %%
sns.catplot(x='Month', kind='count',
            col='Year', data=numtimedata, palette='viridis')


# %%
df_month.loc[:, 'pct_change'] = df_month.Profit.pct_change()*100
fig, ax = plt.subplots()
df_month['pct_change'].plot(kind='bar', color='coral', ax=ax)
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
plt.xticks(rotation=45)
ax.legend()

# %%


rcParams['figure.figsize'] = 11, 9
decomposition = sm.tsa.seasonal_decompose(df_month['Profit'], model='Additive')


# %%
fig = decomposition.plot()
plt.show()

# %%


# %%
# MODELING
# modelo parta sacar el profit_per_km dado los imputs
# voy a hallar el mayor ratio de inversion
# modelo para ver que compañia preferirian
# No te olvides separar las variables de flag_target_city
#

# %%


# %%

regdata = data_num.loc[:, np.isin(data_num.columns, ['Profit', 'Price Charged', 'Users', 'Population',
                                                     'Cost of Trip'], invert=True)]

# %%
regdata.corr()
sns.heatmap(regdata.corr(), annot=True)


# Modelo para ver a que compañia irian
# flag company


# Modelo para hallar el profit dada la transaccion historica

# %%

# %%
df = df_month

df.reset_index(inplace=True)
# %%
df = df.rename(columns={'Date of Travel': 'ds', 'Profit': 'y'})
df = df.loc[:, ['ds', 'y']]
df.head()
# %%
df_prop = fbprophet.Prophet()
df_prop.fit(df)

# %%

df_forecast = df_prop.make_future_dataframe(periods=30*2, freq='D')
df_forecast = df_prop.predict(df_forecast)
