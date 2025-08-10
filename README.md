# 	Customer Experience Insights & Process Improvement 
![](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/Happy-passenger.jpg)

# Problem Analysis

## Problem Statement
- Following the pandemic, the airline industry suffered a massive setback, with ICAO estimating a 371 billion dollar loss in 2020, and a 329 billion dollar loss with reduced seat capacity. As a result, in order to revitalise the industry in the face of the current recession, it is absolutely necessary to understand the customer pain points and improve their satisfaction with the services provided.

- This data set contains a survey on air passenger satisfaction survey.Need to predict Airline passenger satisfaction level:1.Satisfaction 2.Neutral or dissatisfied.

- Select the best predictive models for predicting passengers satisfaction.

## Key Observations
- This is a binary classification problem,it is necessary to predict which of the two levels of satisfaction with the airline the passenger belongs to:Satisfaction, Neutral or dissatisfied

- Before diving into the data, thinking intuitively and being an avid traveller myself, from my experience, the main factors should be:

1. Delays in the flight

2. Staff efficiency to address customer needs

3. Services provided in the flight

# Data Gathering and Initial Insights

## Installing and Importing the required packages


```python
## Data Analysis packages
import numpy as np
import pandas as pd

## Data Visualization packages
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib
%matplotlib inline
from pylab import rcParams
import missingno as msno

## General Tools
import os
import re
import joblib
import json
import warnings


# sklearn library
import sklearn

### sklearn preprocessing tools
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,accuracy_score,roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer,FunctionTransformer,OneHotEncoder


# Error Metrics 
from sklearn.metrics import r2_score #r2 square
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score


### Machine learning classification Models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier #stacstic gradient descent clasifeier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier


#crossvalidation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
#from sklearn.metrics import plot_confusion_matrix


#hyper parameter tunning
from sklearn.model_selection import GridSearchCV,cross_val_score,RandomizedSearchCV
```

## Downloading the dataset

- The dataset is from Kaggle. it provides cutting-edge data science, faster and better than most people ever thought possible. Kaggle offers both public and private data science competitions and on-demand consulting by an elite global talent pool.
- When you execute od.download, you will be asked to provide your Kaggle username and API key. Follow these instructions to create an API key: http://bit.ly/kaggle-creds
- Dataset link https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

## Information about the dataset


There is the following information about the passengers of some airline:

1. **Gender:** male or female
2. **Customer type:** regular or non-regular airline customer
3. **Age**: the actual age of the passenger
4. **Type of travel:** the purpose of the passenger's flight (personal or business travel)
5. **Class:** business, economy, economy plus
6. **Flight distance**
7. **Inflight wifi service:** satisfaction level with Wi-Fi service on board (0: not rated; 1-5)
8. **Departure/Arrival time convenient:** departure/arrival time satisfaction level (0: not rated; 1-5)
9. **Ease of Online booking:** online booking satisfaction rate (0: not rated; 1-5)
10. **Gate location:** level of satisfaction with the gate location (0: not rated; 1-5)
11. **Food and drink:** food and drink satisfaction level (0: not rated; 1-5)
12. **Online boarding:** satisfaction level with online boarding (0: not rated; 1-5)
13. **Seat comfort:** seat satisfaction level (0: not rated; 1-5)
14. Inflight entertainment: satisfaction with inflight entertainment (0: not rated; 1-5)
15. **On-board service:** level of satisfaction with on-board service (0: not rated; 1-5)
16. **Leg room service:** level of satisfaction with leg room service (0: not rated; 1-5)
17. **Baggage handling:** level of satisfaction with baggage handling (0: not rated; 1-5)
18. **Checkin service:** level of satisfaction with checkin service (0: not rated; 1-5)
19. **Inflight service:** level of satisfaction with inflight service (0: not rated; 1-5)
20. **Cleanliness:** level of satisfaction with cleanliness (0: not rated; 1-5)
21. **Departure delay in minutes:**
22. **Arrival delay in minutes:**
23. **Satisfaction:** Airline satisfaction level(Satisfaction, neutral or dissatisfaction).

**Train Dataset**


```python
train_df = pd.read_csv("./data/train.csv")
train_df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>Gender</th>
      <th>Customer Type</th>
      <th>Age</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>...</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
      <th>satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>70172</td>
      <td>Male</td>
      <td>Loyal Customer</td>
      <td>13</td>
      <td>Personal Travel</td>
      <td>Eco Plus</td>
      <td>460</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>25</td>
      <td>18.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5047</td>
      <td>Male</td>
      <td>disloyal Customer</td>
      <td>25</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>235</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>6.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>110028</td>
      <td>Female</td>
      <td>Loyal Customer</td>
      <td>26</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>1142</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>24026</td>
      <td>Female</td>
      <td>Loyal Customer</td>
      <td>25</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>562</td>
      <td>2</td>
      <td>5</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>11</td>
      <td>9.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>119299</td>
      <td>Male</td>
      <td>Loyal Customer</td>
      <td>61</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>214</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>satisfied</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
## Initial Statistical description

train_df.describe()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>Age</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>Gate location</th>
      <th>Food and drink</th>
      <th>Online boarding</th>
      <th>Seat comfort</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103594.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>51951.500000</td>
      <td>64924.210502</td>
      <td>39.379706</td>
      <td>1189.448375</td>
      <td>2.729683</td>
      <td>3.060296</td>
      <td>2.756901</td>
      <td>2.976883</td>
      <td>3.202129</td>
      <td>3.250375</td>
      <td>3.439396</td>
      <td>3.358158</td>
      <td>3.382363</td>
      <td>3.351055</td>
      <td>3.631833</td>
      <td>3.304290</td>
      <td>3.640428</td>
      <td>3.286351</td>
      <td>14.815618</td>
      <td>15.178678</td>
    </tr>
    <tr>
      <th>std</th>
      <td>29994.645522</td>
      <td>37463.812252</td>
      <td>15.114964</td>
      <td>997.147281</td>
      <td>1.327829</td>
      <td>1.525075</td>
      <td>1.398929</td>
      <td>1.277621</td>
      <td>1.329533</td>
      <td>1.349509</td>
      <td>1.319088</td>
      <td>1.332991</td>
      <td>1.288354</td>
      <td>1.315605</td>
      <td>1.180903</td>
      <td>1.265396</td>
      <td>1.175663</td>
      <td>1.312273</td>
      <td>38.230901</td>
      <td>38.698682</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>31.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25975.750000</td>
      <td>32533.750000</td>
      <td>27.000000</td>
      <td>414.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>51951.500000</td>
      <td>64856.500000</td>
      <td>40.000000</td>
      <td>843.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>77927.250000</td>
      <td>97368.250000</td>
      <td>51.000000</td>
      <td>1743.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>12.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>103903.000000</td>
      <td>129880.000000</td>
      <td>85.000000</td>
      <td>4983.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1592.000000</td>
      <td>1584.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Observations**
- The average delay in flights are 15 minutes, with a deviation of 38
- Median of the delays are 0, which means 50% of the flights from this data, were not delayed


```python
## removing the first two columns
train_df.drop(["Unnamed: 0", 'id'], axis=1, inplace=True)
```


```python
train_df.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Customer Type</th>
      <th>Age</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>Gate location</th>
      <th>...</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
      <th>satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>Loyal Customer</td>
      <td>13</td>
      <td>Personal Travel</td>
      <td>Eco Plus</td>
      <td>460</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>25</td>
      <td>18.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>disloyal Customer</td>
      <td>25</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>235</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>6.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 23 columns</p>
</div>




```python
## shape of the train dataset
train_df.shape
```

    (103904, 23)

```python
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 103904 entries, 0 to 103903
    Data columns (total 23 columns):
     #   Column                             Non-Null Count   Dtype  
    ---  ------                             --------------   -----  
     0   Gender                             103904 non-null  object 
     1   Customer Type                      103904 non-null  object 
     2   Age                                103904 non-null  int64  
     3   Type of Travel                     103904 non-null  object 
     4   Class                              103904 non-null  object 
     5   Flight Distance                    103904 non-null  int64  
     6   Inflight wifi service              103904 non-null  int64  
     7   Departure/Arrival time convenient  103904 non-null  int64  
     8   Ease of Online booking             103904 non-null  int64  
     9   Gate location                      103904 non-null  int64  
     10  Food and drink                     103904 non-null  int64  
     11  Online boarding                    103904 non-null  int64  
     12  Seat comfort                       103904 non-null  int64  
     13  Inflight entertainment             103904 non-null  int64  
     14  On-board service                   103904 non-null  int64  
     15  Leg room service                   103904 non-null  int64  
     16  Baggage handling                   103904 non-null  int64  
     17  Checkin service                    103904 non-null  int64  
     18  Inflight service                   103904 non-null  int64  
     19  Cleanliness                        103904 non-null  int64  
     20  Departure Delay in Minutes         103904 non-null  int64  
     21  Arrival Delay in Minutes           103594 non-null  float64
     22  satisfaction                       103904 non-null  object 
    dtypes: float64(1), int64(17), object(5)
    memory usage: 18.2+ MB
    

- Only Arrival Delay in Minutes has null values. Lets visualize to see any patterns in the missing values




```python
msno.matrix(train_df)
```

![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_20_1.png)
    


**Observations**
- There are 103904 rows for 23 features in our data
- we see in the training data, that all the datatypes belongs to a numeric class that is int, float and object
- only arrival delay in minutes have some null values


```python
# percentage of null values

train_df.isnull().sum()
```
    Gender                                 0
    Customer Type                          0
    Age                                    0
    Type of Travel                         0
    Class                                  0
    Flight Distance                        0
    Inflight wifi service                  0
    Departure/Arrival time convenient      0
    Ease of Online booking                 0
    Gate location                          0
    Food and drink                         0
    Online boarding                        0
    Seat comfort                           0
    Inflight entertainment                 0
    On-board service                       0
    Leg room service                       0
    Baggage handling                       0
    Checkin service                        0
    Inflight service                       0
    Cleanliness                            0
    Departure Delay in Minutes             0
    Arrival Delay in Minutes             310
    satisfaction                           0
    dtype: int64



- The number of null values is 310 in "Arrival Delay in Minutes" column
- The percentage of null values is ~ 0.3%


```python
round(train_df.describe().T, 2)
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>103904.0</td>
      <td>39.38</td>
      <td>15.11</td>
      <td>7.0</td>
      <td>27.0</td>
      <td>40.0</td>
      <td>51.0</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>Flight Distance</th>
      <td>103904.0</td>
      <td>1189.45</td>
      <td>997.15</td>
      <td>31.0</td>
      <td>414.0</td>
      <td>843.0</td>
      <td>1743.0</td>
      <td>4983.0</td>
    </tr>
    <tr>
      <th>Inflight wifi service</th>
      <td>103904.0</td>
      <td>2.73</td>
      <td>1.33</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Departure/Arrival time convenient</th>
      <td>103904.0</td>
      <td>3.06</td>
      <td>1.53</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Ease of Online booking</th>
      <td>103904.0</td>
      <td>2.76</td>
      <td>1.40</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Gate location</th>
      <td>103904.0</td>
      <td>2.98</td>
      <td>1.28</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Food and drink</th>
      <td>103904.0</td>
      <td>3.20</td>
      <td>1.33</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Online boarding</th>
      <td>103904.0</td>
      <td>3.25</td>
      <td>1.35</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Seat comfort</th>
      <td>103904.0</td>
      <td>3.44</td>
      <td>1.32</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Inflight entertainment</th>
      <td>103904.0</td>
      <td>3.36</td>
      <td>1.33</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>On-board service</th>
      <td>103904.0</td>
      <td>3.38</td>
      <td>1.29</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Leg room service</th>
      <td>103904.0</td>
      <td>3.35</td>
      <td>1.32</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Baggage handling</th>
      <td>103904.0</td>
      <td>3.63</td>
      <td>1.18</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Checkin service</th>
      <td>103904.0</td>
      <td>3.30</td>
      <td>1.27</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Inflight service</th>
      <td>103904.0</td>
      <td>3.64</td>
      <td>1.18</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Cleanliness</th>
      <td>103904.0</td>
      <td>3.29</td>
      <td>1.31</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Departure Delay in Minutes</th>
      <td>103904.0</td>
      <td>14.82</td>
      <td>38.23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>1592.0</td>
    </tr>
    <tr>
      <th>Arrival Delay in Minutes</th>
      <td>103594.0</td>
      <td>15.18</td>
      <td>38.70</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>1584.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
# Duplicate values
train_df.duplicated().sum()
```
    0

```python
# target variable
train_df.satisfaction.value_counts()[1]/len(train_df.satisfaction)*100
```

    43.333269171542966

 - This problem is a binary classification problem of classes 0 or 1 denoting customer satisfaction, The class 1 has 43.33% of total values. Hence, this is a balanced learning problem. hence will not be requiring any resampling techniques to tackle this

## Independent Variables or features**


```python
train_df.columns[:-1]
```

    Index(['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
           'Flight Distance', 'Inflight wifi service',
           'Departure/Arrival time convenient', 'Ease of Online booking',
           'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
           'Inflight entertainment', 'On-board service', 'Leg room service',
           'Baggage handling', 'Checkin service', 'Inflight service',
           'Cleanliness', 'Departure Delay in Minutes',
           'Arrival Delay in Minutes'],
          dtype='object')

```python

```

# Exploratory Data Analysis and Visualization

Before training a machine learning model, it's always a good idea to explore the distributions of various columns and see how they are related to the target column. Let's explore and visualize the data using the Plotly, Matplotlib and Seaborn libraries. 


```python
train_df.corr(numeric_only= True)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>Gate location</th>
      <th>Food and drink</th>
      <th>Online boarding</th>
      <th>Seat comfort</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>0.099461</td>
      <td>0.017859</td>
      <td>0.038125</td>
      <td>0.024842</td>
      <td>-0.001330</td>
      <td>0.023000</td>
      <td>0.208939</td>
      <td>0.160277</td>
      <td>0.076444</td>
      <td>0.057594</td>
      <td>0.040583</td>
      <td>-0.047529</td>
      <td>0.035482</td>
      <td>-0.049427</td>
      <td>0.053611</td>
      <td>-0.010152</td>
      <td>-0.012147</td>
    </tr>
    <tr>
      <th>Flight Distance</th>
      <td>0.099461</td>
      <td>1.000000</td>
      <td>0.007131</td>
      <td>-0.020043</td>
      <td>0.065717</td>
      <td>0.004793</td>
      <td>0.056994</td>
      <td>0.214869</td>
      <td>0.157333</td>
      <td>0.128740</td>
      <td>0.109526</td>
      <td>0.133916</td>
      <td>0.063184</td>
      <td>0.073072</td>
      <td>0.057540</td>
      <td>0.093149</td>
      <td>0.002158</td>
      <td>-0.002426</td>
    </tr>
    <tr>
      <th>Inflight wifi service</th>
      <td>0.017859</td>
      <td>0.007131</td>
      <td>1.000000</td>
      <td>0.343845</td>
      <td>0.715856</td>
      <td>0.336248</td>
      <td>0.134718</td>
      <td>0.456970</td>
      <td>0.122658</td>
      <td>0.209321</td>
      <td>0.121500</td>
      <td>0.160473</td>
      <td>0.120923</td>
      <td>0.043193</td>
      <td>0.110441</td>
      <td>0.132698</td>
      <td>-0.017402</td>
      <td>-0.019095</td>
    </tr>
    <tr>
      <th>Departure/Arrival time convenient</th>
      <td>0.038125</td>
      <td>-0.020043</td>
      <td>0.343845</td>
      <td>1.000000</td>
      <td>0.436961</td>
      <td>0.444757</td>
      <td>0.004906</td>
      <td>0.070119</td>
      <td>0.011344</td>
      <td>-0.004861</td>
      <td>0.068882</td>
      <td>0.012441</td>
      <td>0.072126</td>
      <td>0.093333</td>
      <td>0.073318</td>
      <td>0.014292</td>
      <td>0.001005</td>
      <td>-0.000864</td>
    </tr>
    <tr>
      <th>Ease of Online booking</th>
      <td>0.024842</td>
      <td>0.065717</td>
      <td>0.715856</td>
      <td>0.436961</td>
      <td>1.000000</td>
      <td>0.458655</td>
      <td>0.031873</td>
      <td>0.404074</td>
      <td>0.030014</td>
      <td>0.047032</td>
      <td>0.038833</td>
      <td>0.107601</td>
      <td>0.038762</td>
      <td>0.011081</td>
      <td>0.035272</td>
      <td>0.016179</td>
      <td>-0.006371</td>
      <td>-0.007984</td>
    </tr>
    <tr>
      <th>Gate location</th>
      <td>-0.001330</td>
      <td>0.004793</td>
      <td>0.336248</td>
      <td>0.444757</td>
      <td>0.458655</td>
      <td>1.000000</td>
      <td>-0.001159</td>
      <td>0.001688</td>
      <td>0.003669</td>
      <td>0.003517</td>
      <td>-0.028373</td>
      <td>-0.005873</td>
      <td>0.002313</td>
      <td>-0.035427</td>
      <td>0.001681</td>
      <td>-0.003830</td>
      <td>0.005467</td>
      <td>0.005143</td>
    </tr>
    <tr>
      <th>Food and drink</th>
      <td>0.023000</td>
      <td>0.056994</td>
      <td>0.134718</td>
      <td>0.004906</td>
      <td>0.031873</td>
      <td>-0.001159</td>
      <td>1.000000</td>
      <td>0.234468</td>
      <td>0.574556</td>
      <td>0.622512</td>
      <td>0.059073</td>
      <td>0.032498</td>
      <td>0.034746</td>
      <td>0.087299</td>
      <td>0.033993</td>
      <td>0.657760</td>
      <td>-0.029926</td>
      <td>-0.032524</td>
    </tr>
    <tr>
      <th>Online boarding</th>
      <td>0.208939</td>
      <td>0.214869</td>
      <td>0.456970</td>
      <td>0.070119</td>
      <td>0.404074</td>
      <td>0.001688</td>
      <td>0.234468</td>
      <td>1.000000</td>
      <td>0.420211</td>
      <td>0.285066</td>
      <td>0.155443</td>
      <td>0.123950</td>
      <td>0.083280</td>
      <td>0.204462</td>
      <td>0.074573</td>
      <td>0.331517</td>
      <td>-0.018982</td>
      <td>-0.021949</td>
    </tr>
    <tr>
      <th>Seat comfort</th>
      <td>0.160277</td>
      <td>0.157333</td>
      <td>0.122658</td>
      <td>0.011344</td>
      <td>0.030014</td>
      <td>0.003669</td>
      <td>0.574556</td>
      <td>0.420211</td>
      <td>1.000000</td>
      <td>0.610590</td>
      <td>0.131971</td>
      <td>0.105559</td>
      <td>0.074542</td>
      <td>0.191854</td>
      <td>0.069218</td>
      <td>0.678534</td>
      <td>-0.027898</td>
      <td>-0.029900</td>
    </tr>
    <tr>
      <th>Inflight entertainment</th>
      <td>0.076444</td>
      <td>0.128740</td>
      <td>0.209321</td>
      <td>-0.004861</td>
      <td>0.047032</td>
      <td>0.003517</td>
      <td>0.622512</td>
      <td>0.285066</td>
      <td>0.610590</td>
      <td>1.000000</td>
      <td>0.420153</td>
      <td>0.299692</td>
      <td>0.378210</td>
      <td>0.120867</td>
      <td>0.404855</td>
      <td>0.691815</td>
      <td>-0.027489</td>
      <td>-0.030703</td>
    </tr>
    <tr>
      <th>On-board service</th>
      <td>0.057594</td>
      <td>0.109526</td>
      <td>0.121500</td>
      <td>0.068882</td>
      <td>0.038833</td>
      <td>-0.028373</td>
      <td>0.059073</td>
      <td>0.155443</td>
      <td>0.131971</td>
      <td>0.420153</td>
      <td>1.000000</td>
      <td>0.355495</td>
      <td>0.519134</td>
      <td>0.243914</td>
      <td>0.550782</td>
      <td>0.123220</td>
      <td>-0.031569</td>
      <td>-0.035227</td>
    </tr>
    <tr>
      <th>Leg room service</th>
      <td>0.040583</td>
      <td>0.133916</td>
      <td>0.160473</td>
      <td>0.012441</td>
      <td>0.107601</td>
      <td>-0.005873</td>
      <td>0.032498</td>
      <td>0.123950</td>
      <td>0.105559</td>
      <td>0.299692</td>
      <td>0.355495</td>
      <td>1.000000</td>
      <td>0.369544</td>
      <td>0.153137</td>
      <td>0.368656</td>
      <td>0.096370</td>
      <td>0.014363</td>
      <td>0.011843</td>
    </tr>
    <tr>
      <th>Baggage handling</th>
      <td>-0.047529</td>
      <td>0.063184</td>
      <td>0.120923</td>
      <td>0.072126</td>
      <td>0.038762</td>
      <td>0.002313</td>
      <td>0.034746</td>
      <td>0.083280</td>
      <td>0.074542</td>
      <td>0.378210</td>
      <td>0.519134</td>
      <td>0.369544</td>
      <td>1.000000</td>
      <td>0.233122</td>
      <td>0.628561</td>
      <td>0.095793</td>
      <td>-0.005573</td>
      <td>-0.008542</td>
    </tr>
    <tr>
      <th>Checkin service</th>
      <td>0.035482</td>
      <td>0.073072</td>
      <td>0.043193</td>
      <td>0.093333</td>
      <td>0.011081</td>
      <td>-0.035427</td>
      <td>0.087299</td>
      <td>0.204462</td>
      <td>0.191854</td>
      <td>0.120867</td>
      <td>0.243914</td>
      <td>0.153137</td>
      <td>0.233122</td>
      <td>1.000000</td>
      <td>0.237197</td>
      <td>0.179583</td>
      <td>-0.018453</td>
      <td>-0.020369</td>
    </tr>
    <tr>
      <th>Inflight service</th>
      <td>-0.049427</td>
      <td>0.057540</td>
      <td>0.110441</td>
      <td>0.073318</td>
      <td>0.035272</td>
      <td>0.001681</td>
      <td>0.033993</td>
      <td>0.074573</td>
      <td>0.069218</td>
      <td>0.404855</td>
      <td>0.550782</td>
      <td>0.368656</td>
      <td>0.628561</td>
      <td>0.237197</td>
      <td>1.000000</td>
      <td>0.088779</td>
      <td>-0.054813</td>
      <td>-0.059196</td>
    </tr>
    <tr>
      <th>Cleanliness</th>
      <td>0.053611</td>
      <td>0.093149</td>
      <td>0.132698</td>
      <td>0.014292</td>
      <td>0.016179</td>
      <td>-0.003830</td>
      <td>0.657760</td>
      <td>0.331517</td>
      <td>0.678534</td>
      <td>0.691815</td>
      <td>0.123220</td>
      <td>0.096370</td>
      <td>0.095793</td>
      <td>0.179583</td>
      <td>0.088779</td>
      <td>1.000000</td>
      <td>-0.014093</td>
      <td>-0.015774</td>
    </tr>
    <tr>
      <th>Departure Delay in Minutes</th>
      <td>-0.010152</td>
      <td>0.002158</td>
      <td>-0.017402</td>
      <td>0.001005</td>
      <td>-0.006371</td>
      <td>0.005467</td>
      <td>-0.029926</td>
      <td>-0.018982</td>
      <td>-0.027898</td>
      <td>-0.027489</td>
      <td>-0.031569</td>
      <td>0.014363</td>
      <td>-0.005573</td>
      <td>-0.018453</td>
      <td>-0.054813</td>
      <td>-0.014093</td>
      <td>1.000000</td>
      <td>0.965481</td>
    </tr>
    <tr>
      <th>Arrival Delay in Minutes</th>
      <td>-0.012147</td>
      <td>-0.002426</td>
      <td>-0.019095</td>
      <td>-0.000864</td>
      <td>-0.007984</td>
      <td>0.005143</td>
      <td>-0.032524</td>
      <td>-0.021949</td>
      <td>-0.029900</td>
      <td>-0.030703</td>
      <td>-0.035227</td>
      <td>0.011843</td>
      <td>-0.008542</td>
      <td>-0.020369</td>
      <td>-0.059196</td>
      <td>-0.015774</td>
      <td>0.965481</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(20, 10))
sns.heatmap(train_df.corr(numeric_only= True), annot=True, vmax=1, cmap='coolwarm')
plt.show()
```


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_34_0.png)
    


- departure delay in minutes and arrival dalay in minutes are highly co-related!

## Data distribution graphs


```python
sns.set(rc={
    "font.size":15,
    "axes.titlesize":10,
    "axes.labelsize":15},
    style="darkgrid")
fig, axs = plt.subplots(6, 3, figsize=(20,30))
fig.tight_layout(pad=4.0)

for f, ax in zip(train_df, axs.ravel()):
    sns.set(font_scale = 2)
    ax = sns.histplot(ax=ax, data=train_df, x=train_df[f], kde=True, color='purple')
    ax.set_title(f)
```


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_37_0.png)
    

```python

```

## Piechart perrcentage distribution features


```python
new_train_df = train_df.copy()
```


```python
new_train_df.drop(['Age','Flight Distance','Departure Delay in Minutes', 'Arrival Delay in Minutes','satisfaction'], axis=1, inplace=True)
```


```python
sns.set(rc={
            "font.size":10,
            "axes.titlesize":10,
            "axes.labelsize":13},
             style="darkgrid")
fig, axes = plt.subplots(6, 3, figsize = (20, 30))
for i, col in enumerate(new_train_df):
    column_values = new_train_df[col].value_counts()
    labels = column_values.index
    sizes = column_values.values
    axes[i//3, i%3].pie(sizes,labels = labels, colors = sns.color_palette("RdGy_r"),autopct = '%1.0f%%', startangle = 90)
    axes[i//3, i%3].axis('equal')
    axes[i//3, i%3].set_title(col)
plt.show()
```


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_42_0.png)
    


**Observations:**
- The number of men and women in this sample is approximately the same
- The vast majority of the airline's customers are repeat customers
- Most of the clients flew for business rather than personal reasons
- About half of the passengers were in business class
- More than 60% of passengers were satisfied with the luggage transportation service(rated 4-5 out of 5)
- More than 50% of pessengers were compfortable sitting in thier seats(rated 4-5 out of 5)



```python
## Satisfaction
```


```python
train_df.satisfaction.value_counts()
```

    neutral or dissatisfied    58879
    satisfied                  45025
    Name: satisfaction, dtype: int64

```python
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,6))
train_df.satisfaction.value_counts().plot.pie(explode=(0, 0.05), colors=sns.color_palette("RdYlBu"),autopct='%1.1f%%',ax=ax1)
ax1.set_title("Percentage of Satisfaction")
sns.countplot(x= "satisfaction", data=train_df, ax=ax2, palette='RdYlBu')
ax2.set_title("Distribution of Satisfaction")

```




    Text(0.5, 1.0, 'Distribution of Satisfaction')




    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_47_1.png)
    


**Observation:**
- As per the given data, 56.7% people are dissatisfied and neutral
- And 43.3 people are satisfied


To analyse and visualise the data lets divide data columns into categorical and numerical columns.


```python
# numerical and categorical features
numerical_cols = train_df.select_dtypes(include=np.number).columns.to_list()
categorical_cols = train_df.select_dtypes('object').columns.to_list()
```


```python
#numerical columns
print("Total number of columns are:",len(numerical_cols))
print(numerical_cols)
```

    Total number of columns are: 18
    ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    


```python
#Categorical Columns
print("Total number of columns are:",len(categorical_cols))
print(categorical_cols)
```

    Total number of columns are: 5
    ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    


```python
categorical_cols.remove('satisfaction')
```


```python

```

# Exploratory Data Analysis and Visualization on Numerical Columns


```python
sns.set(rc={
            "font.size":10,
            "axes.titlesize":10,
            "axes.labelsize":15},
             style="darkgrid",
            )
fig, axs = plt.subplots(6, 3, figsize=(15, 30))
fig.tight_layout(pad=3.0)

for f, ax in zip(numerical_cols, axs.ravel()):
    sns.set(font_scale=2)
    ax= sns.boxplot(ax=ax, data=train_df, y=train_df[f], palette='BuGn')
```


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_56_0.png)
    


**Observations:**

Flight distance, checkin service, Departure Delay in minutes, Arrival delay in minutes has some outliers

### Barplot representation of numerical features


```python
sns.set(rc={'figure.figsize':(8,6),
            "font.size":10,
            "axes.titlesize":10,
            "axes.labelsize":15},
             style="darkgrid")

for col in numerical_cols:
    sns.barplot(data=train_df, x="satisfaction", y=col, palette='BuGn')
    plt.show()
```


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_0.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_1.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_2.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_3.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_4.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_5.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_6.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_7.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_8.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_9.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_10.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_11.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_12.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_13.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_14.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_15.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_16.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_60_17.png)
    


**Observations:**

- From above graphs,it is clear that the age and Gate location, does not play a huge role in flight satisfaction.
- And also the gender does not tell us mush as seen in the earlier plot. hence we can rop these values 



# Exploratory Data Analysis and Visualization on Categorical Column

### Barplot representaion on Categorical Column


```python
sns.set(rc={'figure.figsize':(11.7,8.27),
            "font.size":10,
            "axes.titlesize":10,
            "axes.labelsize":15},
             style="darkgrid",
            )
for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=train_df,x=col, hue='satisfaction', palette='PuRd_r')
    plt.legend(loc=(1.05,0.5))
    
```


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_65_0.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_65_1.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_65_2.png)
    



    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_65_3.png)
    


**Observations:**

- Gender doesn't play an important role in the satisfaction, as men and women seems to equally concerned about the same factors
- Number of loyal customers for this airline is high, however, the dissatisfaction level is high irrespective of the loyalty. Airline will have to work on maintaining the loyal customers
- Business Travellers seems to be more satisfied with the flight, than the personal travellers
- People in business class seems to be the most satisfied lot, and those in economy class are least satisfied

## Arrival Delay in Minutes VS Departure Delay in minutes.


```python
train_df.groupby('satisfaction')['Arrival Delay in Minutes'].mean()
```

    satisfaction
    neutral or dissatisfied    17.127536
    satisfied                  12.630799
    Name: Arrival Delay in Minutes, dtype: float64


```python
sns.set(rc={
            "font.size":10,
            "axes.titlesize":10,
            "axes.labelsize":13},
             style="darkgrid")
plt.figure(figsize=(10, 5), dpi=100)
sns.scatterplot(data=train_df, x="Arrival Delay in Minutes", y= "Departure Delay in Minutes", hue='satisfaction', palette="magma_r",alpha=0.8)
```




    <Axes: xlabel='Arrival Delay in Minutes', ylabel='Departure Delay in Minutes'>




    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_70_1.png)
    


**Observations:**

The arrival and departure delay seems to have a linear relationship, which makes complete sense! And well, there is 1 customer who was satisfied even after a delay of 1300 minutes!!

## Flight distance vs Departure Delay in Minutes


```python
sns.set(rc={
            "font.size":10,
            "axes.titlesize":10,
            "axes.labelsize":13},
             style="darkgrid")
plt.figure(figsize=(10, 5), dpi=100)
sns.scatterplot(data=train_df, x="Flight Distance", y= "Departure Delay in Minutes", hue='satisfaction', palette="magma_r",alpha=0.8)
plt.ylim(0,1000)
```

    (0.0, 1000.0)

![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_73_1.png)
    


**Observations:**
- The most important takeaway here is the longer the flight distance, most passengers are okay with flight delay in departure, which is strance finding from this plot!
- So departure delay is less of a factor for a long distance flight, comparitively, however, short distance travellers does not seem to be excited about the departure delays, which also makes sense

## Age and Customer type


```python
f, ax = plt.subplots(1,2, figsize=(15, 5))
sns.boxplot(data=train_df, x="Customer Type", y= "Age",palette = "gnuplot2_r", ax=ax[0])
sns.histplot(data=train_df, x="Age", hue="Customer Type", multiple="stack", palette = "gnuplot2_r",edgecolor = ".3", linewidth = .5, ax = ax[1])
```




    <Axes: xlabel='Age', ylabel='Count'>




    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_76_1.png)
    


**Observations:**
- From above we can conclude that most of the airline's regular customers are between the ages of 30 and 50(their average age is slightly above 40)
- The age range of non-regular customers is slightly smaller (from 25 to 40 years old, on average - a little less than 30).

## Age vs Class


```python
f, ax  =plt.subplots(1,2,figsize=(15,5))
sns.boxplot(data=train_df, x="Class", y="Age",palette = "gnuplot2_r", ax=ax[0])
sns.histplot(data=train_df, x="Age", hue="Class", multiple="stack", palette="gnuplot2_r",edgecolor = ".3", linewidth = .5, ax = ax[1])
```




    <Axes: xlabel='Age', ylabel='Count'>




    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_79_1.png)
    


- It can be seen that, on average, the age range of those customers who travel in business class is the same (according to the previous box chart) as the age range of regular customers. Based on this observation, it can be assumed that regular customers mainly buy business class for themselves.


```python
f, ax = plt.subplots(1, 2, figsize = (15,5))
sns.boxplot(x = "Class", y = "Flight Distance", palette = "gnuplot2_r", data = train_df, ax = ax[0])
sns.histplot(train_df, x = "Flight Distance", hue = "Class", multiple = "stack", palette = "gnuplot2_r", edgecolor = ".3", linewidth = .5, ax = ax[1])
```




    <Axes: xlabel='Flight Distance', ylabel='Count'>




    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_81_1.png)
    


**Observations:**

- customers whose flight distance is long, mostly fly in business class.

## Flight Distance


```python
f,ax = plt.subplots(2,2, figsize=(15, 8))
sns.boxplot(x = "Inflight entertainment", y = "Flight Distance", palette = "gnuplot2_r", data = train_df, ax = ax[0, 0])
sns.histplot(train_df, x = "Flight Distance", hue = "Inflight entertainment", multiple = "stack", palette = "gnuplot2_r", edgecolor = ".3", linewidth = .5, ax = ax[0, 1])
sns.boxplot(x = "Leg room service", y = "Flight Distance", palette = "gnuplot2_r", data = train_df, ax = ax[1, 0])
sns.histplot(train_df, x = "Flight Distance", hue = "Leg room service", multiple = "stack", palette = "gnuplot2_r", edgecolor = ".3", linewidth = .5, ax = ax[1, 1])
```




    <Axes: xlabel='Flight Distance', ylabel='Count'>




    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_84_1.png)
    


**Observations:**

- The more distance an aircraft passenger travels (respectively, the longer they are in flight)
- The more they are satisfied with the entertainment in flight and the extra legroom (on average).

# Data preprocessing and Feature engineering


```python
input_cols = list(train_df.iloc[:, :-1])
target_cols = "satisfaction"
```


```python
pd.options.display.max_columns=30
```


```python
train_df.head()
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Customer Type</th>
      <th>Age</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>Gate location</th>
      <th>Food and drink</th>
      <th>Online boarding</th>
      <th>Seat comfort</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
      <th>satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>Loyal Customer</td>
      <td>13</td>
      <td>Personal Travel</td>
      <td>Eco Plus</td>
      <td>460</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>25</td>
      <td>18.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>disloyal Customer</td>
      <td>25</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>235</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>6.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Female</td>
      <td>Loyal Customer</td>
      <td>26</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>1142</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>Loyal Customer</td>
      <td>25</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>562</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>11</td>
      <td>9.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>Loyal Customer</td>
      <td>61</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>214</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>satisfied</td>
    </tr>
  </tbody>
</table>
</div>



```python
train_df["Gender"] = pd.get_dummies(train_df["Gender"], drop_first=True, dtype="int")
```


```python
train_df["Customer Type"]= pd.get_dummies(train_df["Customer Type"], drop_first=True, dtype="int")
```


```python
train_df["Type of Travel"]= pd.get_dummies(train_df["Type of Travel"], drop_first=True, dtype="int")
```


```python
from sklearn.preprocessing import LabelEncoder
```


```python
le = LabelEncoder()
```


```python
train_df["Class"]=le.fit_transform(train_df["Class"])
```


```python
train_df["Class"]
```




    0         2
    1         0
    2         0
    3         0
    4         0
             ..
    103899    1
    103900    0
    103901    0
    103902    1
    103903    0
    Name: Class, Length: 103904, dtype: int32




```python
train_df["Arrival Delay in Minutes"]
```

    0         18.0
    1          6.0
    2          0.0
    3          9.0
    4          0.0
              ... 
    103899     0.0
    103900     0.0
    103901    14.0
    103902     0.0
    103903     0.0
    Name: Arrival Delay in Minutes, Length: 103904, dtype: float64




```python
from sklearn.impute import SimpleImputer
```


```python
median=train_df["Arrival Delay in Minutes"].median()
```


```python
train_df["Arrival Delay in Minutes"].fillna(median, inplace=True)
```


```python
train_df.isnull().sum()
```

    Gender                               0
    Customer Type                        0
    Age                                  0
    Type of Travel                       0
    Class                                0
    Flight Distance                      0
    Inflight wifi service                0
    Departure/Arrival time convenient    0
    Ease of Online booking               0
    Gate location                        0
    Food and drink                       0
    Online boarding                      0
    Seat comfort                         0
    Inflight entertainment               0
    On-board service                     0
    Leg room service                     0
    Baggage handling                     0
    Checkin service                      0
    Inflight service                     0
    Cleanliness                          0
    Departure Delay in Minutes           0
    Arrival Delay in Minutes             0
    satisfaction                         0
    dtype: int64




```python
train_df
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Customer Type</th>
      <th>Age</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>Gate location</th>
      <th>Food and drink</th>
      <th>Online boarding</th>
      <th>Seat comfort</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
      <th>satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>1</td>
      <td>2</td>
      <td>460</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>25</td>
      <td>18.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>235</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>6.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>1142</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>562</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>11</td>
      <td>9.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>214</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>103899</th>
      <td>0</td>
      <td>1</td>
      <td>23</td>
      <td>0</td>
      <td>1</td>
      <td>192</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>103900</th>
      <td>1</td>
      <td>0</td>
      <td>49</td>
      <td>0</td>
      <td>0</td>
      <td>2347</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>103901</th>
      <td>1</td>
      <td>1</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>1995</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>7</td>
      <td>14.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>103902</th>
      <td>0</td>
      <td>1</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>1000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>103903</th>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>1723</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
  </tbody>
</table>
<p>103904 rows × 23 columns</p>
</div>




```python
train_df["satisfaction"] = le.fit_transform(train_df["satisfaction"])
```


```python
train_df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Customer Type</th>
      <th>Age</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>Gate location</th>
      <th>Food and drink</th>
      <th>Online boarding</th>
      <th>Seat comfort</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
      <th>satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>1</td>
      <td>2</td>
      <td>460</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>25</td>
      <td>18.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>235</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>6.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>1142</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>562</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>11</td>
      <td>9.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>214</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>103899</th>
      <td>0</td>
      <td>1</td>
      <td>23</td>
      <td>0</td>
      <td>1</td>
      <td>192</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>103900</th>
      <td>1</td>
      <td>0</td>
      <td>49</td>
      <td>0</td>
      <td>0</td>
      <td>2347</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>103901</th>
      <td>1</td>
      <td>1</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>1995</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>7</td>
      <td>14.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>103902</th>
      <td>0</td>
      <td>1</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>1000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>103903</th>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>1723</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>103904 rows × 23 columns</p>
</div>



## Save the processed data


```python
train_df.to_csv(path_or_buf="processed_data/train_df.csv", index=False)
```


```python
train_df = pd.read_csv('processed_data/train_df.csv')
```


```python
train_df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Customer Type</th>
      <th>Age</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>Gate location</th>
      <th>Food and drink</th>
      <th>Online boarding</th>
      <th>Seat comfort</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
      <th>satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>1</td>
      <td>2</td>
      <td>460</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>25</td>
      <td>18.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>235</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>6.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>1142</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>562</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>11</td>
      <td>9.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>214</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Splitting the data
```python
from sklearn.model_selection import train_test_split
```
```python
train_val_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
```
```python
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)
```


```python
print(train_df.shape)
print(val_df.shape)
print(test_df.shape)
```

    (62342, 23)
    (20781, 23)
    (20781, 23)
    


```python
# train_df['satisfaction'] = train_df['satisfaction'].map({'neutral or dissatisfied':0 , 'satisfied':1})
# val_df['satisfaction'] = val_df['satisfaction'].map({'neutral or dissatisfied':0 , 'satisfied':1})
# test_df['satisfaction'] = test_df['satisfaction'].map({'neutral or dissatisfied':0 , 'satisfied':1})
```


```python
train_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Customer Type</th>
      <th>Age</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>Gate location</th>
      <th>Food and drink</th>
      <th>Online boarding</th>
      <th>Seat comfort</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
      <th>satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>83488</th>
      <td>0</td>
      <td>0</td>
      <td>51</td>
      <td>1</td>
      <td>0</td>
      <td>366</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31648</th>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>1</td>
      <td>1</td>
      <td>109</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22340</th>
      <td>1</td>
      <td>0</td>
      <td>50</td>
      <td>1</td>
      <td>1</td>
      <td>78</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>68992</th>
      <td>0</td>
      <td>0</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>1770</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>17</td>
      <td>8.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>100108</th>
      <td>1</td>
      <td>0</td>
      <td>19</td>
      <td>1</td>
      <td>1</td>
      <td>762</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>44593</th>
      <td>1</td>
      <td>0</td>
      <td>54</td>
      <td>0</td>
      <td>2</td>
      <td>989</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>25</td>
      <td>17.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59278</th>
      <td>1</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>3358</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29978</th>
      <td>1</td>
      <td>0</td>
      <td>58</td>
      <td>1</td>
      <td>2</td>
      <td>787</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>92224</th>
      <td>0</td>
      <td>0</td>
      <td>57</td>
      <td>1</td>
      <td>1</td>
      <td>431</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>67702</th>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>1</td>
      <td>1</td>
      <td>227</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>62342 rows × 23 columns</p>
</div>



## Scaling the Numeric features


```python
from sklearn.preprocessing import MinMaxScaler
```


```python
scaler = MinMaxScaler()
```


```python
# select the columns to be used for training/prediction

# training dataset
X_train_ = train_df.drop("satisfaction", axis=1)
X_train = scaler.fit_transform(X_train_) ##scaled
y_train = train_df.satisfaction
```


```python
X_val_ = val_df.drop("satisfaction", axis=1)
X_val = scaler.transform(X_val_) ##scaled
y_val = val_df.satisfaction
```


```python
X_test_ = test_df.drop("satisfaction", axis=1)
X_test = scaler.transform(X_test_) ##scaled
y_test = test_df.satisfaction
```

# Model Training Experiments

## Data Modelling

### Helper Functions


```python
def plot_roc_curve(y_true,y_prob_preds,ax):
    """
    To plot the ROC curve for the given predictions and model

    """ 
    fpr,tpr,threshold = roc_curve(y_true,y_prob_preds)
    roc_auc = auc(fpr,tpr)
    ax.plot(fpr,tpr,"b",label="AUC = %0.2f" % roc_auc)
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc='lower right')
    ax.plot([0,1],[0,1],'r--')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate");
    plt.show();
```


```python
def plot_confustion_matrix(y_true,y_preds,axes,name=''):
    """
    To plot the Confusion Matrix for the given predictions

    """     
    cm = confusion_matrix(y_true, y_preds)
    group_names = ['TN','FP','FN','TP']
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',ax=axes)
    axes.set_ylim([2,0])
    axes.set_xlabel('Prediction')
    axes.set_ylabel('Actual')
    axes.set_title(f'{name} Confusion Matrix');
```


```python
def make_classification_report(model,inputs,targets,model_name=None,record=False):
    """
     To Generate the classification report with all the metrics of a given model with confusion matrix as well as ROC AUC curve.

    """
    ### Getting the model name from model object
    if model_name is None: 
        model_name = str(type(model)).split(".")[-1][0:-2]

    ### Making the predictions for the given model
    preds = model.predict(inputs)
    if model_name in ["LinearSVC"]:
        prob_preds = model.decision_function(inputs)
    else:
        prob_preds = model.predict_proba(inputs)[:,1]

    ### printing the ROC AUC score
    auc_score = roc_auc_score(targets,prob_preds)
    print("ROC AUC Score : {:.2f}%\n".format(auc_score * 100.0))
    

    ### Plotting the Confusion Matrix and ROC AUC Curve
    fig, axes = plt.subplots(1, 2, figsize=(18,6))
    plot_confustion_matrix(targets,preds,axes[0],model_name)
    plot_roc_curve(targets,prob_preds,axes[1])
   
```

## Non Tree Models

## Logistic Rregression

<p>This type of statistical model (also known as logit model) is often used for classification and predictive analytics. Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1.</p> 
<p>In logistic regression, a logit transformation is applied on the odds—that is, the probability of success divided by the probability of failure. This is also commonly known as the log odds, or the natural logarithm of odds, and this logistic function is represented by the following formulas:<p>

Logit(pi) = 1/(1+ exp(-pi))

ln(pi/(1-pi)) = Beta_0 + Beta_1*X_1 + … + B_k*K_k 

![](https://media5.datahacker.rs/2021/01/44-1536x707.jpg) 

<p>In this logistic regression equation, logit(pi) is the dependent or response variable and x is the independent variable. The beta parameter, or coefficient, in this model is commonly estimated via maximum likelihood estimation (MLE). This method tests different values of beta through multiple iterations to optimize for the best fit of log odds. All of these iterations produce the log likelihood function, and logistic regression seeks to maximize this function to find the best parameter estimate. Once the optimal coefficient (or coefficients if there is more than one independent variable) is found, the conditional probabilities for each observation can be calculated, logged, and summed together to yield a predicted probability.</p> 
<p>For binary classification, a probability less than .5 will predict 0 while a probability greater than 0 will predict 1.  After the model has been computed, it’s best practice to evaluate the how well the model predicts the dependent variable, which is called goodness of fit.</p>

[source](https://www.ibm.com/topics/logistic-regression) 


![](https://media5.datahacker.rs/2021/01/83-1536x868.jpg) 


```python
# Import the model
from sklearn.linear_model import LogisticRegression

#fit the model
model = LogisticRegression()
model.fit(X_train,y_train)

# prediction
pred_train = model.predict(X_train)
pred_val = model.predict(X_test)


# model name
model_name = str(type(model)).split(".")[-1][0:-2]
print(f"\t\t{model_name.upper()} MODEL\n")

print('Training part:')
print(classification_report(y_train, pred_train,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print('validation part:')
print(classification_report(y_val, pred_val,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print("Accuracy score for training dataset",accuracy_score(y_train, pred_train))
print("Accuracy score for validation dataset",accuracy_score(y_val, pred_val))

make_classification_report(model,X_val,y_val)
```

    		LOGISTICREGRESSION MODEL
    
    Training part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.88      0.90      0.89     35308
                  satisfaction       0.87      0.83      0.85     27034
    
                      accuracy                           0.87     62342
                     macro avg       0.87      0.87      0.87     62342
                  weighted avg       0.87      0.87      0.87     62342
    
    validation part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.57      0.58      0.58     11858
                  satisfaction       0.43      0.42      0.43      8923
    
                      accuracy                           0.51     20781
                     macro avg       0.50      0.50      0.50     20781
                  weighted avg       0.51      0.51      0.51     20781
    
    Accuracy score for training dataset 0.874161881235764
    Accuracy score for validation dataset 0.5129685770655887
    ROC AUC Score : 92.65%
    
    


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_131_1.png)
    


**Observations**
- The auc roc score is 92.65 %
- But this model is not working good with validation data. And also not predecting the True Positives.


## Gaussian Naive Bayes

Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other.
<p><strong>Note</strong>: The assumptions made by Naive Bayes are not generally correct in real-world situations. In-fact, the independence assumption is never correct but often works well in practice.</p> 

<center><p><strong>Bayes’ Theorem</strong></p></center>

Bayes’ Theorem finds the probability of an event occurring given the probability of another event that has already occurred. Bayes’ theorem is stated mathematically as the following equation:

<center><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-7777aa719ea14857115695676adc0914_l3.svg" alt="Naive Bayes Equations" width="200" height="50"></center> 

where A and B are events and P(B) ≠ 0.

Basically, we are trying to find the probability of event A, given the event B is true. Event B is also termed as evidence.
- P(A) is the priori of A (the prior probability, i.e. Probability of event before evidence is seen). The evidence is an attribute value of an unknown instance(here, it is event B).
- P(A|B) is a posteriori probability of B, i.e. probability of event after evidence is seen.  

Now, with regards to our dataset, we can apply Bayes’ theorem in following way:

<center><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e85875a7ff9e9b557eab6281cc7ff078_l3.svg" alt="Naive Bayes Equations" width="200" height="50"></center>  

where, y is class variable and X is a dependent feature vector (of size n)  
<center><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-5385a4693c3fb17811cf36593978a601_l3.svg" alt="Naive Bayes Equations" width="200" height="50"></center> 

After substituting and solving the above equation we get the below 

<center><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c778553cb5a67518205ac6ea18502398_l3.svg" alt="Naive Bayes Equations" width="300" height="45"></center> 

Now, To create a classifier model. we need to find the probability of given set of inputs for all possible values of the class variable y and pick up the output with maximum probability. This can be expressed mathematically as:  

<center><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-f3637f468262bfbb4accb97da8110028_l3.svg" alt="Naive Bayes Equations" width="300" height="45"></center> 

So, finally, we are left with the task of calculating P(y) and P(xi | y).

Please note that P(y) is also called class probability and P(xi | y) is called conditional probability.

The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of P(xi | y). 

<center><p><strong>Gaussian Naive Bayes classifier</strong></p></center> 

In Gaussian Naive Bayes, continuous values associated with each feature are assumed to be distributed according to a Gaussian distribution. A Gaussian distribution is also called Normal distribution. When plotted, it gives a bell shaped curve which is symmetric about the mean of the feature values as shown below: 
<center><img src="https://media.geeksforgeeks.org/wp-content/uploads/naive-bayes-classification-1.png" alt="Naive Bayes Equations" width="300" height="200"></center>  

The likelihood of the features is assumed to be <a href="https://en.wikipedia.org/wiki/Gaussian_function">Gaussian</a>, hence, conditional probability is given by:  

<center><img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-7fb78d7323fcbade0cb664161a8e84c4_l3.svg" alt="Naive Bayes Equations" width="300" height="45"></center>


```python
# import the model
from sklearn.naive_bayes import GaussianNB

#fit the model
model =GaussianNB()
model.fit(X_train,y_train)

# prediction
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

# model name
model_name = str(type(model)).split(".")[-1][0:-2]
print(f"\t\t{model_name.upper()} MODEL\n")

print('Training part:')
print(classification_report(y_train, pred_train,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print('validation part:')
print(classification_report(y_val, pred_val,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print("Accuracy score for training dataset",accuracy_score(y_train, pred_train))
print("Accuracy score for validation dataset",accuracy_score(y_val, pred_val))

make_classification_report(model,X_val,y_val)
```

    		GAUSSIANNB MODEL
    
    Training part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.87      0.90      0.88     35308
                  satisfaction       0.86      0.82      0.84     27034
    
                      accuracy                           0.86     62342
                     macro avg       0.86      0.86      0.86     62342
                  weighted avg       0.86      0.86      0.86     62342
    
    validation part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.87      0.90      0.88     11858
                  satisfaction       0.86      0.82      0.84      8923
    
                      accuracy                           0.86     20781
                     macro avg       0.86      0.86      0.86     20781
                  weighted avg       0.86      0.86      0.86     20781
    
    Accuracy score for training dataset 0.8636874017516282
    Accuracy score for validation dataset 0.8641066358693037
    ROC AUC Score: 92.33%
    
    


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_136_1.png)
    


**Observations**

- The ROC AUC score is 92.33%. But the Recall and F1 scores are low. Thus we can say our model is failing to predict the True Positives
- The Recall and F1 Score of the GaussianNB is more less than Logistic Regresssion.
- This model working better with validation data.

## SVM(Support Vector Machines)

Support Vector Machine, or SVM, is one of the most popular supervised learning algorithms, and it can be used both for classification as well as regression problems. However, in machine learning, it is primarily used for classification problems. 

- In the SVM algorithm, each data item is plotted as a point in n-dimensional space, where n is the number of features we have at hand, and the value of each feature is the value of a particular coordinate.

- The goal of the SVM algorithm is to create the best line, or decision boundary, that can segregate the n-dimensional space into distinct classes, so that we can easily put any new data point in the correct category, in the future. This best decision boundary is called a hyperplane. 
- The best separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class. Indeed, there are many hyperplanes that might classify the data. Aas reasonable choice for the best hyperplane is the one that represents the largest separation, or margin, between the two classes.

The SVM algorithm chooses the extreme points that help in creating the hyperplane. These extreme cases are called support vectors, while the SVM classifier is the frontier, or hyperplane, that best segregates the distinct classes.

The diagram below shows two distinct classes, denoted respectively with blue and green points. 

![](https://ml-cheatsheet.readthedocs.io/en/latest/_images/svm.png)  

Support Vector Machine can be of two types:

- Linear SVM: A linear SVM is used for linearly separable data, which is the case of a dataset that can be classified into two distinct classes by using a single straight line.
- Non-linear SVM: A non-linear SVM is used for non-linearly separated data, which means that a dataset cannot be classified by using a straight line. 


<center><table>
<tr><td><center><img src="https://ml-cheatsheet.readthedocs.io/en/latest/_images/svm_linear.png" alt="LinearSVC " width="300" height="300"></center>
</td>
<td><center><img src="https://ml-cheatsheet.readthedocs.io/en/latest/_images/svm_nonlinear_1.png" alt="LinearSVC " width="300" height="300"></center>
</td></tr>
<tr><td><center>LinearSVM</center></td> 
<td><center>Non-linear SVM</center></td> </tr>
</table></center> 

We need to choose the best Kernel according to our need.

- The linear kernel is mostly preferred for text classification problems as it performs well for large datasets.
- Gaussian kernels tend to give good results when there is no additional information regarding data that is not available.
- Rbf kernel is also a kind of Gaussian kernel which projects the high dimensional data and then searches a linear separation for it.
- Polynomial kernels give good results for problems where all the training data is normalized. 


```python
# import the model
from sklearn.svm import LinearSVC

#fit the model
model =LinearSVC()
model.fit(X_train,y_train)

# prediction
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

# model name
model_name = str(type(model)).split(".")[-1][0:-2]
print(f"\t\t{model_name.upper()} MODEL\n")

print('Training part:')
print(classification_report(y_train, pred_train,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print('validation part:')
print(classification_report(y_val, pred_val,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print("Accuracy score for training dataset",accuracy_score(y_train, pred_train))
print("Accuracy score for validation dataset",accuracy_score(y_val, pred_val))

make_classification_report(model,X_val,y_val)
```

    C:\Users\prajw\anaconda3\Lib\site-packages\sklearn\svm\_classes.py:32: FutureWarning: The default value of `dual` will change from `True` to `'auto'` in 1.5. Set the value of `dual` explicitly to suppress the warning.
      warnings.warn(
    

    		LINEARSVC MODEL
    
    Training part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.88      0.91      0.89     35308
                  satisfaction       0.87      0.83      0.85     27034
    
                      accuracy                           0.87     62342
                     macro avg       0.87      0.87      0.87     62342
                  weighted avg       0.87      0.87      0.87     62342
    
    validation part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.88      0.90      0.89     11858
                  satisfaction       0.87      0.84      0.85      8923
    
                      accuracy                           0.87     20781
                     macro avg       0.87      0.87      0.87     20781
                  weighted avg       0.87      0.87      0.87     20781
    
    Accuracy score for training dataset 0.8734721375637612
    Accuracy score for validation dataset 0.8743082623550359
    ROC AUC Score : 92.59%
    
    


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_140_2.png)
    


**Observations**

- The ROC AUC score is 92.59%. 
- But the Recall and F1 scores are low. Thus we can say our model is failing to predict the True Positives

## K-Nearest Neighbours

<p>K-nearest neighbors is a supervised machine learning algorithm for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-nearest neighbors are used for classification or regression.</p>  
<p>The main idea behind K-NN is to find the K nearest data points, or neighbors, to a given data point and then predict the label or value of the given data point based on the labels or values of its K nearest neighbors.</p>
<p>K can be any positive integer, but in practice, K is often small, such as 3 or 5. The “K” in K-nearest neighbors refers to the number of items that the algorithm uses to make its prediction whether its a classification problem or a regression problem.</p>
<center><img src="https://www.datasciencecentral.com/wp-content/uploads/2021/10/1327962.png" alt="KNN " width="500" height="600"></center>

Once K and distance metric are selected, K-NN algorithm goes through the following steps:
- Calculate distance: The K-NN algorithm calculates the distance between a new data point and all training data points. This is done using the selected distance metric.
- Find nearest neighbors: Once distances are calculated, K-nearest neighbors are determined based on a set value of K.
- Predict target class label: After finding out K nearest neighbors, we can then predict the target class label for a new data point by taking majority vote from its K neighbors (in case of classification) or by taking average from its K neighbors (in case of regression).

Below are the different distance functions to calculate the nearest neighbours

![](https://www.saedsayad.com/images/KNN_similarity.png) 


```python
# import the model
from sklearn.neighbors import KNeighborsClassifier

#fit the model
model =KNeighborsClassifier()
model.fit(X_train,y_train)

# prediction
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

# model name
model_name = str(type(model)).split(".")[-1][0:-2]
print(f"\t\t{model_name.upper()} MODEL\n")

print('Training part:')
print(classification_report(y_train, pred_train,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print('validation part:')
print(classification_report(y_val, pred_val,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print("Accuracy score for training dataset",accuracy_score(y_train, pred_train))
print("Accuracy score for validation dataset",accuracy_score(y_val, pred_val))

make_classification_report(model,X_val,y_val)
```

    		KNEIGHBORSCLASSIFIER MODEL
    
    Training part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.93      0.98      0.95     35308
                  satisfaction       0.97      0.90      0.94     27034
    
                      accuracy                           0.95     62342
                     macro avg       0.95      0.94      0.94     62342
                  weighted avg       0.95      0.95      0.95     62342
    
    validation part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.91      0.96      0.94     11858
                  satisfaction       0.95      0.88      0.91      8923
    
                      accuracy                           0.93     20781
                     macro avg       0.93      0.92      0.92     20781
                  weighted avg       0.93      0.93      0.93     20781
    
    Accuracy score for training dataset 0.9460075069776395
    Accuracy score for validation dataset 0.9263750541359896
    ROC AUC Score: 96.69%
    
    


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_144_1.png)
    


**Observations:**
- The ROC AUC score is 96.69%.
- The Recall and F1 scores are good.
- But the model is failing to predict the True Positives.

## SGDClassifier

<center><strong><h5>Gradient Descent</strong></center>


Gradient Descent is a generic optimization algorithm capable of finding optimal solutions to a wide range of problems. 
- The general idea is to tweak parameters iteratively in order to minimize the cost function.
- An important parameter of Gradient Descent (GD) is the size of the steps, determined by the learning rate hyperparameters. If the learning rate is too small, then the algorithm will have to go through many iterations to converge, which will take a long time, and if it is too high we may jump the optimal value.

<strong>Note</strong>: When using Gradient Descent, we should ensure that all features have a similar scale (e.g. using Scikit-Learn’s StandardScaler class), or else it will take much longer to converge.
 

Types of Gradient Descent: There are three types of Gradient Descent:  

- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-batch Gradient Descent

<strong>Stochastic Gradient Descent</strong> 
- The word 'stochastic' means a system or process linked with a random probability. Hence, in Stochastic Gradient Descent, a few samples are selected randomly instead of the whole data set for each iteration. 

- If the sample size is very large, it becomes computationally very expensive to find the golbal minima over the entire dataset. With SGD a random sample is selected to perform each iteration. This sample is randomly shuffled and selected for performing the iteration. 

![](https://images.deepai.org/glossary-terms/dd6cdd6fcfea4af1a1075aac0b5aa110/sgd.png) 

In SGDClassifier from scikit learn implements regularized linear models with stochastic gradient descent (SGD) learning. The model it fits can be controlled with the loss parameter; by default, it fits a linear support vector machine (SVM). The various loss function supported is

- 'hinge' gives a linear SVM.

- 'log_loss’ gives logistic regression, a probabilistic classifier.

- 'modified_huber' is another smooth loss that brings tolerance to
outliers as well as probability estimates.

- 'squared_hinge' is like a hinge but is quadratically penalized.

- 'perceptron' is the linear loss used by the perceptron algorithm.



```python
# import the model
from sklearn.linear_model import SGDClassifier

#fit the model
model =SGDClassifier(loss='modified_huber',n_jobs=-1,random_state=42)
model.fit(X_train,y_train)

# prediction
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

# model name
model_name = str(type(model)).split(".")[-1][0:-2]
print(f"\t\t{model_name.upper()} MODEL\n")

print('Training part:')
print(classification_report(y_train, pred_train,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print('validation part:')
print(classification_report(y_val, pred_val,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print("Accuracy score for training dataset",accuracy_score(y_train, pred_train))
print("Accuracy score for validation dataset",accuracy_score(y_val, pred_val))

make_classification_report(model,X_val,y_val)
```

    		SGDCLASSIFIER MODEL
    
    Training part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.91      0.80      0.85     35308
                  satisfaction       0.77      0.89      0.83     27034
    
                      accuracy                           0.84     62342
                     macro avg       0.84      0.84      0.84     62342
                  weighted avg       0.85      0.84      0.84     62342
    
    validation part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.91      0.79      0.85     11858
                  satisfaction       0.76      0.90      0.83      8923
    
                      accuracy                           0.84     20781
                     macro avg       0.84      0.84      0.84     20781
                  weighted avg       0.85      0.84      0.84     20781
    
    Accuracy score for training dataset 0.837781912675243
    Accuracy score for validation dataset 0.8373514267840816
    ROC AUC Score : 92.37%
    
    


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_148_1.png)
    


**Observations:**
- The ROC AUC score is 92.37%. But the Recall and F1 scores are low.

## Tree Based models

## **Tree Based models**

A decision tree is a non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks. A decision tree starts at a single point (or ‘node’) which then branches (or ‘splits’) in two or more directions. Each branch offers different possible outcomes, incorporating a variety of decisions and chance events until a final outcome is achieved. 

<center>
<img src="https://i.imgur.com/4RX9be3.png" width="600" height="400">
</center> 


While there are multiple ways to select the best attribute at each node, two methods, information gain and Gini impurity, act as popular splitting criteria for decision tree models. They help to evaluate the quality of each test condition and how well it will be able to classify samples into a class.  

**Entropy and Information Gain** 

- Entropy is a concept that stems from information theory, which measures the impurity of the sample values. It is defined by the following formula, where: 

<center>
<img src="https://www.humaneer.org/static/7968dcf20ae9fa961be59cd8bbdf5a24/0d1a4/6003fba5-03fe-45f4-9cb2-acb0231c29e8.png" width="200" height="60">
</center> 
<center>
<table>
<tr><td>S - Set of all instances</td></tr>
<tr><td>N - Number of distinct class values</td></tr>
<tr><td>Pi - Event probablity</td></tr>
<table>
</center> 

- Information gain indicates how much information a particular variable or feature gives us about the final outcome. It can be found out by subtracting the entropy of a particular attribute inside the data set from the entropy of the whole data set.

<center>
<img src="https://www.humaneer.org/static/572ca05e5658d32bc53009f2cd766711/f1c64/cf354c51-a73a-4202-a56f-8b1a82e7136e.png" width="400" height="60">
</center> 

<center>
<table>
<tr><td>H(S) - entropy of whole data set S</td></tr>
<tr><td>|Sj| - number of instances with j value of an attribute A</td></tr>
<tr><td>|S| - total number of instances in the dataset</td></tr>
<tr><td>v - set of distinct values of an attribute A</td></tr>
<tr><td>H(Sj) - entropy of subset of instances for attribute A</td></tr>
<tr><td>H(A, S) - entropy of an attribute A</td></tr>
<table>
</center> 







```python
# import the model
from sklearn.tree import DecisionTreeClassifier

#fit the model
model =DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

# prediction
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

# model name
model_name = str(type(model)).split(".")[-1][0:-2]
print(f"\t\t{model_name.upper()} MODEL\n")

print('Training part:')
print(classification_report(y_train, pred_train,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print('validation part:')
print(classification_report(y_val, pred_val,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print("Accuracy score for training dataset",accuracy_score(y_train, pred_train))
print("Accuracy score for validation dataset",accuracy_score(y_val, pred_val))

make_classification_report(model,X_val,y_val)
```

    		DECISIONTREECLASSIFIER MODEL
    
    Training part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       1.00      1.00      1.00     35308
                  satisfaction       1.00      1.00      1.00     27034
    
                      accuracy                           1.00     62342
                     macro avg       1.00      1.00      1.00     62342
                  weighted avg       1.00      1.00      1.00     62342
    
    validation part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.96      0.95      0.95     11858
                  satisfaction       0.94      0.94      0.94      8923
    
                      accuracy                           0.95     20781
                     macro avg       0.95      0.95      0.95     20781
                  weighted avg       0.95      0.95      0.95     20781
    
    Accuracy score for the training dataset is 1.0
    Accuracy score for validation dataset 0.9486069005341418
    ROC AUC Score: 94.79%
    
    


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_153_1.png)
    


**Observations:**
- The ROC AUC score is 94.79%.
- The Recall and F1 scores are good.
- But the model will cause overfitting. as the accuracy score for the training dataset is 1.


## Random Forest classifier

Random Forest Classifier is an Ensemble algorithm. Random forest classifier creates a set of decision trees from randomly selected subset of the training set. It then aggregates the votes from different decision trees to decide the final class of the test object.  

This works well because a single decision tree may be prone to noise, but the aggregate of many decision trees reduces the effect of noise giving more accurate results. 

![](https://1.cms.s81c.com/sites/default/files/2020-12-07/Random%20Forest%20Diagram.jpg) 


Random forest algorithms have three main hyperparameters, which need to be set before training. These include node size, the number of trees, and the number of features sampled. From there, the random forest classifier can be used to solve for regression or classification problems. 
- The random forest algorithm is made up of a collection of decision trees, and each tree in the ensemble is comprised of a data sample drawn from a training set with replacement, called the bootstrap sample. 
- Of that training sample, one-third of it is set aside as test data, known as the out-of-bag (oob) sample. 
- Another instance of randomness is then injected through feature bagging, adding more diversity to the dataset and reducing the correlation among decision trees. 
- Depending on the type of problem, the determination of the prediction will vary. For a regression task, the individual decision trees will be averaged, and for a classification task, a majority vote—i.e. the most frequent categorical variable—will yield the predicted class. 
- Finally, the oob sample is then used for cross-validation, finalizing that prediction.


```python
#import the model

from sklearn.ensemble import RandomForestClassifier

#fit the model
model =RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

# prediction
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

# model name
model_name = str(type(model)).split(".")[-1][0:-2]
print(f"\t\t{model_name.upper()} MODEL\n")

print('Training part:')
print(classification_report(y_train, pred_train,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print('validation part:')
print(classification_report(y_val, pred_val,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print("Accuracy score for training dataset",accuracy_score(y_train, pred_train))
print("Accuracy score for validation dataset",accuracy_score(y_val, pred_val))

make_classification_report(model,X_val,y_val)
```

    		RANDOMFORESTCLASSIFIER MODEL
    
    Training part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       1.00      1.00      1.00     35308
                  satisfaction       1.00      1.00      1.00     27034
    
                      accuracy                           1.00     62342
                     macro avg       1.00      1.00      1.00     62342
                  weighted avg       1.00      1.00      1.00     62342
    
    validation part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.96      0.98      0.97     11858
                  satisfaction       0.97      0.94      0.95      8923
    
                      accuracy                           0.96     20781
                     macro avg       0.96      0.96      0.96     20781
                  weighted avg       0.96      0.96      0.96     20781
    
    Accuracy score for training dataset 1.0
    Accuracy score for validation dataset 0.9609739666041095
    ROC AUC Score: 99.34%
    
    


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_158_1.png)
    


**Observations:**
- The ROC AUC score is 99.37%.
- The Recall and F1 scores are good.
- But model can cause overfitting, as the accuracy score for training dataset is 1, 
- But after hypertunning we can train this model model is working much better with the validation dataset set as compared to other trained model.


##  ADA Boost Classifer 

AdaBoost is an ensemble learning method (also known as “meta-learning”) which was initially created to increase the efficiency of binary classifiers. AdaBoost uses an iterative approach to learn from the mistakes of weak classifiers, and turn them into strong ones.

Rather than being a model in itself, AdaBoost can be applied on top of any classifier to learn from its shortcomings and propose a more accurate model. It is usually called the “best out-of-the-box classifier” for this reason.

Stumps have one node and two leaves. AdaBoost uses a forest of such stumps rather than trees.

**Adaboost works in the following steps:** 

- Initially, Adaboost selects a training subset randomly.
It iteratively trains the AdaBoost machine learning model by selecting the training set based on the accurate prediction of the last training.
It assigns the higher weight to wrong classified observations so that in the next iteration these observations will get the high probability for classification.

- Also, It assigns the weight to the trained classifier in each iteration according to the accuracy of the classifier. The more accurate classifier will get high weight.

- This process iterate until the complete training data fits without any error or until reached to the specified maximum number of estimators.
To classify, perform a "vote" across all of the learning algorithms you built.

**Pros of Aaboost**

AdaBoost is easy to implement. It iteratively corrects the mistakes of the weak classifier and improves accuracy by combining weak learners. You can use many base classifiers with AdaBoost. AdaBoost is not prone to overfitting. This can be found out via experiment results, but there is no concrete reason available.

Cons of Aaboost
AdaBoost is sensitive to noise data. It is highly affected by outliers because it tries to fit each point perfectly. AdaBoost is slower compared to XGBoost.


```python
#import the model

from sklearn.ensemble import AdaBoostClassifier
#fit the model
model =AdaBoostClassifier()
model.fit(X_train,y_train)

# prediction
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

# model name
model_name = str(type(model)).split(".")[-1][0:-2]
print(f"\t\t{model_name.upper()} MODEL\n")

print('Training part:')
print(classification_report(y_train, pred_train,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print('validation part:')
print(classification_report(y_val, pred_val,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print("Accuracy score for training dataset",accuracy_score(y_train, pred_train))
print("Accuracy score for validation dataset",accuracy_score(y_val, pred_val))

make_classification_report(model,X_val,y_val)
                        
```

    		ADABOOSTCLASSIFIER MODEL
    
    Training part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.93      0.94      0.94     35308
                  satisfaction       0.92      0.91      0.92     27034
    
                      accuracy                           0.93     62342
                     macro avg       0.93      0.93      0.93     62342
                  weighted avg       0.93      0.93      0.93     62342
    
    validation part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.94      0.94      0.94     11858
                  satisfaction       0.92      0.92      0.92      8923
    
                      accuracy                           0.93     20781
                     macro avg       0.93      0.93      0.93     20781
                  weighted avg       0.93      0.93      0.93     20781
    
    Accuracy score for training dataset 0.9275929549902152
    Accuracy score for validation dataset 0.9282517684423272
    ROC AUC Score : 97.74%
    
    


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_163_1.png)
    


**Observations:**
- The ROC AUC score is 97.74%.
- The Recall and F1 scores are good but comparatively lower than the random forest.

## Gradient Boosting Classifier

Gradient boosting classifiers are a group of machine learning algorithms that combine many weak learning models together to create a strong predictive model. Decision trees are usually used when doing gradient boosting. Gradient boosting models are becoming popular because of their effectiveness at classifying complex datasets.

The gradient boosting algorithm is one of the most powerful algorithms in the field of machine learning. As we know that the errors in machine learning algorithms are broadly classified into two categories i.e. Bias Error and Variance Error. As gradient boosting is one of the boosting algorithms it is used to minimize bias error of the model.

**Gradient Boosting has three main components:**

1.**Loss Function -** The role of the loss function is to estimate how good the model is at making predictions with the given data. This could vary depending on the problem at hand. For example, if we're trying to predict the weight of a person depending on some input variables (a regression problem), then the loss function would be something that helps us find the difference between the predicted weights and the observed weights. On the other hand, if we're trying to categorize if a person will like a certain movie based on their personality, we'll require a loss function that helps us understand how accurate our model is at classifying people who did or didn't like certain movies.

2.**Weak Learner -** A weak learner is one that classifies our data but does so poorly, perhaps no better than random guessing. In other words, it has a high error rate. These are typically decision trees (also called decision stumps, because they are less complicated than typical decision trees).

3.**Additive Model -** This is the iterative and sequential approach of adding the trees (weak learners) one step at a time. After each iteration, we need to be closer to our final model. In other words, each iteration should reduce the value of our loss function.


```python
#import the model

from sklearn.ensemble import GradientBoostingClassifier

#fit the model
model =GradientBoostingClassifier()
model.fit(X_train,y_train)

# prediction
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

# model name
model_name = str(type(model)).split(".")[-1][0:-2]
print(f"\t\t{model_name.upper()} MODEL\n")

print('Training part:')
print(classification_report(y_train, pred_train,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print('validation part:')
print(classification_report(y_val, pred_val,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print("Accuracy score for training dataset",accuracy_score(y_train, pred_train))
print("Accuracy score for validation dataset",accuracy_score(y_val, pred_val))

make_classification_report(model,X_val,y_val)
```

    		GRADIENTBOOSTINGCLASSIFIER MODEL
    
    Training part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.94      0.96      0.95     35308
                  satisfaction       0.95      0.92      0.93     27034
    
                      accuracy                           0.94     62342
                     macro avg       0.94      0.94      0.94     62342
                  weighted avg       0.94      0.94      0.94     62342
    
    validation part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.94      0.96      0.95     11858
                  satisfaction       0.94      0.92      0.93      8923
    
                      accuracy                           0.94     20781
                     macro avg       0.94      0.94      0.94     20781
                  weighted avg       0.94      0.94      0.94     20781
    
    Accuracy score for traing dataset 0.942735234673254
    Accuracy score for validation dataset 0.9418218565035369
    ROC AUC Score : 98.71%
    
    


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_167_1.png)
    


**Observations:**
- The ROC AUC score is 98.71%.
- The Recall and F1 scores are good.
- We can choose this dataset to train our model.


```python

```

## Gradient Boosting Machines(XGBoost)

XgBoost stands for Extreme Gradient Boosting. It implements machine learning algorithms under the Gradient Boosting framework. 

- In this algorithm, decision trees are created in sequential form. Weights play an important role in XGBoost. 
- Weights are assigned to all the independent variables which are then fed into the decision tree which predicts results. 
- The weight of variables predicted wrong by the tree is increased and these variables are then fed to the second decision tree. These individual classifiers/predictors then ensemble to give a strong and more precise model.
- It can work on regression, classification, ranking, and user-defined prediction problems. 

![](https://miro.medium.com/max/809/1*ozf-ftCx-jy2jII4cEv9YA.png)


```python
#import the model

from xgboost import XGBClassifier

#fit the model
model =XGBClassifier()
model.fit(X_train,y_train)

# prediction
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

# model name
model_name = str(type(model)).split(".")[-1][0:-2]
print(f"\t\t{model_name.upper()} MODEL\n")

print('Training part:')
print(classification_report(y_train, pred_train,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print('validation part:')
print(classification_report(y_val, pred_val,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print("Accuracy score for training dataset",accuracy_score(y_train, pred_train))
print("Accuracy score for validation dataset",accuracy_score(y_val, pred_val))

make_classification_report(model,X_val,y_val)
```

    		XGBCLASSIFIER MODEL
    
    Training part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.98      0.99      0.98     35308
                  satisfaction       0.99      0.97      0.98     27034
    
                      accuracy                           0.98     62342
                     macro avg       0.98      0.98      0.98     62342
                  weighted avg       0.98      0.98      0.98     62342
    
    validation part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.96      0.98      0.97     11858
                  satisfaction       0.97      0.94      0.96      8923
    
                      accuracy                           0.96     20781
                     macro avg       0.96      0.96      0.96     20781
                  weighted avg       0.96      0.96      0.96     20781
    
    Accuracy score for training dataset 0.9802861634211286
    Accuracy score for validation dataset 0.9622732303546508
    ROC AUC Score: 99.51%
    
    


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_172_1.png)
    


Observations:

- The ROC AUC score is 99.51%.slightly higher than gradient Boosting.
- The Recall and F1 scores are good.
- We can choose this dataset to train our model. Ans can also improve our model with Hyperparameter tuning.


```python

```

## LightBoost

LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency.
- Lower memory usage.
- Better accuracy.
- Support of parallel, distributed, and GPU learning.
- Capable of handling large-scale data. 

LightGBM uses histogram-based algorithms, which bucket continuous feature (attribute) values into discrete bins. This speeds up training and reduces memory usage. 

LightGBM grows trees leaf-wise (best-first). It will choose the leaf with max delta loss to grow. Holding #leaf fixed, leaf-wise algorithms tend to achieve lower loss than level-wise algorithms.
![](https://lightgbm.readthedocs.io/en/latest/_images/leaf-wise.png)

Leaf-wise may cause over-fitting when #data is small, so LightGBM includes the max_depth parameter to limit tree depth. However, trees still grow leaf-wise even when max_depth is specified. 


```python
#import the model

import lightgbm as lgb

#fit the model
model =lgb.LGBMClassifier()
model.fit(X_train,y_train)

# prediction
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

# model name
model_name = str(type(model)).split(".")[-1][0:-2]
print(f"\t\t{model_name.upper()} MODEL\n")

print('Training part:')
print(classification_report(y_train, pred_train,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print('validation part:')
print(classification_report(y_val, pred_val,
                                    target_names=['neutral or dissatisfaction', 'satisfaction']))
print("Accuracy score for traing dataset",accuracy_score(y_train, pred_train))
print("Accuracy score for validation dataset",accuracy_score(y_val, pred_val))

make_classification_report(model,X_val,y_val)
```

    [LightGBM] [Info] Number of positive: 27034, number of negative: 35308
    [LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.002503 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 929
    [LightGBM] [Info] Number of data points in the train set: 62342, number of used features: 22
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.433640 -> initscore=-0.267014
    [LightGBM] [Info] Start training from score -0.267014
    		LGBMCLASSIFIER MODEL
    
    Training part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.96      0.98      0.97     35308
                  satisfaction       0.98      0.94      0.96     27034
    
                      accuracy                           0.97     62342
                     macro avg       0.97      0.96      0.97     62342
                  weighted avg       0.97      0.97      0.97     62342
    
    validation part:
                                precision    recall  f1-score   support
    
    neutral or dissatisfaction       0.96      0.98      0.97     11858
                  satisfaction       0.97      0.94      0.96      8923
    
                      accuracy                           0.96     20781
                     macro avg       0.96      0.96      0.96     20781
                  weighted avg       0.96      0.96      0.96     20781
    
    Accuracy score for training dataset 0.9671970742035867
    Accuracy score for validation dataset 0.9630431644290458
    ROC AUC Score: 99.49%
    
    


    
![png](https://github.com/praj2408/Airline-Passenger-Satisfaction-ML-Project/blob/main/docs/output_177_1.png)
    


**Observations:**
- this model is performing best with our Dataset.
- The ROC AUC score is 99.49%.
- The Recall and F1 scores are Very good.
- We can choose this dataset to train our model.

