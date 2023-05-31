#!/usr/bin/env python
# coding: utf-8

# # Case-Study Title: Customers RFM Clustering (Market Segmentation based on Behavioral Approach)
# ###### Data Analysis methodology: CRISP-DM
# ###### Dataset: Iranian online e-commerce platform's customers transactions data in first 4 months of year 1398 (from 1398/01/01 to 1398/04/31)
# ###### Case Goal: Detect and Segment similar customers of e-commerce platform business (Customer Segmentation using RFM model)

# # Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', None)


# # Read Data from File

# In[3]:


data = pd.read_csv('CS_04.csv')


# In[4]:


data.shape  # 40537 records, 5 variables


# # Business Understanding
#  * know business process and issues
#  * know the context of the problem
#  * know the order of numbers in the business

# # Data Understanding
# ## Data Inspection (Data Understanding from Free Perspective)
# ### Dataset variables definition

# In[5]:


data.columns


# * **order_id**:        ID of customer's order (Transaction unique ID)
# * **created_ts**:      Date of ordering in EN (date of Transaction)
# * **shamsy_date**:     Date of ordering in FA (Jalali date)
# * **customer_id**:     ID of customer
# * **total_purchase**:  sum of purchase for ordering transaction (Transaction total payment Price in Rials)

# In[6]:


type(data)


# In[7]:


data.head()


# In[8]:


data.tail()


# In[9]:


data.info()


# In[10]:


# Do we have any NA in our Variables?
data.isna().sum()

# We have no MV problem in this dataset


# In[11]:


# Check for abnormality in data
data.describe(include='all')


# ### Simple Timeseries Analysis

# In[12]:


# Plot Histogram of 'total_purchase'
plt.hist(data['total_purchase'], color = 'green', bins = 50)
plt.xlabel('Total Purchase/Rls')
plt.ylabel('Frequency')


# In[13]:


data['created_ts'] = pd.to_datetime(data['created_ts'])  # change String to DateTime


# In[16]:


data['date'] = data['created_ts'].dt.date  # extract Date part of DateTime


# In[17]:


# Calculate Daily Demand
daily_demand = data.groupby(by=['date'])['order_id'].count()
daily_demand_rials = data.groupby(by=['date'])['total_purchase'].sum()


# In[18]:


# Plot changes of Daily Demand during 4-months
daily_demand.plot()
plt.xticks(rotation = 90)


# # Data PreProcessing
# ## Create RFM Dataset

# ### Frequency
# each customer had how many purchases (how many purchase records) in Analysis time-range?

# In[20]:


customer_f = pd.DataFrame({'freq': data.groupby(['customer_id'])['order_id'].count()})
customer_f


# In[21]:


# Plot Histogram of frequency
plt.hist(customer_f['freq'], color = 'green', bins = 50)
plt.xlabel('Purchase Frequency')
plt.ylabel('Frequency')


# In[22]:


customer_f.describe()

# the 'min' is 1: because this data is purchase data and everyone should have at-least 1 purchase to be in it!


# ### Recency
# how long has it been since each customer's last purchase related-to Analysis date?

# In[23]:


data.tail(1)

# our last data is for '2019-07-22' and so our reference point is '2019-07-23'
# (we take the next day of the last day as the reference point to prevent having 0 in our Recency data)


# In[24]:


# Calculate difference between reference-point Date and every order-record Date in days
r_date = pd.to_datetime('2019-07-23').date() - data['date']

r_date[0]  # the difference between '2019-07-23' and '2019-03-21' is 124 days


# In[26]:


data['r_date'] = r_date.dt.days  # extract days
data.head()


# In[27]:


# Recency: find minimum 'r_date' for each 'customer_id'
customer_r = pd.DataFrame({'recency': data.groupby(by = ['customer_id'])['r_date'].min()})
customer_r  # how many days have passed since the last purchase relative to the reference point for each customer


# In[28]:


# Plot Histogram of recency
plt.hist(customer_r['recency'], color = 'green', bins = 50)
plt.xlabel('Recency')
plt.ylabel('Frequency')


# In[29]:


customer_r.describe()

# 'min' is 1 because our reference point is '2019-07-23'
# 'max' is 124 because our Analysis time-range is 4 months


# ### Monetary
# each customer had how much purchases in Analysis time-range?

# In[30]:


customer_m = pd.DataFrame({'monetary': data.groupby(by = data['customer_id'])['total_purchase'].sum()})
customer_m


# In[32]:


# Plot Histogram of monetary
plt.hist(customer_m['monetary'], color = 'green', bins = 50)
plt.xlabel('Monetary')
plt.ylabel('Frequency')


# In[33]:


customer_m.describe()


# ### RFM Dataframe for Customers

# In[34]:


df = customer_f.merge(customer_r, left_index = True, right_index = True)
df


# In[35]:


rfm_customer = df.merge(customer_m, left_index = True, right_index = True)
rfm_customer  # R, F, M for each customer_id (prepared data for RFM Clustering)


# In[36]:


plt.scatter(x = rfm_customer['freq'], y = rfm_customer['recency'], alpha = 0.3)


# > there is no linear relationship

# In[37]:


plt.scatter(x = rfm_customer['freq'], y = rfm_customer['monetary'], alpha = 0.3)


# > there is a powerful linear relationship

# In[39]:


rfm_customer[['freq', 'monetary']].corr(method = 'pearson')  # Pearson correlation


# In[41]:


# We hold just F and R in the dataframe
rfm_customer_2 = rfm_customer[['freq', 'recency']]  # remove 'monetary' from Clustering because of powerful correlation
rfm_customer_2  # we will Cluster customers based-on these two features


# In[42]:


# Scale features
from sklearn.preprocessing import StandardScaler
scaled_data = StandardScaler().fit_transform(rfm_customer_2)  # Z-Normalization
scaled_data


# In[45]:


scaled_data = pd.DataFrame(scaled_data, 
                           columns = rfm_customer_2.columns,
                           index = rfm_customer_2.index)
scaled_data


# # Clustering
# ## Model 1: K-Means Clustering

# In[46]:


# First try
from sklearn.cluster import KMeans
seg_km1 = KMeans(n_clusters = 5, init = 'random', random_state = 123, n_init = 1).fit(scaled_data)


# In[47]:


rfm_customer['seg_km1'] = seg_km1.predict(scaled_data)  # label Cluster of each datapoint
rfm_customer


# In[48]:


# Results of Clustering
rfm_customer.groupby(['seg_km1'])[['freq', 'recency', 'monetary']].mean()

# calculate mean of different Customer Segments in freq, recency and monetary in 4-months


# > The Cluster with label 2 is our loyal customers
# > * purchase 16 times in past 4-months
# > * last purchase was 10 days ago
# > * purchase 546,000 Tomans in past 4-months

# In[50]:


# Give sense about distribution of our Clusters
plt.scatter(x = rfm_customer['freq'], y = rfm_customer['recency'], c = rfm_customer['seg_km1'], alpha = 0.3)
plt.xlabel('freq')
plt.ylabel('recency')


# In[51]:


# Second try
seg_km2 = KMeans(n_clusters = 5, init = 'random', random_state = 1000, n_init = 1).fit(scaled_data)


# In[52]:


rfm_customer['seg_km2'] = seg_km2.predict(scaled_data)
rfm_customer


# > **Note**: the Cluster labels are not important; the Cluster attributes are important

# In[54]:


# Results of Clustering
rfm_customer.groupby(['seg_km2'])[['freq', 'recency', 'monetary']].mean()


# > The Cluster with label 2 is our loyal customers
# > * purchase 11 times in past 4-months
# > * last purchase was 10 days ago
# > * purchase 399,000 Tomans in past 4-months

# In[55]:


# Give sense about distribution of our Clusters
plt.scatter(x = rfm_customer['freq'], y = rfm_customer['recency'], c = rfm_customer['seg_km2'], alpha = 0.3)
plt.xlabel('freq')
plt.ylabel('recency')


# > The Clusters are changed fundamentally (the boundaries are moved completely); so, our Clustering was not Robust!

# ## Model 2: K-Means++ Clustering

# In[56]:


# First try
seg_km3 = KMeans(n_clusters = 5, init = 'k-means++', random_state = 1000, n_init = 15).fit(scaled_data)


# In[57]:


rfm_customer['seg_km3'] = seg_km3.predict(scaled_data)
rfm_customer


# In[58]:


# Results of Clustering
rfm_customer.groupby(['seg_km3'])[['freq', 'recency', 'monetary']].mean()


# In[59]:


plt.scatter(x = rfm_customer['freq'], y = rfm_customer['recency'], c = rfm_customer['seg_km3'], alpha = 0.3)
plt.xlabel('freq')
plt.ylabel('recency')


# In[60]:


# Second try
seg_km4 = KMeans(n_clusters = 5, init = 'k-means++', random_state = 1234, n_init = 15).fit(scaled_data)


# In[61]:


rfm_customer['seg_km4'] = seg_km4.predict(scaled_data)
rfm_customer


# In[62]:


# Results of Clustering
rfm_customer.groupby(['seg_km4'])[['freq', 'recency', 'monetary']].mean()


# > same Results with **'First try'**

# In[63]:


plt.scatter(x = rfm_customer['freq'], y = rfm_customer['recency'], c = rfm_customer['seg_km4'], alpha = 0.3)
plt.xlabel('freq')
plt.ylabel('recency')


# # Optimal Number of Clusters
# ## Elbow method

# In[64]:


sse = []
for n in range(1, 8):  # n is Number of Clusters
    kmeans = KMeans(n_clusters = n,
                    init = 'k-means++',
                    random_state = 1234,
                    n_init = 10)  # do Clustering 7 times per each Number of Clusters
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)


# In[65]:


plt.plot(range(1, 8), sse)
plt.xticks(range(1, 8))
plt.xlabel('Number of Clusters')
plt.ylabel('Total within Sum of Squares Error')
plt.show()

# the less SSE is better


# > 6 Clusters is good

# ## Average Silhouette Coefficient

# In[68]:


from sklearn.metrics import silhouette_score

silhouette_coefficients = []
for n in range(2, 8):  # we should have at-least 2 Clusters in Silhouette
    kmeans = KMeans(n_clusters = n,
                    init = 'k-means++',
                    random_state = 1234,
                    n_init = 10)  # do Clustering 6 times per each Number of Clusters
    kmeans.fit(scaled_data)
    score = silhouette_score(scaled_data, kmeans.labels_)
    silhouette_coefficients.append(score)


# In[69]:


plt.plot(range(2, 8), silhouette_coefficients)
plt.xticks(range(2, 8))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.show()

# the more Silhouette score is better


# > 3 Clusters is good (based-on mathematical reasons)
# 
# > 4 Clusters is good (based-on applied reasons)
