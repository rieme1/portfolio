#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning

# In[1]:


#Import the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime
import seaborn as sns


# In[2]:


# Import dataset and first 5 rows of the dataframe
imdb = pd.read_csv("IMDbMovies.csv")
imdb.head()


# ## Understanding the data
# 
# ### About Features:
# #### Title: The name of the movie.
# #### Summary: A brief overview of the movie's plot.
# #### Director: The person responsible for overseeing the creative aspects of the film.
# #### Writer: The individual who crafted the screenplay and story for the movie.
# #### Main Genres: The primary categories or styles that the movie falls under.
# #### Motion Picture Rating: The age-appropriate classification for viewers.
# #### Motion Picture Rating Categories: i.e (G (General Audience): Suitable for all ages; no offensive content.)
# #### Runtime: The total duration of the movie.
# #### Release Year: The year in which the movie was officially released.
# #### Rating: The average score given to the movie by viewers.
# #### Number of Ratings: The total count of ratings submitted by viewers.
# #### Budget: The estimated cost of producing the movie.
# #### Gross in US & Canada: The total earnings from the movie's screening in the United States and Canada.
# #### Gross worldwide: The overall worldwide earnings of the movie.
# #### Opening Weekend Gross in US & Canada: The amount generated during the initial weekend of the movie's release in the United States and Canada.
# 
# 

# In[53]:


#Inspect the data types
imdb.info()


# In[11]:


# Missing values.
imdb.isna().sum()


# In[3]:


#The Opening Weekend Gross in US & Canada column combines the opeming date and the gross opening $. 
#Split it into two columns: opening_weekend_date and n_ratings .

# Initialize lists to store the extracted and formatted data
opening_weekend_date = []
gross_opening_weekend = []

# Loop through each value in the 'Opening Weekend Gross in US & Canada' column
for i in imdb['Opening Weekend Gross in US & Canada']:
    # Use regular expression to find the first occurrence of a capital letter (indicating the month)
    t = re.search('[A-Z]', str(i))
    
    # Check if a capital letter is found
    if t != None:
        # Extract the substring from the found capital letter to the end of the string,
        # and format it as a datetime object with the format '%b %d, %Y'
        opening_weekend_date.append(datetime.strptime(str(i)[t.start():], '%b %d, %Y').strftime('%m.%d.%Y'))
        
        # Extract the numeric digits before the capital letter, convert to float,
        # and divide by 10^6 to represent millions. Round to three decimal places.
        gross_opening_weekend.append(round(float(''.join(re.findall('[0-9]', str(i)[:t.start()]))) / 10**6, 3))
    else:
        # If no capital letter is found, append numpy NaN to both lists
        opening_weekend_date.append(np.nan)
        gross_opening_weekend.append(np.nan)

# Create new columns in the DataFrame with the extracted and formatted data
imdb['Opening Weekend in US & Canada'] = opening_weekend_date
imdb['Gross Opening Weekend (in millions)'] = gross_opening_weekend

# Drop the original 'Opening Weekend Gross in US & Canada' column
imdb.drop(['Opening Weekend Gross in US & Canada'], axis=1, inplace=True)


# In[89]:


imdb


# In[4]:


#runtime - from hours to minutes

def runtime_to_minutes(time_str):
    if pd.isna(time_str):  # Check if the value is NaN
        return np.nan
    try:
        hours, minutes = time_str.split('h')
        hours = int(hours.strip())
        minutes = int(minutes.strip().replace('m', ''))
        total_minutes = hours * 60 + minutes
        return total_minutes
    except ValueError:
        return np.nan

# Apply the function to the 'Runtime' column
imdb['Runtime'] = imdb['Runtime'].apply(runtime_to_minutes)

imdb


# In[5]:


#release year
imdb['Release Year'] = pd.to_numeric(imdb['Release Year'], errors='coerce') # Convert non-numeric values to NaN
imdb['Release Year'] = imdb['Release Year'].astype(pd.Int32Dtype()) # Convert float years to integers
# rating 
imdb['Rating'] = imdb['Rating'].str.split('/').str[0] # Extract only the numerical rating


# In[6]:


# number of ratings
def convert_ratings(rating_str): #replace strings (38K) with floats
    if isinstance(rating_str, str):
        if 'K' in rating_str:
            return float(rating_str.replace('K', '')) * 1000
        elif 'M' in rating_str:
            return float(rating_str.replace('M', '')) * 1000000
        else:
            return float(rating_str)
    else:
        return float(rating_str)
    
# Apply the function to the 'Number of Ratings' column
imdb['Number of Ratings'] = imdb['Number of Ratings'].apply(convert_ratings)


# In[7]:


def convert_budget(budget_str):
    if isinstance(budget_str, float):
        return budget_str
    else:
        # Extract numeric value from budget string
        numeric_value = float(''.join(filter(str.isdigit, budget_str)))
        return numeric_value

# Apply the function to the 'Budget' column
imdb['Budget'] = imdb['Budget'].apply(convert_budget)


# In[8]:


# Define a function to correct columns with monetary values in millions
def corrected_columns(dataframe, c):
    # Create a new list to store the corrected values
    new_list = [
        # If the element is a string representation of 'nan', set the value to numpy NaN.
        np.nan if str(b).lower() == 'nan' 
        # If the element is not 'nan', extract numeric digits and convert to float,
        # then divide by 10^6 to represent millions.
        else round(float(''.join(re.findall('[0-9]', str(b)))) / 10**6, 3) 
        # Iterate through each element in the column of the DataFrame.
        for b in dataframe[c]
    ]
    return new_list

# Apply the corrected_columns function to the 'Budget' column
imdb['Budget (in millions)'] = corrected_columns(imdb, 'Budget')

# Apply the corrected_columns function to the 'Gross in US & Canada' column
imdb['Gross in US & Canada (in millions)'] = corrected_columns(imdb, 'Gross in US & Canada')

# Apply the corrected_columns function to the 'Gross worldwide' column
imdb['Gross worldwide (in millions)'] = corrected_columns(imdb, 'Gross worldwide')

# Drop the original 'Gross in Us & Canada' and 'Gross worldwide' columns
imdb.drop(['Gross in US & Canada', 'Gross worldwide', 'Budget'], axis=1, inplace=True)


# In[96]:


imdb


# ## Data Analysis
# ###    1) Univariate Data Analysis
# ###    2) Bivariate/Multivariate Analysis

# In[9]:


# Descriptive statistics
descriptive_stats = imdb.describe()
descriptive_stats


# In[129]:


sns.pairplot(imdb, height=2.2, aspect=1.25)
plt.savefig('pariplot.png');
# Visualize some relationships


# In[10]:


# Convert 'Rating' column to integer type
imdb['Rating'] = imdb['Rating'].astype(float)
# Plotting the distribution of movie ratings
plt.figure(figsize=(10, 6))
sns.displot(imdb['Rating'], bins=20, kde=True) 
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# In[11]:


# Plotting the distribution of movie runtimes
plt.figure(figsize=(10, 6))
sns.histplot(imdb['Runtime'], bins=30, kde=True, color='orange')
plt.title('Distribution of Movie Runtimes')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Frequency')
plt.show()


# In[234]:


# Plotting the distribution of movie budget
plt.figure(figsize=(10, 6))
sns.histplot(imdb['Gross in US & Canada (in millions)'], bins=30, kde=True, color='pink')
plt.title('Distribution of Movie - Gross in US & Canada (in millions)')
plt.xlabel('Gross in US & Canada (in millions)')
plt.ylabel('Frequency')
plt.show()

# Descriptive statistics for movie runtimes
us_descriptive_stats = imdb['Gross in US & Canada (in millions)'].describe()
us_descriptive_stats


# In[235]:


# Plotting the distribution of movie budget
plt.figure(figsize=(10, 6))
sns.histplot(imdb['Gross worldwide (in millions)'], bins=30, kde=True, color='purple')
plt.title('Distribution - Gross worldwide (in millions)')
plt.xlabel('Gross worldwide (in millions)')
plt.ylabel('Frequency')
plt.show()

# Descriptive statistics for movie runtimes
gww_descriptive_stats = imdb['Gross worldwide (in millions)'].describe()
gww_descriptive_stats


# In[236]:


# Plotting the distribution of movie opening weekend
plt.figure(figsize=(10, 6))
sns.histplot(imdb['Gross Opening Weekend (in millions)'], bins=30, kde=True, color='yellow')
plt.title('Distribution -Gross Opening Weekend (in millions)')
plt.xlabel('Gross Opening Weekend (in millions)')
plt.ylabel('Frequency')
plt.show()

# Descriptive statistics for movie runtimes
weekend_descriptive_stats = imdb['Gross Opening Weekend (in millions)'].describe()
weekend_descriptive_stats


# In[16]:


imdb['Release Year'] = imdb['Release Year'].astype('float64')
# Plotting the distribution of movie realese year
plt.figure(figsize=(10, 6))
sns.histplot(imdb['Release Year'], bins=30, kde=True, color='red')
plt.title('Distribution of realease year')
plt.xlabel('Release Year')
plt.ylabel('Frequency')
plt.show()

# Descriptive statistics for movie runtimes
Release_descriptive_stats = imdb['Release Year'].describe()
Release_descriptive_stats




# ## Chi Squared tests
# ### Motion picture rating vs main genres
# 

# In[17]:


# Splitting the genres and counting the occurrences of each genre
genre_counts = imdb['Main Genres'].str.split(',', expand=True).stack().value_counts()

genre_counts


# In[20]:


# Simplifying the genre data to include only the primary genre if it's one of the selected genres
selected_genres = ['Drama', 'Comedy', 'Action', 'Adventure', 'Thriller']
# Adjusting the function to handle NaN values in "Main Genres"
imdb['Primary Genre'] = imdb['Main Genres'].fillna('Other').apply(lambda x: x.split(',')[0] if x.split(',')[0] in selected_genres else 'Other')

# the contingency table for Motion Picture Rating vs. Primary Genre
contingency_table = pd.crosstab(imdb['Primary Genre'], imdb['Motion Picture Rating'])

pd.set_option('display.max_rows', None)

contingency_table


# In[21]:


#Focusing the Chi-square test on a subset of the most common motion picture ratings (e.g., G, PG, PG-13, R)
# Filtering the contingency table for selected ratings: G, PG, PG-13, R

selected_ratings = ['G', 'PG', 'PG-13', 'R']
filtered_contingency_table = contingency_table[selected_ratings]

# Performing the Chi-square test on the filtered contingency table
from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(filtered_contingency_table)

chi2, p, dof, expected


# In[22]:


# Setting up the matplotlib figure
plt.figure(figsize=(10, 8))

# Generating a heatmap
sns.heatmap(filtered_contingency_table, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)

# Adding titles and labels
plt.title('Heatmap of Movie Genres vs. Motion Picture Ratings')
plt.ylabel('Primary Genre')
plt.xlabel('Motion Picture Rating')

# Displaying the heatmap
plt.show()


# # Machine Learning
# ## K Means Clustering 
# 

# In[ ]:


#Pick features
#Drop nans and outliers - IQR methods


# In[218]:


# Selecting the relevant features for K-Means clustering
#features = imdb[['Gross worldwide (in millions)', 'Gross Opening Weekend (in millions)']]

features = imdb[['Gross worldwide (in millions)', 'Budget (in millions)']]
# Drop rows with any NaN values to clean the data before identifying outliers

features = (features - features.mean()) / features.std() # Perform Z-score standardization
features_clean = features.dropna()

# Calculating the IQR for each feature
Q1 = features_clean.quantile(0.25)
Q3 = features_clean.quantile(0.75)
IQR = Q3 - Q1

# Defining the bounds for what's considered an acceptable range (no outlier)
lower_bound = Q1 - 4.5 * IQR
upper_bound = Q3 + 4.5 * IQR

# Removing outliers
features_no_outliers = features_clean[(features_clean >= lower_bound) & (features_clean <= upper_bound)]

# Drop the rows where at least one element is NaN after outlier removal
features_no_outliers = features_no_outliers.dropna()

features_no_outliers.describe()


# In[220]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Selecting numerical features for clustering
#features = imdb[['Gross worldwide (in millions)', 'Budget (in millions)']].dropna()

# Normalizing the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_no_outliers)

# Determining the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the results onto a line graph to observe the 'Elbow'
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.grid(True)
plt.show()


# In[227]:


# Applying K-Means clustering with a chosen number of clusters to the original feature set
kmeans_final = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans_final.fit_predict(features_scaled)

# Adding the cluster labels to the original data (for those rows used in clustering)
features_no_outliers['Cluster'] = cluster_labels

features_no_outliers


# In[228]:


# Calculating the mean values for features within each cluster
cluster_means = features_no_outliers.groupby('Cluster').mean()

cluster_means


# In[229]:


from sklearn.metrics import silhouette_score

# Retrieve the feature columns for silhouette score calculation
feature_columns = features_no_outliers.drop('Cluster', axis=1)

# Retrieve the cluster assignments
cluster_assignments = features_no_outliers['Cluster']

# Calculate the silhouette score
silhouette_avg = silhouette_score(feature_columns, cluster_assignments)

silhouette_avg


# 

# In[233]:


# Extracting the two features for plotting
feature_x = features_no_outliers.columns[0]
feature_y = features_no_outliers.columns[1]

# Plotting the clusters
plt.figure(figsize=(10, 8))
plt.scatter(features_no_outliers[feature_x], features_no_outliers[feature_y], c=features_no_outliers['Cluster'], cmap='viridis', marker='o', edgecolor='k', s=70, alpha=0.7)


plt.title('K-means Clustering Visualization')
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.legend()
plt.colorbar(label='Cluster Label')

plt.show()



# In[ ]:





# In[ ]:





# In[ ]:




