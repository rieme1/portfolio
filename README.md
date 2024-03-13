# IMDB Data
## Cleaning data
### Intro
A comprehensive dataset of movies released over the years. This dataset contains information such as;
Source: Kaggle
 - Title: The name of the movie.
- Summary: A brief overview of the movie's plot.
- Director: The person responsible for overseeing the creative aspects of the film.
- Writer: The individual who crafted the screenplay and story for the movie.
- Main Genres: The primary categories or styles that the movie falls under.
- Motion Picture Rating: The age-appropriate classification for viewers.
- Motion Picture Rating Categories: i.e (G (General Audience): Suitable for all ages; no offensive content.)
- Runtime: The total duration of the movie.
- Release Year: The year in which the movie was officially released.
- Rating: The average score given to the movie by viewers.
- Number of Ratings: The total count of ratings submitted by viewers.
- Budget: The estimated cost of producing the movie.
- Gross in US & Canada: The total earnings from the movie's screening in the United States and Canada.
- Gross worldwide: The overall worldwide earnings of the movie.
- Opening Weekend Gross in US & Canada: The amount generated during the initial weekend of the movie's release in the United States and Canada.

### Task list
- Load the IMDb data.
- Using the Pandas dataframe drop function, get rid of unnecessary columns.
- Identify the number of missing values within each column.
- Replace "N/A", "NaN", and "Null" with an empty string.
Format Columns to floats:
 - Runtime from hours to minutes, release year, rating, number of ratings
 - Floats in millions - Budget,gross worldwide, gross usa and canada, Opening Weekend Gross in US & Canada(split in 2 columns)  
Drop unecessary columns

Output the cleaned up file onto a new csv called "IMDB_clean.csv".

Next: Exploration in Python and Tableau

## Exploration part 1 - in Python
- Descriptive statistics
- Distribution of continuous/numerical data i.e movie ratings, runtimes etc

## Part 2 - in Tableau 
- Exploring Categorical data i.e Top 20 titles by Gross worldwide, runtime distribution by genre, number of releases by year etc

## Chi Squared tests
### Motion picture rating(R, PG, 13+ ETC) vs main genres
- Splitting the genres and counting the occurrences of each genre
- Simplifying the genre data to include only the primary genre if it's one of the selected genres
- The contingency table for Motion Picture Rating vs. Primary Genre
- Focusing the Chi-square test on a subset of the most common motion picture ratings (e.g., G, PG, PG-13, R)

Heatmap 
The color intensity represents the number of movies in each category, with lighter colors indicating fewer movies and darker colors indicating more movies.
The genres are listed on the y-axis: Action, Adventure, Comedy, Drama, Other, and Thriller.
The motion picture ratings are listed on the x-axis: G, PG, PG-13, and R.

Key observations:

The "Other" genre has the highest count of R-rated movies, followed by Comedy and Drama.
PG-13 movies are most common in the Comedy genre, closely followed by the "Other" category.
G-rated movies are least common across all genres, with no G-rated movies in the Thriller category.
Adventure and Drama genres have a relatively balanced distribution across PG and PG-13 ratings.
This heatmap provides a quick and clear overview of the dataset, making it easy to spot which combinations of genre and rating are most or least common.

## K-Means clustering
Features selected
 - Gross worldwide (in millions)
 - Budget (in millions)
Outliers identified with IQR - defining bounds for what's considered an acceptable range then dropping the rows where at least one element is NaN after outlier removal

### Elbow Method and silhouette score
Used to determine the optimal number of clusters - in this case 4
Calculating the silhouette score to determine if the clustering structure is strong enough. 
  In the context of silhouette scores, which range from -1 to 1:

Scores closer to 1 indicate excellent cluster cohesion and separation. A score above 0.6 is generally considered good, implying that the clusters are well-distinguished from each other and that data points are well-matched to their clusters.
A score of 
0.6672154597605636
This score suggests that, on average, data points are much closer to their own cluster members than to those of the nearest neighboring cluster. It reflects well-separated clusters and a good fit between the data points and their assigned clusters.

### Results and possible developments
Cluster Characteristics:
 - The yellow cluster, predominantly at the lower end of both budget and gross worldwide scales, could indicate a group of items that are possibly low-budget productions or ones that didn’t achieve high revenues. This cluster is denser and seems to contain more items than the purple cluster, suggesting there are more items with these characteristics in your dataset.
- The purple cluster is more spread out and covers a wider range of gross worldwide earnings and budgets. This could indicate a more varied group of items, possibly mid to high-budget productions, or ones that had varying levels of success in terms of revenue.
  
Outliers:
- There appear to be potential outliers, especially in the purple cluster, where a few points lie far from the main cluster body. This might indicate exceptional items with significantly higher revenues or budgets compared to the rest.

Density and Distribution:
- The density of points is higher at the lower end of the budget and gross worldwide scales, which tapers off as the values increase. This is a common distribution in many financial datasets, where a large number of items have low to moderate values, while few have very high values.
Cluster Overlap:
- There's a slight overlap between clusters, particularly in the middle range of the plot. It might be worth examining these items more closely to see if they share any unique characteristics that aren’t captured by the budget and gross revenue features alone.
