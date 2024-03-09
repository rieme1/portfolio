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
Load the IMDb data.
Using the Pandas dataframe drop function, get rid of unnecessary columns.
identify the number of missing values within each column.
Replace "N/A", "NaN", and "Null" with an empty string.
Format Columns to floats:
 runtime from hours to minutes, release year, rating, number of ratings
 floats in millions - Budget,gross worldwide, gross usa and canada, Opening Weekend Gross in US & Canada(split in 2 columns)  
Drop unecessary columns

Output the cleaned up file onto a new csv called "IMDB_clean.csv".

Next: Exploration in Python and Tableau

