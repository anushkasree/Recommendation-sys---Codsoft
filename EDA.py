import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_main = pd.read_csv("training_ratings_for_kaggle_comp.csv")
#df_movies=pd.read_csv('file.csv',sep='::',names=["movie","movie_name","genre"])
df_movies = pd.read_csv('movies.dat',sep='::',names=["movie","movie_name","genre"],engine='python')
df_movies.to_csv('file.csv', index=False)  
df_main = pd.merge(df_main,df_movies,on="movie",how='left')
print()
print("=========================================")
print("Best movies with their ratings : ")
print()
print(df_main.head(5))
print("=========================================")
print()

average_ratings = df_main.groupby("movie_name")['rating'].mean().sort_values(ascending=False)
print()
print("=========================================")
print("Average rating of movies head in descending order : ")
print()
print(average_ratings.head(5))
print("=========================================")
print()

count_of_ratings = df_main.groupby("movie_name")['rating'].count().sort_values(ascending=False)
print()
print("=========================================")
print("Count of ratings head in descending order : ")
print()
print(count_of_ratings.head(5))
print("=========================================")
print()

ratings = pd.DataFrame(average_ratings)
ratings["count_of_ratings"] = pd.DataFrame(count_of_ratings)

plt.hist(ratings['count_of_ratings'].tolist(),bins=70)
plt.title("Histogram of count of ratings given to movies")
plt.show()

plt.hist(ratings['rating'].tolist(),bins=70)
plt.title("Histogram of average ratings given to movies")
plt.show()

plt.scatter(y=ratings['count_of_ratings'].tolist(),x=ratings['rating'].tolist())
plt.xlabel("Average ratings")
plt.ylabel("Count of ratings")
plt.show()
