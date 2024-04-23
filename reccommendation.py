import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df_main = pd.read_csv("training_ratings_for_kaggle_comp.csv")
print(df_main.head(3))

#converting .dat into a dataframe
column_names = ['movie','movie_name','genre']
df_movie = pd.read_csv("movies.dat",sep="::",names=column_names,engine='python')
df_movie.to_csv('file.csv', index=False)  

#merging into main df
df_main = df_main.merge(df_movie,on='movie',how='left')

#Getting mean ratings of each movie
print("---------------Best movies of all-------------------")
print(df_main.groupby('movie_name')['rating'].mean().sort_values(ascending=False))

#Getting number of ratings for each movie
print(df_main.groupby('movie_name')['rating'].count().sort_values(ascending=False))

#Making a dataframe for average rating and the number of ratings
ratings = pd.DataFrame(df_main.groupby('movie_name')['rating'].mean())
ratings['num_of_ratings'] = df_main.groupby('movie_name')['rating'].count()

movie_matrix = df_main.pivot_table(index='user',columns='movie_name',values='rating')
print(movie_matrix.head(3))

print(ratings.sort_values('num_of_ratings',ascending=False).head(10))

Star_Wars_user_ratings = movie_matrix['Star Wars: Episode VI - Return of the Jedi (1983)']
print(Star_Wars_user_ratings.head(5))

similar_to_start_wars = movie_matrix.corrwith(Star_Wars_user_ratings)

corr_Star_Wars = pd.DataFrame(similar_to_start_wars,columns=['Correlation'])
corr_Star_Wars.dropna(inplace=True)

print(corr_Star_Wars.sort_values('Correlation',ascending=False))
corr_Star_Wars =  corr_Star_Wars.join(ratings['num_of_ratings'])

#Final Prediction
print("Recommended movies who watched : Star Wars: Episode VI - Return of the Jedi (1983)\n")
print(corr_Star_Wars[corr_Star_Wars['num_of_ratings']>100].sort_values('Correlation',ascending=False).head(5))
