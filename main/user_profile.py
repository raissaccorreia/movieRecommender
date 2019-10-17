import pandas as pd
import numpy as np

#linha,trash,movieId,tagId,relevance,label
#movie_profile = pd.read_csv("../ML_Dataset/ml-latest/genome-tags.csv")

#userId,movieId,rating,timestamp
ratings = pd.read_csv("../ML_Dataset/ml-latest-small/ratings.csv")
#movieId,title,genres
movie_genres = pd.read_csv("../ML_Dataset/ml-latest-small/movies.csv")

genre_dict={'Action':0,'Adventure':1,'Animation':2,"Children":3,"Comedy":4,"Crime":5,"Documentary":6,"Drama":7,"Fantasy":8,
"Film-Noir":9,"Horror":10,"Musical":11, "Mystery":12,"Romance":13,"Sci-Fi":14, "Thriller":15,"War":16,"Western":17,"IMAX":18,
"(no genres listed)":19}

num_movies = 193610
genre_movie_matrix = np.zeros(shape=(num_movies,len(genre_dict)),dtype=int)

for i in range(movie_genres.iloc[:, 2].values.size):
    genre_list = movie_genres.iloc[i, 2]
    genre_array = genre_list.split("|")
    for j in range(len(genre_array)):        
        index_y = genre_dict[genre_array[j]]
        genre_movie_matrix[movie_genres.iloc[i, 0],index_y] = 1

df = pd.DataFrame(genre_movie_matrix)
print(df)
#ate aqui tudo perfeitinho <3

#CRIAR DATAFRAME(NUM_USUARIO X GENERO - FLOAT CORRELATION)
#calcular n dado que esta por ordem de usuario
#criar array tamanho(1xnum_genero)
#calcular array iterando genero_assoc*rating +=
#divide array todo por n e append no novo dataset

############ parte futura ########

# Criar Matriz usuario-genero-relevance
# Produto Usuario->MovieId->Genero->Relavance*Rating

#Criar relacao usuario-tag(1128Dim)
    # Criar Matriz usuario-tag-relevance
    # Produto Usuario->MovieId->TagId->Relavance*Rating

#Criar cluster de usuarios

