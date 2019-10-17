# Movie Recommender

## Files Index

### In the main folder we have a lot of python scripts:

- rating_normalizer: responsible for normalize the ratings from 0.5-5 to 0.1-1

- crop_tags: in the genome_tags.csv we get only the tags above a certain level of relevance to describe a movie

- join_tag_tagid: creates a single csv with the tag and its respective label and its associated movies

- evaluations: list of evalution methods(RMSE, MAE, F1, RANKING DISTANCE and LONG TAIL PLOT)

- correlations: list of correlation methods(cosine similarity, soft cosine similarity, MSE and Euclidean Distance)
