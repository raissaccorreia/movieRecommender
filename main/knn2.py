import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import math

# * 0 or 1 relation movie-gender each line a movie
movie_vectors = pd.read_csv("../ML_Dataset/ml-latest-small/movie_profiles.csv")
