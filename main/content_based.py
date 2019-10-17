import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ds = pd.read_csv("/home/nikita/Downloads/sample-data.csv")

# TF-IDF Content Based Implementation with Scikit Learn
# TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
# IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

tf = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 3), min_df=0, stop_words="english"
)
tfidf_matrix = tf.fit_transform(ds["description"])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
results = {}
for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [
        (cosine_similarities[idx][i], ds["id"][i]) for i in similar_indices
    ]
    results[row["id"]] = similar_items[1:]

def item(id):
  return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]
# Just reads the results out of the dictionary.def
def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")    recs = results[item_id][:num]
    for rec in recs:
       print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")

recommend(item_id=11, num=5)


##Advantages of Content Based Filtering
#User independence: Collaborative filtering needs other users’ ratings to find similarities between the users
# and then give suggestions. Instead, the content-based method only has to analyze the items and a single user’s profile for the recommendation, which makes the process less cumbersome. Content-based filtering would thus produce more reliable results with fewer users in the system.
#Transparency: Collaborative filtering gives recommendations based on other unknown users who have
# the same taste as a given user, but with content-based filtering items are recommended on a feature-level basis.
#No cold start: As opposed to collaborative filtering, new items can be suggested before being rated by a
#  substantial number of users.
##Disadvantages of Content Based Filtering
#Limited content analysis: If the content doesn’t contain enough information to discriminate the items
# precisely, the recommendation itself risks being imprecise.
#Over-specialization: Content-based filtering provides a limited degree of novelty, since it has to match
# up the features of a user’s profile with available items. In the case of item-based filtering,
#  only item profiles are created and users are suggested items similar to what they rate or search for,
#  instead of their past history. A perfect content-based filtering system may suggest nothing unexpected or surprising.
