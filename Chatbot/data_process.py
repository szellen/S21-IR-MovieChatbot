import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import math
import matplotlib.pylab as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

######################### Prepare movie metadata ################################
def prepare_metadata():
    metadata = pd.read_csv("C:\\Users\\szell\\uva\\2021Spring\\IR\S21-Information-Retrieval-Project\\chatbot_server\\Recommender Data\\movies_metadata.csv")
    ratings = pd.read_csv("C:\\Users\\szell\\uva\\2021Spring\\IR\S21-Information-Retrieval-Project\\chatbot_server\\Recommender Data\\ratings.csv")
    credits = pd.read_csv("C:\\Users\\szell\\uva\\2021Spring\\IR\S21-Information-Retrieval-Project\\chatbot_server\\Recommender Data\\credits.csv")
    keywords = pd.read_csv("C:\\Users\\szell\\uva\\2021Spring\\IR\S21-Information-Retrieval-Project\\chatbot_server\\Recommender Data\\keywords.csv")

    metadata = metadata.iloc[0:1000,:]

    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    metadata['id'] = metadata['id'].astype('int')

    metadata = metadata.merge(credits, on='id')
    metadata = metadata.merge(keywords, on='id')

    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(literal_eval)

    metadata['director'] = metadata['crew'].apply(get_director)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(get_list)

    features = ['cast', 'keywords', 'director', 'genres']

    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data)

    metadata['soup'] = metadata.apply(create_soup, axis=1)
    metadata.to_pickle("metadata.pkl")
    return metadata

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x] #cleaning up spaces in the data
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

#Getting a list of the actors, keywords and genres
def get_list(x):
    if isinstance(x, list): #checking to see if the input is a list or not
        names = [i['name'] for i in x] #if we take a look at the data, we find that
        #the word 'name' is used as a key for the names actors, 
        #the actual keywords and the actual genres

        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []


######################### Prepare rating data ################################
def prepare_logged_in_data():
    file="C:\\Users\\szell\\uva\\2021Spring\\IR\\S21-Information-Retrieval-Project\\chatbot_server\\Recommender Data\\"
    cols_ratings = ['userId', 'movieId', 'rating']
    ratings = pd.read_csv(file+'ratings.csv', usecols=cols_ratings)
    ratings=ratings.rename(columns={"userId": "user_id","movieId":"movie_id"})
    ratings['movie_id'] = ratings['movie_id'].astype('str')

    trim_size = 10000
    ratings = ratings[ratings.user_id <=trim_size ] #portion of the user's rating

    df = ratings.groupby(['user_id']).size().reset_index(name='counts')
    data = pd.merge(df, ratings)
    df = df.sort_values("counts", ascending=False)

    MIN_RATE = 100
    data = data.sort_values("counts", ascending=False)
    data = data[data.counts > MIN_RATE] 
    ratings = data.drop("counts", axis=1)

    cols_titles = ['id', 'original_title']
    titles = pd.read_csv(file+'movies_metadata.csv', usecols=cols_titles)
    titles = titles.iloc[0:1000,:] #only take first 1000 movies
    titles=titles.rename(columns={"id": "movie_id","original_title":"title"})

    data = pd.merge(ratings, titles,on=["movie_id", "movie_id"])
    data['for_testing'] = False
    grouped = data.groupby('user_id', group_keys=False).apply(assign_to_testset)
    data_train = data[grouped.for_testing == False]
    data_test = data[grouped.for_testing == True]
    minidata = data

    minidata.loc[:,'for_testing'] = False
    grouped = minidata.groupby('user_id', group_keys=False).apply(assign_to_testset)
    minidata_train = minidata[grouped.for_testing == False]
    minidata_test = minidata[grouped.for_testing == True]
    minidata.to_pickle("minidata.pkl")


def assign_to_testset(df):
    sampled_ids = np.random.choice(df.index,size=np.int64(np.ceil(df.index.size * 0.2)),replace=False)
    df.loc[sampled_ids, 'for_testing'] = True
    return df



if __name__ == "__main__":
    prepare_metadata()
    prepare_logged_in_data()