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

class CF: #Collaborative Filtering
    def __init__ (self,df,simfunc):
        self.df=df
        self.simfunc=simfunc
        self.sim = pd.DataFrame(np.sum([0]),columns=df.user_id.unique(), index=df.user_id.unique())

    def compute_similarities(self, df, user):
        user1=user
        allusers=set(self.df.user_id)
        self.sim = {} #we are going to create a dictionary with the calculated similarities between users
        # for user1 in allusers:

        self.sim.setdefault(user1, {})
        #we take all the movies whatched by user1
        movies_user1=df[df['user_id']==user1]['movie_id']
        #we take all the users that have whatched any of the movies user1 has
        data_mini=pd.merge(df,movies_user1,on='movie_id')
        
        for user2 in allusers:
            if user1==user2:continue
            self.sim.setdefault(user2, {})
            if (user1 in self.sim[user2]):continue
            # we calculate the similarity between user1 and user2
            simi=self.simfunc(data_mini,user1,user2)
            if (simi<0):
                self.sim[user1][user2]=0
                # self.sim[user2][user1]=0
            else: # we store the similarity in the dictionary
                self.sim[user1][user2]=simi
                # self.sim[user2][user1]=simi
        return self.sim
    
    def predict(self,user,movie):
        allratings=self.df[(self.df['movie_id']==movie)]
        allusers_movie=set(allratings.user_id)
        
        numerator=0.0
        denominator=0.0
        
        for u in allusers_movie:
            if u==user:continue
            #we calculate the numerator and denominator of the prediction formula we saw at the beginning
            numerator+=self.sim[user][u]*float(allratings[allratings['user_id']==u]['rating'])
            denominator+=self.sim[user][u]
                
        if denominator==0: 
            if self.df.rating[self.df['movie_id']==movie].mean()>0:
            # if the sum of similarities is 0 we use the mean of ratings for that movie
                return self.df.rating[self.df['movie_id']==movie].mean()
            else:
            # else return mean rating for that user
                return self.df.rating[self.df['user_id']==user].mean()
        
        return numerator/denominator




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

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x] #cleaning up spaces in the data
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

# metadata['soup'] = metadata.apply(create_soup, axis=1)


def get_genres():
  genres = input("What Movie Genre are you interested in (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
  genres = " ".join(["".join(n.split()) for n in genres.lower().split(',')])
  return genres

def get_actors():
  actors = input("Who are some actors within the genre that you love (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
  actors = " ".join(["".join(n.split()) for n in actors.lower().split(',')])
  return actors

def get_directors():
  directors = input("Who are some directors within the genre that you love (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
  directors = " ".join(["".join(n.split()) for n in directors.lower().split(',')])
  return directors

def get_keywords():
  keywords = input("What are some of the keywords that describe the movie you want to watch, like elements of the plot, whether or not it is about friendship, etc? (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
  keywords = " ".join(["".join(n.split()) for n in keywords.lower().split(',')])
  return keywords

def get_searchTerms(genres, actors, directors, keywords):
  searchTerms = [] 
#   genres = get_genres()
  if genres != 'skip':
    searchTerms.append(genres)

#   actors = get_actors()
  if actors != 'skip':
    searchTerms.append(actors)

#   directors = get_directors()
  if directors != 'skip':
    searchTerms.append(directors)

#   keywords = get_keywords()
  if keywords != 'skip':
    searchTerms.append(keywords)
  
  return searchTerms

# def prepare_metadata():
#     metadata = pd.read_csv("C:\\Users\\szell\\uva\\2021Spring\\IR\S21-Information-Retrieval-Project\\chatbot_server\\Recommender Data\\movies_metadata.csv")
#     ratings = pd.read_csv("C:\\Users\\szell\\uva\\2021Spring\\IR\S21-Information-Retrieval-Project\\chatbot_server\\Recommender Data\\ratings.csv")
#     credits = pd.read_csv("C:\\Users\\szell\\uva\\2021Spring\\IR\S21-Information-Retrieval-Project\\chatbot_server\\Recommender Data\\credits.csv")
#     keywords = pd.read_csv("C:\\Users\\szell\\uva\\2021Spring\\IR\S21-Information-Retrieval-Project\\chatbot_server\\Recommender Data\\keywords.csv")

#     metadata = metadata.iloc[0:1000,:]

#     keywords['id'] = keywords['id'].astype('int')
#     credits['id'] = credits['id'].astype('int')
#     metadata['id'] = metadata['id'].astype('int')

#     metadata = metadata.merge(credits, on='id')
#     metadata = metadata.merge(keywords, on='id')

#     features = ['cast', 'crew', 'keywords', 'genres']
#     for feature in features:
#         metadata[feature] = metadata[feature].apply(literal_eval)

#     metadata['director'] = metadata['crew'].apply(get_director)

#     features = ['cast', 'keywords', 'genres']
#     for feature in features:
#         metadata[feature] = metadata[feature].apply(get_list)

#     features = ['cast', 'keywords', 'director', 'genres']

#     for feature in features:
#         metadata[feature] = metadata[feature].apply(clean_data)

#     metadata['soup'] = metadata.apply(create_soup, axis=1)
#     metadata.to_pickle("metadata.pkl")
#     return metadata


def make_recommendation(metadata, genres, actors, directors, keywords):
  new_row = metadata.iloc[-1,:].copy() #creating a copy of the last row of the 
  #dataset, which we will use to input the user's input
  
  #grabbing the new wordsoup from the user
  searchTerms = get_searchTerms(genres, actors, directors, keywords)  
  new_row.iloc[-1] = " ".join(searchTerms) #adding the input to our new row
  
  #adding the new row to the dataset
  metadata = metadata.append(new_row)
  
  #Vectorizing the entire matrix as described above!
  count = CountVectorizer(stop_words='english')
  count_matrix = count.fit_transform(metadata['soup'])

  #running pairwise cosine similarity 
  cosine_sim2 = cosine_similarity(count_matrix, count_matrix) #getting a similarity matrix
  
  #sorting cosine similarities by highest to lowest
  sim_scores = list(enumerate(cosine_sim2[-1,:]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  #matching the similarities to the movie titles and ids
  ranked_titles = []
  for i in range(1, 51):
    indx = sim_scores[i][0]
    ranked_titles.append([metadata['title'].iloc[indx], metadata['imdb_id'].iloc[indx], metadata['id'].iloc[indx]])
  
  return ranked_titles





##########################LOGGED IN USER################################
# def prepare_logged_in_data():
#     file="C:\\Users\\szell\\uva\\2021Spring\\IR\\S21-Information-Retrieval-Project\\chatbot_server\\Recommender Data\\"
#     cols_ratings = ['userId', 'movieId', 'rating']
#     ratings = pd.read_csv(file+'ratings.csv', usecols=cols_ratings)
#     ratings=ratings.rename(columns={"userId": "user_id","movieId":"movie_id"})
#     ratings['movie_id'] = ratings['movie_id'].astype('str')

#     trim_size = 10000
#     ratings = ratings[ratings.user_id <=trim_size ] #portion of the user's rating

#     df = ratings.groupby(['user_id']).size().reset_index(name='counts')
#     data = pd.merge(df, ratings)
#     df = df.sort_values("counts", ascending=False)

#     MIN_RATE = 100
#     data = data.sort_values("counts", ascending=False)
#     data = data[data.counts > MIN_RATE] 
#     ratings = data.drop("counts", axis=1)

#     cols_titles = ['id', 'original_title']
#     titles = pd.read_csv(file+'movies_metadata.csv', usecols=cols_titles)
#     titles = titles.iloc[0:1000,:] #only take first 1000 movies
#     titles=titles.rename(columns={"id": "movie_id","original_title":"title"})

#     data = pd.merge(ratings, titles,on=["movie_id", "movie_id"])
#     data['for_testing'] = False
#     grouped = data.groupby('user_id', group_keys=False).apply(assign_to_testset)
#     data_train = data[grouped.for_testing == False]
#     data_test = data[grouped.for_testing == True]
#     minidata = data

#     minidata.loc[:,'for_testing'] = False
#     grouped = minidata.groupby('user_id', group_keys=False).apply(assign_to_testset)
#     minidata_train = minidata[grouped.for_testing == False]
#     minidata_test = minidata[grouped.for_testing == True]
#     minidata.to_pickle("minidata.pkl")


def assign_to_testset(df):
    sampled_ids = np.random.choice(df.index,size=np.int64(np.ceil(df.index.size * 0.2)),replace=False)
    df.loc[sampled_ids, 'for_testing'] = True
    return df


def SimPearson(df,user1,user2,items_min=1):
    #movies rated by of user1
    data_user1=df[df['user_id']==user1]
    #movies rated by of user2
    data_user2=df[df['user_id']==user2]
    
    #movies rated by both
    both_rated=pd.merge(data_user1,data_user2,on='movie_id')
    
    if len(both_rated)<2:
        return 0
    if len(both_rated)<items_min:
        return 0
    res=pearsonr(both_rated.rating_x,both_rated.rating_y)[0]
    if(np.isnan(res)):
        return 0
    return res

def create_user_profile(minidata, userID):
    CF_userbased=CF(df=minidata,simfunc=SimPearson)
    CF_userbased.compute_similarities(df=minidata, user=userID)
    return CF_userbased

def make_recommendation_logged(metadata, minidata, CF_userbased, userID, genres, actors, directors, keywords): #minidata --> data with rating
    ranked_titles  = make_recommendation(metadata, genres, actors, directors, keywords)

    popular_mov_list = []
    minidata["movie_id"] = pd.to_numeric(minidata["movie_id"]) 
    for mov in ranked_titles:
        mov_id = mov[2]
        query_string = "movie_id=="+str(mov_id)
        query_result = minidata.query(query_string)
        if (query_result.size > 0): #movie with enough rating
            popular_mov_list.append(mov)

    # CF_userbased=CF(df=minidata,simfunc=SimPearson)
    # dicsim=CF_userbased.compute_similarities(df=minidata, user=userID)

    for i in range (0, len(popular_mov_list)):
        mov = popular_mov_list[i]
        example_pred=CF_userbased.predict(user=userID,movie=mov[2])
        popular_mov_list[i] = popular_mov_list[i] + [example_pred] #append predicted score
    popular_mov_list = sorted(popular_mov_list, key=lambda x: x[3], reverse=True)
    sorted_popular_mov_list = [movie[0:4] for movie in popular_mov_list]
    return sorted_popular_mov_list

def test():
    metadata = pd.read_pickle("metadata.pkl")
    minidata = pd.read_pickle("minidata.pkl")
    userID = 20
    CF_userbased = create_user_profile(minidata, userID)
    print (make_recommendation_logged(metadata, minidata, CF_userbased=CF_userbased, userID=userID, genres="Romance", actors="emma", directors="skip", keywords="friendship"))



if __name__ == "__main__":
    test()

