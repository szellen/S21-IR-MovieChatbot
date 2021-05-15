import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import math
import matplotlib.pylab as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

######################### Content-based Filtering ################################
def make_recommendation(metadata,jsdata):
  new_row = metadata.iloc[-1,:].copy() #creating a copy of the last row of the 
  #dataset, which we will use to input the user's input
  
  #grabbing the new wordsoup from the user
#   searchTerms = get_searchTerms(genres, actors, directors, keywords)  
  jsdata = list(jsdata.split(" "))
  searchTerms = jsdata
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
  for i in range(1, 21):
    indx = sim_scores[i][0]
    ranked_titles.append([metadata['title'].iloc[indx], metadata['imdb_id'].iloc[indx], metadata['id'].iloc[indx],metadata['genres'].iloc[indx], "Most Relevant", "NaN"])
  return ranked_titles


######################### Collabrative Filtering ################################
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

def make_recommendation_logged(metadata, minidata, CF_userbased, userID, jsdata): #minidata --> data with rating
    ranked_titles = make_recommendation(metadata, jsdata)
    ranked_titles = [mov[0:4] for mov in ranked_titles] #remove last column
    popular_mov_list = []
    minidata["movie_id"] = pd.to_numeric(minidata["movie_id"]) 
    for mov in ranked_titles:
        mov_id = mov[2]
        query_string = "movie_id=="+str(mov_id)
        query_result = minidata.query(query_string)
        if (query_result.size > 0): #movie with enough rating
            popular_mov_list.append(mov)

    for i in range (0, len(popular_mov_list)):
        mov = popular_mov_list[i]
        example_pred=CF_userbased.predict(user=userID,movie=mov[2])
        example_pred = round(example_pred,3)
        popular_mov_list[i] = popular_mov_list[i] + ["Highest Predicted Score"] + [example_pred]  #append predicted score
    sorted_popular_mov_list = sorted(popular_mov_list, key=lambda x: x[5], reverse=True) #ranked by predicted score

    #combine with relevent result 
    complete_recommend_list = sorted_popular_mov_list
    index = 0
    col_list_title = [mov[2] for mov in sorted_popular_mov_list]

    while (len(complete_recommend_list) < 8):
        #if not already return 
        if (ranked_titles[index][2] not in col_list_title):
            relevant_mov = ranked_titles[index] + ["Most Relevant"] + ["NaN"]
            complete_recommend_list.append(relevant_mov)
        index+=1

    return complete_recommend_list[0:8]
    # return sorted_popular_mov_list

#for testing purpose    
def test():
    metadata = pd.read_pickle("metadata.pkl")
    minidata = pd.read_pickle("minidata.pkl")
    userID = 20
    CF_userbased = create_user_profile(minidata, userID)
    print (make_recommendation_logged(metadata, minidata, CF_userbased=CF_userbased, userID=userID, jsdata="Romance"))



if __name__ == "__main__":
    test()

