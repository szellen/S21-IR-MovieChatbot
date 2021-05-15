import os
from flask import Flask, render_template, Response, request, redirect
from server_model import create_user_profile, make_recommendation, make_recommendation_logged, CF
import pandas as pd

app = Flask(__name__)
userID = 0
minidata = pd.DataFrame
CF_userbased = CF
metadata = pd.read_pickle("metadata.pkl")

#login page
@app.route('/login')
def userLogin():
    return render_template('user_login.html')


#create user profile once login-in
@app.route('/query',methods = ['POST', 'GET'])
def prepareData():
    if request.method == 'POST':
        result=request.form
        global userID
        userID = result.get("UserID")
        if (int(userID) != 0):
            global minidata
            minidata = pd.read_pickle("minidata.pkl")   
            global CF_userbased
            CF_userbased = create_user_profile(minidata,int(userID))
    return render_template('interaction.html')

#search and return results
@app.route('/search/<jsdata>')
def makeRecommendation(jsdata):
    if int(userID) == 0:
        ret = make_recommendation(metadata, jsdata)
    else:
        ret = make_recommendation_logged(metadata, minidata, CF_userbased, int(userID), jsdata)
    return render_template("result.html",result = ret, userID = userID)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)