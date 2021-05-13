import os
from flask import Flask, render_template, Response, request, redirect
from server_model import create_user_profile, make_recommendation, make_recommendation_logged, CF
import pandas as pd

app = Flask(__name__)
userID = 0
minidata = pd.DataFrame
CF_userbased = CF

@app.route('/login')
def userLogin():
    return render_template('user_login.html')

@app.route('/query',methods = ['POST', 'GET'])
def prepareData():
    if request.method == 'POST':
        result=request.form
        global userID
        userID = result.get("UserID")
        global minidata
        minidata = pd.read_pickle("minidata.pkl")   
        global CF_userbased
        CF_userbased = create_user_profile(minidata,int(userID))

    return render_template('question.html', userID = userID)

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        answer1 = result.get("Question1")
        answer2 = result.get("Question2")
        answer3 = result.get("Question3")
        answer4 = result.get("Question4")

        metadata = pd.read_pickle("metadata.pkl")
        if userID == 0:
            ret = make_recommendation(metadata, answer1, answer2, answer3, answer4)
        else:
            ret = make_recommendation_logged(metadata, minidata, CF_userbased, int(userID), answer1, answer2, answer3, answer4)
        return render_template("result.html",result = ret, userID = userID)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)