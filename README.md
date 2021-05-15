# S21-IR-MovieChatbot

A chatbot that recommends personalized movies using content-based filtering techniques and collaborative filtering techniques. 

The work is the final project for Information Retrievel Class offered at UvA in Spring 2021. 

The video demo can be found [here](https://youtu.be/QfIViV7Gfxw)

## File structure
- `templates` contains html for the web application
- `data_process.py` preprocess raw data, and creates two pickel files *metadata.pkl* and *minidata.pkl*. *metadata.pkl* is used for content-based filtering, and *minidata.pkl* is for building a user profile for collaborative filtering. 
- `server.py` starts web application. It contains Flask controller.
- `server_model.py` contains main functions for making recommendation.  

## Running this locally 
1. Clone the repo and install necessary libraries
2. ```cd S21-IR-MovieChatbot/Chatbot```
3. Run ```python server.py``` to start local server, and visit  ```http://127.0.0.1:5000/login```

## Running with other data
1. Clone the repo and install necessary libraries
2. ```cd S21-IR-MovieChatbot```
3. ```mkdir Recommender Data``` and download data files inside this folder
4. Modify Chatbot/data_process.py to preprocess the data. The script will create two pickle files *metadata.pkl* and *minidata.pkl* which the system will be used. 
6. Run ```python Chatbot/server.py``` to start local server, and visit  ```http://127.0.0.1:5000/login``` to continue

## Miscellaneous
The chatbot javascript is adopted from [moviebot](https://github.com/steven-tey/moviebot).

Data used in the project is the [Movies Data](https://www.kaggle.com/rounakbanik/the-movies-dataset) downloadable on Kaggle
