
from re import A
from flask import Flask, render_template, redirect, request
from scipy.sparse import dok
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Init vectorizer 
# tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_vectorizer = TfidfVectorizer()

# Create an instance of Flask
app = Flask(__name__)

with open('model.pkl', "rb") as f:
    model = pickle.load(f)

vect_model2 = pickle.load(open('vectorizer.pk','rb'))

# feature_names = model
# .get_booster().feature_names

# Route to render index.html template using data from Mongo
@app.route("/", methods=["GET", "POST"])
def home():
    output_message = ""

    if request.method == "POST":
        
        article_data = request.form['text']
        article_data = [article_data]
        print(article_data)

        vect_data = vect_model2.transform(article_data)
        print('vect_data', vect_data)




        # data must be converted to df with matching feature names before predict
        result = model.predict(vect_data)
        if result == 'REAL':
            output_message = "Best I can tell this a genuine article"
        else:
            output_message = "Check your sourses, this seems fake"
    
    return render_template("index.html", message = output_message)

if __name__ == "__main__":
    app.run()