import os
import flask
from flask import Flask, render_template, request, redirect
from src.components.inference import  Inference

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk 
from string import punctuation
import re
from nltk.corpus import stopwords
from flask import jsonify

nltk.download('stopwords')

set(stopwords.words('english'))



app = Flask(__name__, template_folder='template')

inferencing = Inference()

@app.route('/')
def home():
    return render_template('home.html')


@app.get("/image_classifier")
def image_classifier():
    return render_template("./image_classifier.html")

@app.get("/sentiment_analysis")
def sentiment_analysis():
    return render_template("./sentiment_analysis.html")


@app.route('/image_classifier', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_id, class_name = inferencing.predict_class(image=img_bytes)
        return jsonify(class_name=class_name)
    return render_template('./home.html')

@app.route('/sentiment_analysis', methods=['POST'])
def my_sentiment_analysis_post():
    stop_words = stopwords.words('english')
    
    text1 = request.form['text1'].lower()
    
    text_final = ''.join(c for c in text1 if not c.isdigit())
    
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound'])/2, 2)

    return render_template('sentiment_analysis.html', final=compound, text1=text_final, text2=dd['pos'], text3=dd['neu'], text4=dd['neg'])

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))