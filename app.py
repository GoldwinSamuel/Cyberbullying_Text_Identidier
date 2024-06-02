from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from functions import custom_input_prediction

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/querypage')
def querypage():
    return render_template('querypage.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet_input = request.form['tweet_input']
    prediction = custom_input_prediction(tweet_input)

    graph=["Age","Ethnicity","Gender","Not Cyberbullying","Other Cyberbullying","Religion"]
    for i in range(len(graph)):
        if prediction==graph[i]:
            img_src = '/static/img/'+str(i)+'.png'
        else:
            continue
    return render_template('result.html', prediction=prediction, img_src=img_src)

if __name__ == '__main__':
    app.run(debug=True)