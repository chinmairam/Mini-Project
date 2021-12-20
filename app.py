from flask import Flask, render_template, url_for, request
import os
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model1 = pickle.load(open('chronic.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    predictions = model1.predict(final)
    output = predictions[0]
    return render_template('predict.html', prediction_text="Prediction is {}".format(output))

    #if int(output) == 0:
    #    return render_template('predict.html', picture="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRisxDm4brzjTGjDI6w0OlpwdoACEqcdfuOVg&usqp=CAU", prediction_text=f'Sorry,Your loan was not approved')
    #else:
    #    return render_template('predict.html', picture="https://image.shutterstock.com/image-vector/loan-approved-stamp-260nw-425124292.jpg", prediction_text=f'Your loan was approved')

if __name__ == '__main__':
    app.run(debug=True)
