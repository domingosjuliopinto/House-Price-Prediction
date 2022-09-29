import pandas as pd
import pickle
import numpy as np
import os
from flask import Flask, render_template, request

# app = Flask(__name__)
app = Flask(__name__, static_url_path='/static') 
data = pd.read_csv('Pune_Cleaned_data.csv')
pipe = pickle.load(open("RidgeModelPune.pkl",'rb'))
picFolder =os.path.join('static','pics')
app.config['UPLOAD_FOLDER'] = picFolder

@app.route('/')
def home():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'],'Background.png')
    locations = sorted(data['site_location'].unique())
    return render_template('index.html', locations=locations, user_image=pic1)

@app.route('/predict', methods=['POST'])
def prediction():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    input = pd.DataFrame([[location,sqft,float(bath),float(bhk)]],columns=['site_location','total_sqft','bath','bhk'])
    prediction = pipe.predict(input)[0] * 1e5 
    return str(np.round(prediction,2))

if __name__ == "__main__":
    app.run(debug=True,port=5000)