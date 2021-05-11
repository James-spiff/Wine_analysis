import streamlit as st 
import logging, io, os, sys, base64, glob 
import pandas as pd 
import pickle
import numpy as np
from PIL import Image 
from sklearn.ensemble import GradientBoostingClassifier

st.write("""
# Wine Quality Designer


## Customize your wine parameters and get real-time quality predictions! 

""")

st.sidebar.header('User Input Features')

def get_wine_image_to_show(wine_color, wine_quality):
	if wine_color == 0:
		wine_color_str = 'white'
	else:
		wine_color_str = 'red'
	return('/static/images/wine_' + wine_color_str + '_' + str(wine_quality) + '.jpg')

def user_input_features():
	global color

	color = st.sidebar.slider('White or Red', min_value=0, max_value=1, step=1)
	fixed_acidity = st.sidebar.slider('Fixed acidity', min_value=3.5, max_value=16.5, step=0.5)
	volatile_acidity = st.sidebar.slider('Volatile acidity', min_value=0.08, max_value=1.60, step=0.08)
	citric_acid = st.sidebar.slider('Citric acid', min_value=0.0, max_value=1.60, step=0.1)
	residual_sugar = st.sidebar.slider('Residual sugar', min_value=0.5, max_value=66.5, step=0.5)
	chlorides = st.sidebar.slider('Chlorides', min_value=0.01, max_value=0.62, step=0.01)
	free_sulfur_dioxide = st.sidebar.slider('Free sulphur dioxide', min_value=1, max_value=291, step=5)
	total_sulfur_dioxide = st.sidebar.slider('Total sulphur dioxide', min_value=5, max_value=440, step=5)
	density = st.sidebar.slider('Density', min_value=0.98, max_value=0.99, step=0.01)
	pH = st.sidebar.slider('pH', min_value=2.7, max_value=4.1, step=0.1)
	sulphates = st.sidebar.slider('Sulphates', min_value=0.2, max_value=2.0, step=0.1)
	alcohol = st.sidebar.slider('Alcohol', min_value=8.0, max_value=15.0, step=0.5)

	data = {'fixed_acidity': fixed_acidity,
		'volatile_acidity': volatile_acidity,
		'citric_acid': citric_acid,
		'residual_sugar': residual_sugar,
		'chlorides': chlorides,
		'free_sulfur_dioxide': free_sulfur_dioxide,
		'total_sulfur_dioxide': total_sulfur_dioxide,
		'density': density,
		'pH': pH,
		'sulphates': sulphates,
		'alcohol': alcohol,
		'color': color}

	features = pd.DataFrame(data, index=[0])
	return features

input_df = user_input_features()
st.write(input_df)

# raw_wine_df = pd.read_csv('wine_df.csv')
# wine_df = raw_wine_df.drop(columns=['quality', 'quality_class'])
gbm_model = pickle.load(open('gbm_model_dump.pkl','rb'))
preds = gbm_model.predict_proba(input_df) 

st.subheader('Prediction Probability')
predicted_quality = [3,6,9][np.argmax(preds[0])]
st.write(predicted_quality)

#images = glob.glob(get_wine_image_to_show(color, predicted_quality))
#with open(get_wine_image_to_show(color, predicted_quality), 'rb') as image_file:
	#encoded_string = base64.b64encode(image_file.read())
#uploaded_file = st.file_uploader(encoded_string)
#image = Image.open(images)
st.image('static/images/quality_wine_logo.jpg', use_column_width=True)
#st.image(get_wine_image_to_show(color, predicted_quality), use_column_width=True)



	