


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

import tensorflow_probability as tfp
tfd = tfp.distributions

from pickle import dump
from pickle import load

# dump(scaler, open('scaler.pkl', 'wb'))

#scaler = load(open('scaler.pkl', 'rb'))

scaler = MinMaxScaler()



tf.random.set_seed(42)

np.random.seed(42)


st.markdown("<body style ='color:#E2E0D9;'></body>", unsafe_allow_html=True)



st.markdown("<h4 style='text-align: center; color: #1B9E91;'>House Price Prediction in Ames,Iowa</h4>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: #1B9E91;'>A multi-step process is used to estimate the range of house prices based on your selection. The modeling process is done using the data found on Kaggle(link at left bottom corner of page)</h5>", unsafe_allow_html=True)


name_list = ['MSSubClass',
 'OverallQual',
 'YearBuilt',
 'YearRemodAdd',
 'BsmtUnfSF',
 'TotalBsmtSF',
 'FstFlrSF',
 'SndFlrSF',
 'GrLivArea',
 'FullBath',
 'HalfBath',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageCars',
 'GarageArea',
 'MoSold',
 'YrSold']

name_list_train = ['MSSubClass',
 'OverallQual',
 'YearBuilt',
 'YearRemodAdd',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'GrLivArea',
 'FullBath',
 'HalfBath',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageCars',
 'GarageArea',
 'MoSold',
 'YrSold']

data = pd.read_csv('train.csv')


data = data[name_list_train].values

scaler.fit(data)

description_list = [
 'What is the building class?',
 'What is the Overall material and finish quality?',
 'In which year was the Original construction date?',
 'In which year was it remodelled?',
 'What is the Unfinished square feet of basement area?',
 'What is the Total square feet of basement area?',
 'What is the First Floor square feet?',
 'What is the Second floor square feet?',
 'What is the Above grade (ground) living area square feet?',
 'What is the number of full bathrooms?',
'What is the number of Half baths?',
'What is the number of  Total rooms above grade (does not include bathrooms)?',
'What is the number of fireplaces?',
'What is the garage capacity in car sizes?',
'What is the size of garage in square feet?',
'In which month was it sold?',
'In which year was it sold?'






 ]
min_list = [20.0,1.0,1872.0,
 1950.0,
 0.0,
 0.0,
 334.0,
 0.0,
 334.0,
 0.0,
 0.0,
 2.0,
 0.0,
 0.0,
 0.0,
 1.0,
 2006.0]

max_list = [190.0,
 10.0,
 2010.0,
 2010.0,
 2336.0,
 6110.0,
 4692.0,
 2065.0,
 5642.0,
 3.0,
 2.0,
 14.0,
 3.0,
 4.0,
 1418.0,
 12.0,
 2010.0]

count = 0

with st.sidebar:

    for i in range(len(name_list)):

            

        variable_name = name_list[i]
        globals()[variable_name] = st.slider(description_list[i] ,min_value=int(min_list[i]), max_value =int(max_list[i]),step=1)
      
    st.write("[Kaggle Link to Data Set](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)")

    


data_df = {

'MSSubClass': [MSSubClass],
 'OverallQual': [OverallQual],
 'YearBuilt': [YearBuilt],
 'YearRemodAdd': [YearRemodAdd],
 'BsmtUnfSF': [BsmtUnfSF],
 'TotalBsmtSF': [TotalBsmtSF],
 '1stFlrSF': [FstFlrSF],
 '2ndFlrSF': [SndFlrSF],
 'GrLivArea':[GrLivArea],
 'FullBath': [FullBath],
 'HalfBath': [HalfBath],
 'TotRmsAbvGrd':[TotRmsAbvGrd],
 'Fireplaces': [Fireplaces],
 'GarageCars': [GarageCars],
 'GarageArea':[GarageArea],
 'MoSold': [MoSold],
 'YrSold' : [YrSold]





}

negloglik = lambda y, p_y: -p_y.log_prob(y) # note this

model1 = tf.keras.models.load_model('model_files/my_keras_model1.h5')

model1 = tf.keras.models.Sequential(model1.layers[:5])

data_df = pd.DataFrame.from_dict(data_df)

data_df_normal = scaler.transform(data_df)

latent_var = model1.predict(data_df_normal)

model2 = tf.keras.models.load_model('model_files/keras_2.h5',compile=False)

yhat = model2(latent_var)



col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    center_button = st.button('Calculate range of house price')



if center_button:

    import time

    #my_bar = st.progress(0)

    with st.spinner('Calculating....'):
        time.sleep(2)



    st.markdown("<h5 style='text-align: center; color: #1B9E91;'>The price range of your house is between:</h5>", unsafe_allow_html=True)


    col1, col2 = st.columns([3, 3])

    lower_number = "{:,.2f}".format(int(yhat.mean().numpy()-1.95*yhat.stddev().numpy()))
    higher_number = "{:,.2f}".format(int(yhat.mean().numpy()+1.95*yhat.stddev().numpy()))

    col1, col2, col3 = st.columns(3)

    

    with col1:
        st.write("")

    with col2:
        st.subheader("USD "+ str(lower_number))
        st.subheader("       AND ")

        st.subheader(" USD "+str(higher_number))


    with col3:
        st.write("")

    

    

    import base64

    file_ = open("kramer_gif.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<center><img src="data:image/gif;base64,{data_url}" alt="cat gif"></center>',
        unsafe_allow_html=True,
    )
    



    

