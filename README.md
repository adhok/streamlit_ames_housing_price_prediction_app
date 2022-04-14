# streamlit_ames_housing_price_prediction_app


This is a streamlit app created by using the Ames Housing dataset on Kaggle (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).

* The app uses a two step approach to estimate the 95% confidence interval of the price of a given house.

* The first step involves using an autoencoder network to reduce the dimensionality of the input.

* The second step is to predict the 95% confidence intervals of prices of houses. This step uses the tensorflow probability library.

Useful criticism and feedback is welcome! You can email me at padhokshaja@gmail.com

The app can be found here https://share.streamlit.io/adhok/streamlit_ames_housing_price_prediction_app/main

*Steps to run this program*

```
pip install requirements.txt 

streamlit run streamlit_app.py
```
