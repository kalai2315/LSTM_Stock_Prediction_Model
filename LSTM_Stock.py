import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
# Define start and end dates
start = '2015-01-01'
end = '2023-12-31'

st.title("Stock Trend Prediction")

# Get user input for the stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

try:
    # Use yfinance to download stock data directly
    df = yf.download(user_input, start=start, end=end)
    
    if not df.empty:
        st.subheader('Data from 2015 to 2023')
        st.write(df.describe())
    else:
        st.warning("No data found for the entered ticker symbol.")
except Exception as e:
    st.error(f"Error fetching data: {e}")

#visualisation

st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize= (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize= (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig1 = plt.figure(figsize= (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig1)

#Splitting the Dataset
train=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler=MinMaxScaler(feature_range=(0,1))
train_array=scaler.fit_transform(train)

# load the model
model=load_model('F:\project\Stock_Prediction_Models\keras_model.h5')

last_100_days=train.tail(100)
final_df = pd.concat([last_100_days, test], ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

y_pred=model.predict(x_test)
scaler=scaler.scale_

scale_data=1/scaler
y_pred = y_pred * scale_data
y_test = y_test * scale_data

st. subheader('Predictions Vs Original')
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_pred,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)