import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import pandas_datareader as pdr
from keras.models import load_model
import streamlit as st

start = '2015-01-01'
end = date.isoformat(date.today())
st.title("STOCK PRICE PREDICTION")
input= st.text_input("enter the stock ticker(ex : for Apple it is AAPL)",'AAPL')
dataset=pdr.DataReader(input,'yahoo',start,end)

#describing the data
st.subheader('Data from 2015 - till date')
st.write(dataset.describe())



#splitting the data set into test and train

training_set= pd.DataFrame(dataset["Close"][0:int(len(dataset)*.70)])
testing_set= pd.DataFrame(dataset["Close"][int(len(dataset)*.70): int(len(dataset))])

#training data
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
dt_train_sc=sc.fit_transform(training_set)

x_train=[]
y_train=[]
for i in range(10,dt_train_sc.shape[0]):
    x_train.append(dt_train_sc[i-10:i])
    y_train.append(dt_train_sc[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)

#visualisations
st.subheader('closing price vs time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(dataset.Close)
st.pyplot(fig)

st.subheader('closing price vs time chart with mv100 & mv200')
st.write("The prices are going to increase if the moving average 100 crosses over the moving average 200 otherwise the price will fall")
mv100=dataset.Close.rolling(100).mean()
mv200=dataset.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(mv100,'r',label="mv 100")
plt.plot(mv200,'g',label="mv 200")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.plot(dataset.Close,'b',label="true price")
st.pyplot(fig)


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
dt_train_sc=sc.fit_transform(training_set)

regressor=load_model('stock_model1.h5')

#tesing part

sc_test=MinMaxScaler(feature_range=(0,1))
prev_data=training_set.tail(70)
f_test=prev_data.append(testing_set,ignore_index=True)
f_test_sc=sc_test.fit_transform(f_test)
x_test=[]
y_test=[]
for i in range(70,f_test_sc.shape[0]):
    x_test.append(f_test_sc[i-70:i])
    y_test.append(f_test_sc[i])
x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted= regressor.predict(x_test)
y_test=sc_test.inverse_transform(y_test)
y_predicted=sc_test.inverse_transform(y_predicted)

fig1=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label="TRUE PRICE")
plt.plot(y_predicted,'r',label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.subheader("True VS Predicted prices")
st.pyplot(fig1)

chuma=pd.DataFrame(dataset['Close'].tail(70))
chuma=pd.DataFrame(sc.fit_transform(chuma))
arr=[]
arr.append(chuma.iloc[:].values)
arr=np.array(arr)
st.subheader("The next day prediction is :")
result=sc.inverse_transform(regressor.predict(arr))
st.text(result[0][0])
prev_close=sc.inverse_transform([[chuma.iloc[-1,0]]])
#st.text(prev_close)
temp=result[0][0]-prev_close[0][0]
if temp > 0 :
    trend='rise'
else :
    trend='fall'
str1=result[0][0]
te='There could be a '+trend+' in the stock price'
st.write(te)

st.subheader('*The predictions are only for research purpose and it may vary by 10-15 % in the real time')

