# Stock Analysis
'''
Description:
29th March 2020
- Up to p.4 of YouTube Tutorial

'''

###Libraries
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc # new: 'from mplfinance.original_flavor import candlestick_ohlc'
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import pandas_datareader.data as web

style.use("ggplot")

##Choose the time period you are going to retrieve
start = dt.datetime(2018, 1, 1)
end = dt.date.today()

##Retrieve the data from Yahoo Finance
df = web.DataReader("MSFT", "yahoo", start, end)

## Resample the data
df_ohlc = df['Adj Close'].resample('10D').ohlc() #ohlc = open,high,low,close
df_vol = df['Volume'].resample('10D').sum()

#reset the index
df_ohlc.reset_index(inplace=True)
#convert datetime object to mdate number
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

##Create new columns for the dataset

# i. Moving average
# min_period = 0 --> allow first 100 rows to calculate the max moving average 
df['100ma'] = df['Adj Close'].rolling(window=100,min_periods = 0).mean()

# ii. Daily Change (in %)
df["Daily_Change"] = (df["Close"] - df["Open"]) / df["Open"] * 100

##Plotting Graphs

#Create the empty subplots
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
ax1.xaxis_date() #convert axis from the raw mdates to dates

#Plot candlestick graph
candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g') #g--> green: rise

#Plot the volumn chart
ax2.fill_between(df_vol.index.map(mdates.date2num),df_vol.values,0)
plt.ticklabel_format(style='plain',axis = 'y') #change from scientific notation to plain number
plt.show()



'''
###Export File
df.to_csv("MSFT.csv")
###Read File
df_MSFT = pd.read_csv("MSFT.csv", parse_dates=True, index_col=0)
###Plotting
a = df["Daily_Change"].plot()
plt.title("MSFT Daily Price Change (1990/01/01 - NOW")
plt.savefig("photo.png")
# plt.show()

# correlation of %change between MSFT and TSLA
print(np.corrcoef(df_MSFT["Daily_Change(%)"], df_TSLA["Daily_Change(%)"]))

'''