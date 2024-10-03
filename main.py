import numpy as np 
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf 
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller
import streamlit as st

options = ['Correlation','Trades']
page = st.sidebar.selectbox('Page',options, index = 0)
today = datetime.today()

startdate = 2000


def getdata(ticker,startdate):
    data = pd.DataFrame()
    try:
        for i in range(len(ticker)):
            info = yf.download(ticker[i], f'{startdate}-01-01', '2024-01-01')
            info = info[['Close']].rename(columns={'Close': ticker[i]})
            data = pd.concat([data, info], axis=1)
    except:
        pass
    return data

ticks = ['DPZ', 'AAPL', 'GOOG', 'AMD', 'MSFT','BRK-B']

startdate = st.sidebar.slider("start year of backtest",min_value=1980,max_value=today.year)

d = getdata(ticks, startdate)

x = st.sidebar.selectbox('Stock 1',ticks,index = 5)
y = st.sidebar.selectbox('Stock 2', ticks,index = 4)


Correlation_matrix = d.corr()
# print(Correlation_matrix)

stock1 = d[x]
stock2 = d[y]


#plt.figure('MSFT vs BRK-B')
#plt.plot(BRKB, label = 'Berkshire Hathaway')
#plt.plot(MSFT, label = 'Microsoft')
 

#plt.figure('MSFT-BRKB')
#plt.plot(BRKB - MSFT)
#plt.show()

result = ts.coint(stock1, stock2)
cointegration_t_statistic = result[0]
p_val = result[1]
critical_values_test_statistic_at_1_5_10 = result[2]

#print(result) 

#print(cointegration_t_statistic)
#print(critical_values_test_statistic_at_1_5_10)
#print(p_val)

#plt.figure('Price ratio')
ratio = stock1/stock2
#plt.plot(ratio)
#plt.axhline(ratio.mean(), color= 'red')
#plt.show()


Spread_ADF = adfuller(stock1-stock2)
Ratio_ADF = adfuller(stock1/stock2)

# print(BRKB_ADF[1])
# print(MSFT_ADF[1])
# print(Spread_ADF[1])
# print(Ratio_ADF[1])

df_zscore = (ratio-ratio.mean())/ratio.std()

initial_capital = 1000
position = None
capital = initial_capital
entry_price_1 = 0 
entry_price_2 = 0
size_2 = 0 
size_1 = 0 

trades = []


for i in range(len(df_zscore)):
    if df_zscore[i] >= 1.5 and position is None:
        entry_price_1 = stock1[i]
        entry_price_2 = stock2[i]
        position = f'short {x} long {y}'
        trade_entry_date = d.index[i]
        print(f"Entering short {x} long {y} on {trade_entry_date} at {x}: {entry_price_1} {y}: {entry_price_2}")
        size_1 = (initial_capital/2)/stock1[i]
        size_2 = (initial_capital/2)/stock2[i]

    elif df_zscore[i] <= -1.5 and position is None:
        entry_price_1 = stock1[i]
        entry_price_2 = stock2[i]
        position = f'long {x} short {y}'
        trade_entry_date = d.index[i]
        print(f"Entering long {x} short {y} on {trade_entry_date} at {x}: {entry_price_1} {y}: {entry_price_2}")
        size_1 = (initial_capital/2)/stock1[i]
        size_2 = (initial_capital/2)/stock2[i]

    elif -0.5 <= df_zscore[i] <= 0.5 and position is not None:
        exit_price_1 = stock1[i]
        exit_price_2 = stock2[i]
        trade_exit_date = d.index[i]

        if position == f'long {x} short {y}':
            profit_1 = (exit_price_1 - entry_price_1)*size_1
            profit_2 = (entry_price_2 - exit_price_2)*size_2

            totalprofit = profit_1 + profit_2

        elif position == f'short {x} long {y}':
            profit_1 = (entry_price_1 - exit_price_1)*size_1
            profit_2 = (exit_price_2 - entry_price_2)*size_2

            totalprofit = profit_1 + profit_2

        capital += totalprofit
        print(f"Capital after trade: {capital}")

        trades.append({
            'Entry Date': trade_entry_date,
            'Exit Date': trade_exit_date,
            'Position': position,
            f'Entry {x}': entry_price_1,
            f'Entry {y}': entry_price_2,
            f'Exit {x}': exit_price_1,
            f'Exit {y}': exit_price_2,
            'Profit': totalprofit,
            'Capital': capital,
            f'size {x}': size_1,
            f'size {y}': size_2})

        position = None

Trades_df = pd.DataFrame(trades)

print(Trades_df)
print(capital)

        
plt.figure(f'{x} and {y} with Buy/Sell markers')
plt.plot(stock1, color = 'blue')
plt.plot(stock2, color = 'orange')

for index, trade in Trades_df.iterrows():
    entry_date = trade['Entry Date']
    exit_date = trade['Exit Date']

    if f'short {x} long {y}' in trade['Position']:
        plt.scatter(entry_date, trade[f'Entry {x}'], color = 'red', marker = '^', s = 100)
        plt.scatter(entry_date, trade[f'Entry {y}'], color = 'green', marker = '^', s = 100)
        plt.scatter(exit_date, trade[f'Exit {x}'], color = 'green', marker = '^', s = 100)
        plt.scatter(exit_date, trade[f'Exit {y}'], color = 'red', marker = '^', s = 100)

    elif f'long BRKB short MSFT' in trade['Position']:
        plt.scatter(entry_date, trade[f'Entry {x}'], color = 'green', marker = '^', s = 100)
        plt.scatter(entry_date, trade[f'Entry {y}'], color = 'red', marker = '^', s = 100)
        plt.scatter(exit_date, trade[f'Exit {x}'], color = 'red', marker = '^', s = 100)
        plt.scatter(exit_date, trade[f'Exit {y}'], color = 'green', marker = '^', s = 100)

plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')




if page == 'Correlation':
    st.title("Correlation matrix of stocks in the S&P 500")
    st.write(Correlation_matrix)



if page == 'Trades':
    st.title('Trades Made')
    st.write(Trades_df)
    st.write(plt.figure(f'{x} and {y} with Buy/Sell markers'))
    

    








  




