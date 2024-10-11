import numpy as np 
import pandas as pd
import yfinance as yf 
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import streamlit as st

options = ['Correlation','Trades']
page = st.sidebar.selectbox('Page',options, index = 0)
today = datetime.today()
datadownloaddate = datetime.today().strftime('%Y-%m-%d')

startdate = 2000

# Create a function to retrieve data for given ticker
def getdata(ticker,startdate):
    data = pd.DataFrame()
    try:
        for i in range(len(ticker)):
            info = yf.download(ticker[i], f'{startdate}-01-01', f'{datadownloaddate}')
            info = info[['Close']].rename(columns={'Close': ticker[i]})
            data = pd.concat([data, info], axis=1)
    except:
        pass
    return data

# List of tickers (will make this longer in the future)
ticks = ['DPZ', 'AAPL', 'GOOG', 'AMD', 'MSFT','BRK-B']

#Allow ths user to select a start date for the trading strategy
startdate = st.sidebar.slider("start year of backtest",min_value=1980,max_value=today.year)

d = getdata(ticks, startdate)

x = st.sidebar.selectbox('Stock 1',ticks,index = 5)
y = st.sidebar.selectbox('Stock 2', ticks,index = 4)

#Create a correlation matrix between the stocks to help visualise which pairs are useful to trade together
Correlation_matrix = d.corr()


stock1 = d[x]
stock2 = d[y]

#Calculate the Z-score for the ratio of the two stocks


ratio = stock1/stock2
    
df_zscore = (ratio-ratio.mean())/ratio.std()

# Set initial statistics
initial_capital = 1000
position = None
capital = initial_capital
entry_price_1 = 0 
entry_price_2 = 0
size_2 = 0 
size_1 = 0 

trades = []

# Check Z Z scores to see if the value exeeds +/- 1.5, if so execute a trade
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

        
    # If the Z score returns to within +- 0.5 we assume the two stocks have returned to steady state so we exit out of the trade 
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

#Convert our trades data into a pandas dataframe
Trades_df = pd.DataFrame(trades)

print(Trades_df)
print(capital)

# Create a graph showing entries and exits for our strategy
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

    if f'long {x} short {y}' in trade['Position']:
        plt.scatter(entry_date, trade[f'Entry {x}'], color = 'green', marker = '^', s = 100)
        plt.scatter(entry_date, trade[f'Entry {y}'], color = 'red', marker = '^', s = 100)
        plt.scatter(exit_date, trade[f'Exit {x}'], color = 'red', marker = '^', s = 100)
        plt.scatter(exit_date, trade[f'Exit {y}'], color = 'green', marker = '^', s = 100)

plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')

# Create Streamlit Pages Containing the correlation matrix and the graph and trades information

if page == 'Correlation':
    st.title("Correlation matrix of stocks in the S&P 500")
    st.write(Correlation_matrix)


Overallprofit = capital - initial_capital
profitperc  = (Overallprofit/initial_capital)*100 

if page == 'Trades':
    st.title('Trades Made')
    st.write(Trades_df)
    st.write(f'Profit: ${round(Overallprofit,2)}')
    st.write(f'Profit %: {round(profitperc,2)}%')
    st.write(plt.figure(f'{x} and {y} with Buy/Sell markers'))

    

    








  




