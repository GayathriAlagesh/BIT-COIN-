import pandas as pd 
import numpy as np 
import itertools 
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams 
from tqdm import tqdm_notebook as tqdm 
import _pickle as pickle
plt.style.use('ggplot')
bc = pd.read_csv('E:/abc/BTC-USD.csv') 
bc['Date'] = pd.to_datetime(bc.Date) #print(bc.dtypes)
#Setting dates as the ind #print(bc.head())
bc.set_index('Date', inplace=True) 
bc = bc[['Close']].loc['2017-01-01':] #print(bc.head())
with open("curr_bitcoin.pickle", 'wb') as fp: 
   pickle.dump(bc, fp)
bc.plot(figsize=(16,5)) 
plt.xlabel('Date') 
plt.ylabel('Price in USD') 
plt.title('Bitcoin Price') 
plt.savefig('btcprice.png') 
plt.show()
#Method #1 - Differencing the Data # Differencing the price
bc_diff = bc.diff(1).dropna()


# Plotting the differences daily 
bc_diff.plot(figsize=(12,5))
plt.title('Plot of the Daily Changes in Price for BTC') 
plt.ylabel('Change in USD')
plt.show()
#Testing for Stationarity
results = adfuller(bc_diff.Close) 
print(f"P-value: {results[1]}")

#Method #2 - Taking the Log then differencing # Converting the data to a logarithmic scale 
bc_log = pd.DataFrame(np.log(bc.Close))
# Plotting the log of the data 
plt.figure(figsize=(16,8)) 
plt.plot(bc_log) 
plt.title('Log of BTC') 
plt.xlabel('Dates') 
plt.savefig('btc_log.png') 
plt.show()

# Differencing the log values
log_diff = bc_log.diff().dropna()
# Plotting the daily log difference
plt.figure(figsize=(16,8))
plt.plot(log_diff)
plt.title('Differencing Log')
plt.savefig('logdiff.png')
plt.show()
#Testing for Stationarity
results = adfuller(log_diff.Close)
print(f"P-value: {results[1]}")

#PACF AND ACF
#ACF and PACF for the Differencing
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,8))
plot_acf(bc_diff, ax=ax1, lags=40)
plot_pacf(bc_diff, ax=ax2, lags=40)
plt.show()

#Appears to be some correlation at day 5 and 10 mostly. #ACF and PACF for the Log Difference
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,8)) 
plot_acf(log_diff, ax=ax1, lags=40) 
plot_pacf(log_diff, ax=ax2, lags=40) 
plt.savefig('acfpacf.png')
plt.show()

def best_param(model, data, pdq, pdqs): 
    ans = []
    for comb in tqdm(pdq):
      for combs in tqdm(pdqs):
        try:
          mod = model(data,
                order=comb, 
                seasonal_order=combs, 
                enforce_stationarity=False, 
                enforce_invertibility=False, 
                freq='D')

          output = mod.fit()
          ans.append([comb, combs, output.aic])
        except:
         continue


    ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic']) 
    return ans_df.loc[ans_df.aic.idxmin()]
p = d = q = range(0,6)
 
d = range(2)


# Creating a list of all possible combinations of p, d, and q. 
pdq = list(itertools.product(p, d, q))

# Keeping seasonality at zeroes 
pdqs = [(0,0,0,0)]

# Finding the best parameters 
best_param(SARIMAX, bc_log, pdq, pdqs)

index = round(len(bc)*.80) 
train = bc_log.iloc[:index] 
test = bc_log.iloc[index:]
# Fitting the model to the training set 
model = SARIMAX(train,
        order=(1, 0, 0), 
        seasonal_order=(0,0,0,0), 
        freq='D', 
        enforce_stationarity=False, 
        enforce_invertibility=False)
output = model.fit() 
print(output.summary()) 
output.plot_diagnostics(figsize=(15,8)) 
plt.show()

fc	= output.get_forecast(len(test)) 
conf = fc.conf_int()
# Transforming the values back to normal
 
fc_series	= np.exp(pd.Series(fc.predicted_mean, index=test.index)) 
lower_series = np.exp(pd.Series(conf.iloc[:, 0], index=test.index)) 
upper_series = np.exp(pd.Series(conf.iloc[:, 1], index=test.index))

etrain = np.exp(train) 
etest = np.exp(test)

predictions = output.get_prediction(start=pd.to_datetime('2018'), dynamic=False) 
pred	= np.exp(predictions.predicted_mean)

# Confidence interval for the training set 
conf_int	= np.exp(predictions.conf_int())
low_conf	= np.exp(pd.Series(conf_int.iloc[:,0], index=train.index)) 
upper_conf = np.exp(pd.Series(conf_int.iloc[:,1], index=train.index))


rcParams['figure.figsize'] = 16, 8


# Plotting the training set, test set,forecast, and confidence interval. 
plt.plot(etrain, label='train')
plt.plot(etest, label='actual') 
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)

pred.plot(label='Fit to Training', color='w')

# Confidence interval for the fitted data 
plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1],
 
color='g',alpha=.5)


# Limiting the viewing size plt.xlim(['2018-06', '2019-10'])
plt.ylim([0, 20000])


plt.title('Fit to Train Data and \nForecasting vs Actual Test Values') 
plt.legend()
plt.savefig('btc_fit_fc.png') 
plt.show()

#CALCULATING THE RMSE FOR ARIMA
forecast = pred 
actual_val = etrain.Close

# Calculating our errors
rmse = np.sqrt(((forecast - actual_val) ** 2).mean()) 
print("The Root Mean Squared Error: ", rmse)
#FORECASTING FUTURE VALUES
# Fitting the model to the entire dataset 
model = SARIMAX(bc_log,
        order=(1, 0, 0), 
        seasonal_order=(0,0,0,0), 
        freq='D', 
        enforce_stationarity=False, 
        enforce_invertibility=False)
output = model.fit()

future = output.get_forecast(steps=30)
 

# Transforming values back
pred_fut = np.exp(future.predicted_mean)


# Confidence interval for our forecasted values 
pred_conf = future.conf_int()

# Transforming value back 
pred_conf = np.exp(pred_conf)

ax = np.exp(bc_log).plot(label='Actual', figsize=(16,8))


# Plottting the forecast 
pred_fut.plot(ax=ax, label='Future Vals')

# Shading in the confidence interval 
ax.fill_between(pred_conf.index,
           pred_conf.iloc[:, 0],
           pred_conf.iloc[:, 1], color='k', alpha=.25)


ax.set_xlabel('Date') 
ax.set_ylabel('Bitcoin Price') 
ax.set_xlim(['2018-01', '2019-11'])

plt.title('Forecasted values') 
plt.legend() 
plt.savefig('fc_val.png') 
plt.show()

# Zooming in on the Graph above
ax = np.exp(bc_log).plot(label='Actual', figsize=(16,8)) 
pred_fut.plot(ax=ax, label='Future Vals')

ax.fill_between(pred_conf.index,
          pred_conf.iloc[:, 0],
          pred_conf.iloc[:, 1], color='k', alpha=.25)


ax.set_xlabel('Date') 
ax.set_ylabel('Bitcoin Price') 
ax.set_xlim(['2019-06','2019-11'])

plt.title('Forecasted values \nZoomed in') 
plt.legend()
plt.savefig('fc_zoom.png') 
plt.show()
