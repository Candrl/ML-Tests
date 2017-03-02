import pandas as pd
import numpy as np
from fbprophet import Prophet

#https://facebookincubator.github.io/prophet/docs/quick_start.html#python-api

'''
ds is datestamp
y is the desired forecasted value, in this case wikipedia page views
'''

df = pd.read_csv('.../data/example.csv')
df['y'] = np.log(df['y'])
df.head()

#Fit current data to timeframe
m = Prophet()
m.fit(df);

#Future timeframes
future = m.make_future_dataframe(periods=365)
future.tail()

#create a forecast with higher and lower ranges
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(forecast);
#includes components -- trends for seasonaility, weekly, holidays (if included)
m.plot_components(forecast);


#Forecasting Growth
#take read file and go further!
df['cap'] = 8.5 #MaxCaryingCapacity
m = Prophet(growth='logistic')
m.fit(df)

future = m.make_future_dataframe(periods=1826) #3Years
future['cap'] = 8.5
fcst = m.predict(future)
m.plot(fcst);

#TrendChangepoints, flexibility
m = Prophet(df, changepoint.prior.scale = 0.5) #Default is 0.05, increasing will make trend MORE flexible. (0.001 will decrease)
forecast = m.fit(df).predict(future)
m.plot(forecast);

m = Prophet(changepoints=['2014-01-01']) #Identify specific changepoints
forecast = m.fit(df).predict(future)
m.plot(forecast);

#Holiday Modeling
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})  #lower_window would be day before, upper_window would be day after
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))

#Bring Holiday set in
m = Prophet(holidays=holidays)
forecast = m.fit(df).predict(future)

#Bring Forecast with Holiday in
forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
        ['ds', 'playoff', 'superbowl']][-10:]

m.plot_components(forecast);

#Smooth out Holiday effect (if they're overfitting -- 10 is default)
m = Prophet(holidays=holidays, holidays_prior_scale=1).fit(df)
forecast = m.predict(future)
forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
    ['ds', 'playoff', 'superbowl']][-10:]

#Uncertainty intervals (default is 80% - yikes)
forecast = Prophet(interval_width=0.95).fit(df).predict(future) #uncertainty with prediction

#Uncertainty in Seasonality; Default is 0
m = forecast(mcmc_samples=500)
forecast = m.fit(df).predict(future)
m.plot_components(forecast); #Plot sampling of Seasonality components (takes much longer)

#Setting Outliers
df.loc[(df['ds'] > '2010-01-01') & (df['ds'] < '2011-01-01'), 'y'] = None
model = Prophet().fit(df)
model.plot(model.predict(future));

#Non-daily time values
df = pd.read_csv('../examples/example_retail_sales.csv')
m = Prophet().fit(df)
future = m.make_future_dataframe(periods=3652) #10 Years
fcst = m.predict(future)
m.plot(fcst);

#Redefine as monthly values
future = m.make_future_dataframe(periods=120, freq='M') #valus are months, 120 months, still 10 years
fcst = m.predict(future)
m.plot(fcst);

exit()
