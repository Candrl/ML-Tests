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

exit()
