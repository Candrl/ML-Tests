import pandas as pd
import numpy as np
from fbprophet import Prophet

df = pd.read_csv('data/test.csv') #, error_bad_lines=False, header=None)
df['y'] = np.log(df['y'])
df.head()

m = Prophet()
m.fit(df);

future = m.make_future_dataframe(periods=3)
future.tail()
forecast = m.predict(future)
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#m.plot(forecast);
exit()
