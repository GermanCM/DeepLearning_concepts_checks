# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Pruebo : <p>
# 
# Estimación univariable de series temporales asociadas a número de viajeros en vuelos desde 1949y

# %%
'''
Time Series example: https://github.com/aarshayj/Analytics_Vidhya/blob/master/Articles/Time_Series_Analysis/Time_Series_AirPassenger.ipynb
'''

import pandas as pd
import numpy as np

data_frame = pd.read_csv('https://raw.githubusercontent.com/aarshayj/analytics_vidhya/master/Articles/Time_Series_Analysis/AirPassengers.csv') 

# %%
data_frame['Month'] = pd.to_datetime(data_frame['Month'])
#setear como index
data_frame.index = data_frame['Month']

# %%
ts = data_frame
#ahora ya nos sobra la columna 'Month', la cual la hemos aÃ±adido como index del dataframe
ts = ts.drop(['Month'], 1)

# %%
#Dickey-Fuller test incluyendo representaciÃ³n de media y SD
from statsmodels.tsa.stattools import adfuller
def test_stationarity_DFULLER(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()  #window = nÃºmero de los Ãºltimos meses escogidos  
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries.iloc[:,0].values, autolag='AIC')   
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
        

# %%
test_stationarity_DFULLER(ts)


# %%
ts_log = np.log(ts)
plt.plot(ts_log)
plt.show()


# %%
#source: https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/ 
#esto es para calcular el error
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# monkey patch around bug in ARIMA class, source: https://machinelearningmastery.com/save-arima-time-series-forecasting-model-python/
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
ARIMA.__getnewargs__ = __getnewargs__

def evaluate_arima_model(ts_values, arima_order):
    size = int(len(ts_values) * 0.60)
    train, test = ts_values[0:size], ts_values[size:len(ts_values)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        #prediction VS expected(test[t])
        predictions.append(yhat)
        #this step is based on 'Walk Forward Validation': 
        #  https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/ 
        # cada nueva prediction va incluyendo en su train set un nuevo elemento de test set
        history.append(test[t])
        #print('expected test value: ', test[t], 'prediction: ', predictions[t])
        
    error = mean_squared_error(test, predictions)
    print('order: ', arima_order, ' error: ',error)
    return error


# %%
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


# %%
import warnings
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")

# %% [markdown]
# *Para realizar el grid search de los hiper params, pruebo a hacerlo tanto con la ts original como con la ts_log*

# %%
evaluate_models(ts.values, p_values, d_values, q_values)

# %% [markdown]
# ## <font color='green'> Y aquÃ­ es donde veo que hacer la serie estacionaria antes de meterla en el ARIMA object puede ir mejor </font>
# 
# fuente: https://machinelearningmastery.com/time-series-data-stationary-python/ 
# Does the statsmodel python library require us to convert the series into stationary series before feeding the series to any of the ARMA or ARIMA models ?
# 
# REPLY Jason Brownlee:
# Ideally, I would. The model can difference to address trends, but I would recommend explicitly pre-processing the data before hand. This will help you better understand your problem/data.
# 
# Primero pruebo con <font color='green'>ts_log</font>:

# %%
evaluate_models(ts_log.values, p_values, d_values, q_values)

# %% [markdown]
# ### <font color='green'> Veo claramente errores muy bajos, me quedo con la ts_log por ahora; podrÃ­a incluso probar a diferenciarla, aunque en teorÃ­a eso lo hace ya el param. d... </font>
# %% [markdown]
# ## 1.- fittedvalues VS forecast(), cuÃ¡l me da las predicciones escaladas correctamente?

# %%
def computeCost_as_MSE(X, Y):  #en teorÃ­a anÃ¡logo a usar mean_squared_error de sklearn-metrics
    inner = np.power((X - Y), 2)
    cost = (np.sum(inner))/(len(X))
    return cost


# %%
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings("ignore")
model = ARIMA(ts_log, order= (10, 2, 0))  
ARIMA_model = model.fit()  

residuals_estimados = np.array(ARIMA_model.fittedvalues)
residuals_reales = np.array(ts_log.values)

mean_squared_error(residuals_estimados, residuals_reales[2:]) #ya que se hace doble differencing


# %%
#y si pruebo con (8, 1, 1)
model = ARIMA(ts_log, order= (8, 1, 1))  
ARIMA_model = model.fit()  

residuals_estimados = np.array(ARIMA_model.fittedvalues)
residuals_reales = np.array(ts_log.values)

mean_squared_error(residuals_estimados, residuals_reales[1:]) #ya que se hace una differencing


# %%
# save model
ARIMA_model.save('ARIMA_model.pkl')


# %%
plt.plot(residuals_reales, color = 'green')
plt.plot(residuals_estimados, color='red')
plt.show()


# %%
residuals_reales[-6:]


# %%
residuals_estimados[-6:]

# %% [markdown]
# *fitted values no escalados, sino segÃºn se haya hecho o no una differencing*

# %%
#haciendo cumsum obtendrÃ© el estimado[-1]
residuals_reales[-2]+residuals_estimados[-1]


# %%
#y serÃ¡ aprox. igual a:
residuals_reales[-1]

# %% [markdown]
# ### Y ahora pruebo con '.predict()'

# %%
from statsmodels.tsa.arima_model import ARIMA
import warnings

series = ts_log['#Passengers'].apply(lambda x: float(x))
arima_order = (8, 1, 1)
size = int(len(series) * 0.80)
train, test = series[0:size], series[size:len(series)]
history = [x for x in train]
predictions = list()
model = ARIMA(history, order=arima_order)
model_fit = model.fit(disp=0)
start_index = len(train)
end_index = start_index + 1
model_forcasting = model_fit.predict(start=start_index, end=end_index)  #model_forcasting = model_fit.forecast() 
yhat = model_forcasting[0]
print(yhat)


# %%
print(train[-1]+yhat)

# %% [markdown]
# ### <font color='green'>conclusiÃ³n</font>: los 'fittedvalues' Y 'predict()' no estÃ¡n escalados segÃºn la serie de entrada, sino acorde a la d (differencing) segÃºn sea 0 o no <p>
#     
# ### ahora pruebo con forecast(), algunos apuntes a tener en cuenta: <p>
#     - ARIMA() recibe el time series dataframe (Ã­ndices son las fechas y los correspondientes values) <p>
#     - history es igual que train, un dataframe al que vamos aÃ±adiendo (walk forward validation) los Ãºltimos registros <p>
#     - test es otro dataframe con fecha-valor al que hay que acceder con iloc si es por Ã­ndices <p>
#     - para ver resultados finales en un dataframe, usamos ts_df

# %%
#dataframe para almacenar histÃ³rico, test values y forecasts:
ts_df = pd.DataFrame()
ts_df['history'] = history
ts_df['test'] = np.nan
ts_df['forecast'] = np.nan


# %%
ts_df.tail()


# %%
arima_order = (8, 1, 1)
size = int(len(ts_log) * 0.80)
train, test = ts_log[:size], ts_log[size:]  #ambos son dataframes con filas fecha-valor

history = train.copy()
forecasts = [] 
for t in range(len(test)):
    model = ARIMA(history, order=arima_order)
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    forecasts.append(yhat)
    #this step is based on 'Walk Forward Validation': 
    #  https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/ 
    # cada nueva prediction va incluyendo en su train set un nuevo elemento de test set
    history = history.append(test.iloc[t])
    
    print('expected test value: ', test.iloc[t]['#Passengers'], '   prediction: ', float(forecasts[t]))
    
    #error = mean_squared_error(test, predictions)
    


# %%
#relleno df
ts_df = pd.DataFrame(columns=['history', 'test', 'forecast'], index=history.index)
ts_df['history'] = history.values
ts_df['test'][-len(test):] = test['#Passengers']
ts_df['forecast'][-len(forecasts):] = [f[0] for f in forecasts]
#seteo las fechas como los index
#ts_df.set_index(history.index, inplace=True)
ts_df

# %% [markdown]
# ### Los '.forecast()' parecen devolverse en su escala original, con: predicciÃ³n, std y el intervalo de confianza. <font color = 'red'>Incluir tambiÃ©n el intervalo de confianza! </font>
# 
# %% [markdown]
# ## Confirmo que: <p>
#     - forecast() devuelve los datos en la escala original aunque se haga differencing
#     - predict() devuelve las predicciones sin reescalar los datos si se ha hecho differencing (como aquÃ­ donde d > 0)
#     - con 'fittedvalues' ocurre como con predict(), que no se devuelven en su escala original
# %% [markdown]
# ## REESCALAR PARA QUITAR ESCALA LOG. Y PLOTEAR LA INFO QUE TENGO EN EL DATAFRAME , CALCULAR SU MSE ETC

# %%
#valores no nulos de test (y de forecast)
test_not_nan_mask = pd.isna(ts_df.test)==False
ts_df[test_not_nan_mask]


# %%
from sklearn.metrics import mean_squared_error

test_with_values = ts_df.test[pd.isna(ts_df.test)==False]
forecasts_with_values = ts_df.forecast[pd.isna(ts_df.forecast)==False]

mean_squared_error(test_with_values, forecasts_with_values) #(np.array(test), np.array(predictions))


# %%
plt.plot(test_with_values, color = 'blue')
plt.plot(forecasts_with_values, color='red')
plt.show()

# %% [markdown]
# ### Y ahora reescalo con np.exp()

# %%
ts_df.history = np.exp(ts_df.history)


# %%
ts_df.test = [np.exp(test_val) for test_val in ts_df.test.values]


# %%
ts_df.forecast = [np.exp(forecast_val) for forecast_val in ts_df.forecast.values] 


# %%
ts_df.tail()


# %%
from sklearn.metrics import mean_squared_error

test_with_values = ts_df.test[pd.isna(ts_df.test)==False]
forecasts_with_values = ts_df.forecast[pd.isna(ts_df.forecast)==False]

mean_squared_error(test_with_values, forecasts_with_values) #(np.array(test), np.array(predictions))

# %% [markdown]
# *este error cuadrÃ¡tico medio significa, haciÃ©ndolo la raiz, que de media tenemos un error en torno a 33 pasajeros de diferencia en nuestras predicciones*

# %%
np.sqrt(1148.28)


# %%
plt.plot(ts_df.history, color = 'blue')
#plt.plot(ts_df.test, color = 'blue')
plt.plot(ts_df.forecast[ts_df.forecast.index > '1959'], color='red')
plt.show()


# %%
def computeRMSE(X, Y):  
    inner = np.power((X - Y), 2)
    return np.sqrt(np.sum(inner) / (len(X)))

print('RMSE: ', computeRMSE(ts_df.history[ts_df.history.index > '1959'], ts_df.forecast[ts_df.forecast.index > '1959']))

# %% [markdown]
# ## Esto parece mejorar la predicciÃ³n hecha en 'Time_Series_Forecast_air_passengers-Copy2'
# %% [markdown]
# ### Siguientes pasos: <p>
#     - probar mÃ¡s formas de afinar en ARIMA (pretratado no sÃ³lo con log() por ej) <p>
#     - representar el intervalo de confianza, ademÃ¡s de aÃ±adirlo al dataframe
#     - compararlo con prophet <p>
#     - ir a otros mÃ©todos como LSTM 

# %%



