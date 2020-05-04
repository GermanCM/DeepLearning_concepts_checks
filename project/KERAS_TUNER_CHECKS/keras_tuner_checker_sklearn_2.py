#%%
#INFO: https://keras-team.github.io/keras-tuner/#keras-tuner-includes-pre-made-tunable-applications-hyperresnet-and-hyperxception
import os
import kerastuner as kt
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
#%%[markdown]
### Ahora pruebo, con keras tuner sklearn, una red neuronal
n_input = 24
def build_model(hp):
  import tensorflow
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.models import Sequential
  from tensorflow.keras import layers
  # define model
  model = Sequential()
  model.add(layers.Dense(input_dim=n_input,
                          units=hp.Int('units',
                                      min_value=32,
                                      max_value=512,
                                      step=32), 
                          activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mse', #metrics=[metrics.mean_squared_error], 
                optimizer=tensorflow.keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])))

  return model 

### Tuner definition & search:
from sklearn.metrics import make_scorer, mean_squared_error
import pandas as pd
from pandas import read_csv

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
  import pandas as pd

  df = pd.DataFrame(data)
  cols = list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
      cols.append(df.shift(i))
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
      cols.append(df.shift(-i))
  # put it all together
  agg = pd.concat(cols, axis=1)
  # drop rows with NaN values
  agg.dropna(inplace=True)
  
  return agg.values  


#data source: https://datasetsearch.research.google.com/search?query=univariate%20time%20series&docid=Z2B66b7T3lUIl0y6AAAAAA%3D%3D&filters=bm9uZQ%3D%3D&property=aXNfYWNjZXNzaWJsZV9mb3JfZnJlZQ%3D%3D
ev_sales_data = read_csv(r'.\datasets\china_electric_vehicles_sales.csv')

ev_sales_data['Date'] = pd.to_datetime(ev_sales_data['Year/Month'])
ev_sales_data.set_index('Date', inplace=True)
ev_sales_data.drop(columns=['Year/Month'], inplace=True)
sales_series_values = ev_sales_data['sales'].values

data = series_to_supervised(sales_series_values, n_input)
train_x, train_y = data[:, :-1], data[:, -1]

#%%
tuner = kt.tuners.Sklearn(
          oracle=kt.oracles.BayesianOptimization(
              objective=kt.Objective('score', 'min'),
              max_trials=10),
          hypermodel=build_model,
          scoring=make_scorer(mean_squared_error),
          ###adaptación propia para poder almacenar modelos H5 mediante este keras_tuner_sklearn
          is_keras_model=True,
          ###
          #metrics=mean_squared_error,
          #cv=model_selection.StratifiedKFold(5),
          directory=os.path.normpath('C:/'),
          project_name='sklearn_bayesian_opt_time_series_example_100_ep_10_max_trials',
          overwrite=True)
#%%

tuner.search(train_x, train_y, epochs=100, batch_size=10) #, validation_split=0.2,verbose=1) #epochs=n_epochs,

#%%
models = tuner.get_best_models(num_models=-1)
print('trained model configs: {}'.format(len(models)))

#%%
# load a model via its trial_ID
#model_with_trial_id = tuner.load_model_via_trial_id(trial_id='38722ac9a36a9cd332d55ee5ece252d8')


# %%
models[1].predict(train_x[3].reshape(1, -1))

# %%
train_y[3]

# %%
y_preds_0 = models[0].predict(train_x)
y_preds_1 = models[1].predict(train_x)
y_preds_2 = models[2].predict(train_x)
y_preds_3 = models[3].predict(train_x)
y_preds_4 = models[4].predict(train_x)
y_preds_5 = models[5].predict(train_x)
y_preds_6 = models[6].predict(train_x)

model_trial_score = metrics.mean_squared_error(train_y, y_preds_0)
print('rmse on train data: {}'.format(model_trial_score))
model_trial_score = metrics.mean_squared_error(train_y, y_preds_1)
print('rmse on train data: {}'.format(model_trial_score))
model_trial_score = metrics.mean_squared_error(train_y, y_preds_2)
print('rmse on train data: {}'.format(model_trial_score))
model_trial_score = metrics.mean_squared_error(train_y, y_preds_3)
print('rmse on train data: {}'.format(model_trial_score))
model_trial_score = metrics.mean_squared_error(train_y, y_preds_4)
print('rmse on train data: {}'.format(model_trial_score))
model_trial_score = metrics.mean_squared_error(train_y, y_preds_5)
print('rmse on train data: {}'.format(model_trial_score))
model_trial_score = metrics.mean_squared_error(train_y, y_preds_6)
print('rmse on train data: {}'.format(model_trial_score))
