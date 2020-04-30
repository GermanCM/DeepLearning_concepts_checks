
#%%[markdown]
### Pruebo con keras tuner sklearn un clasificador de ejemplo
#%%
# Bayesian optimization on a classifier with Keras tuner scikit-learn:
import kerastuner as kt
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
'''
COMPROBAR ESTE EJEMPLO CON UN DATASET DE REGRESIÓN
'''
def build_model(hp):
  model_type = hp.Choice('model_type', ['random_forest', 'ridge'])
  if model_type == 'random_forest':
    model = ensemble.RandomForestClassifier(
        n_estimators=hp.Int('n_estimators', 10, 50, step=10),
        max_depth=hp.Int('max_depth', 3, 10))
  else:
    model = linear_model.RidgeClassifier(
        alpha=hp.Float('alpha', 1e-3, 1, sampling='log'))
  return model

tuner = kt.tuners.Sklearn(
    oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective('score', 'max'),
        max_trials=10),
    hypermodel=build_model,
    scoring=metrics.make_scorer(metrics.accuracy_score),
    cv=model_selection.StratifiedKFold(5),
    directory=os.path.normpath('C:/'),
    project_name='keras_tuner_sklearn_classifier_example_1')

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

tuner.search(X_train, y_train)

#%%
# test prediction with one of the trialed models: 
best_model = tuner.get_best_models(num_models=1)[0]
best_model.predict(X_test[2].reshape(1, -1))


#%%
# Check of the score value on the validation set:
max_trials=10
trials_scores = {}
fit_models = tuner.get_best_models(num_models=max_trials)
for trial_i in range(max_trials):
  model_trial_name = 'model_trial_'+str(trial_i)
  model_trial = fit_models[trial_i]
  model_trial_preds = model_trial.predict(X_test)
  model_trial_score = metrics.accuracy_score(y_test, model_trial_preds) #, multi_class='ovr')
  # formamos diccionario con info de modelos entrenados
  model_trial_dict={'model_params': fit_models[trial_i], 'validation_score': model_trial_score}

  trials_scores[model_trial_name]=model_trial_dict

trials_scores

# %%
test_pred = best_model.predict(X_test[2].reshape(1, -1))
print('test prediction of the best model: {}'.format(test_pred))
# %%
#############################################################################
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
          project_name='sklearn_bayesian_opt_time_series_example')

tuner.search(train_x, train_y) #, validation_split=0.2,verbose=1) #epochs=n_epochs,

#%%
model = tuner.get_best_models(num_models=1)



# %%
