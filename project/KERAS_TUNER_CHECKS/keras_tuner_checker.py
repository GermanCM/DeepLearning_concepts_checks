#%%
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

#%%[markdown]
#### Test on IRIS dataset
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

iris_ds = load_iris()
X = iris_ds.data
# convert integers to dummy variables (i.e. one hot encoded)
y_one_hot_encoded = to_categorical(iris_ds.target)
y_one_hot_encoded[:4]
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot_encoded, test_size=0.33, random_state=42)


#%%[markdown]
'''
Usage: the basics
Here's how to perform hyperparameter tuning for a single-layer dense neural network using random search.

First, we define a model-building function. It takes an argument hp from which you can sample hyperparameters, such as hp.Int('units', min_value=32, max_value=512, step=32) (an integer from a certain range).

This function returns a compiled model.
'''
#%%
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(input_dim=X.shape[1],
                            units=hp.Int('units',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                            
                           activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model
#%%[markdown]
'''
Next, instantiate a tuner. You should specify the model-building function, the name of the objective to optimize (whether to minimize or maximize is automatically inferred for built-in metrics), the total number of trials (max_trials) to test, and the number of models that should be built and fit for each trial (executions_per_trial).

Available tuners are RandomSearch and Hyperband.

Note: the purpose of having multiple executions per trial is to reduce results variance and therefore be able to more accurately assess the performance of a model. If you want to get results faster, you could set executions_per_trial=1 (single round of training for each model configuration).
'''

# %%
import os

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    #directory=r'./KERAS_TUNER_CHECKS',
    #directory=os.path.normpath('.\keras_tuner_checks_logs'),
    directory=os.path.normpath('C:/'),
    project_name='iris_data_keras_tuner_test')

#%%[markdown]
'''
summary of the search space:
'''
# %%
tuner.search_space_summary()


# %%[markdown]
'''
Start the search for the best hyperparameter configuration. The call to search has the same signature as model.fit().
Here's what happens in search: models are built iteratively by calling the model-building function, which populates the hyperparameter space (search space) tracked by the hp object. The tuner progressively explores the space, recording metrics for each configuration.
'''

# %%
tuner.search(X_train, y_train,
             epochs=5,
             validation_data=(X_test, y_test)
             #validation_split=0.2,verbose=1)
             )

#%%
tuner.results_summary()

#%%[markdown]
### Best models achieved with the hyperparametrization
models = tuner.get_best_models(num_models=2)

# %%[markdown]
### Evaluation score:
print('single prediction on a test instance: {}'.format(models[0].predict(X_test[-1].reshape(1, -1)))) 

# %%[markdown]
### Predictions on the test set:
test_set_predictions = models[0].predict(X_test)

# %%[markdown]
#### Esperamos obtener un score = 0.90 para el best_model_0 según el 'Results summary'
import tensorflow

categ_acc_test_set = tensorflow.keras.metrics.CategoricalAccuracy()

#best_model_0_eval_set_acc = tensorflow.keras.metrics.CategoricalAccuracy(y_test.reshape(-1,3), test_set_predictions.reshape(-1,3))
_ = categ_acc_test_set.update_state(y_test, test_set_predictions)
print('best_model_0_eval_set_acc: {}'.format(categ_acc_test_set.result().numpy()))

# %%[markdown]
#### Y con el train set?
categ_acc_train_set = tensorflow.keras.metrics.CategoricalAccuracy()

train_set_predictions = models[0].predict(X_train)
_ = categ_acc_train_set.update_state(y_train, train_set_predictions)
print('best_model_0_train_set_acc: {}'.format(categ_acc_train_set.result().numpy()))

# %%[markdown]
#### Obtengo resultado de accuracy mejor que el 0.90 esperado, pruebo si al menos el mejor modelo es el [0]
import tensorflow

models = tuner.get_best_models(num_models=5)

test_set_predictions_0 = models[0].predict(X_test)
test_set_predictions_1 = models[1].predict(X_test)
test_set_predictions_2 = models[2].predict(X_test)
test_set_predictions_3 = models[3].predict(X_test)
test_set_predictions_4 = models[4].predict(X_test)
#%%
categ_acc_test_set = tensorflow.keras.metrics.CategoricalAccuracy()

#best_model_0_eval_set_acc = tensorflow.keras.metrics.CategoricalAccuracy(y_test.reshape(-1,3), test_set_predictions.reshape(-1,3))
_ = categ_acc_test_set.update_state(y_test, test_set_predictions_0)
print('best_model_0_eval_set_acc: {}'.format(categ_acc_test_set.result().numpy()))

_ = categ_acc_test_set.update_state(y_test, test_set_predictions_1)
print('best_model_0_eval_set_acc: {}'.format(categ_acc_test_set.result().numpy()))

_ = categ_acc_test_set.update_state(y_test, test_set_predictions_2)
print('best_model_0_eval_set_acc: {}'.format(categ_acc_test_set.result().numpy()))

_ = categ_acc_test_set.update_state(y_test, test_set_predictions_3)
print('best_model_0_eval_set_acc: {}'.format(categ_acc_test_set.result().numpy()))

_ = categ_acc_test_set.update_state(y_test, test_set_predictions_4)
print('best_model_0_eval_set_acc: {}'.format(categ_acc_test_set.result().numpy()))


# %%
models = tuner.get_best_models(num_models=1)
test_set_predictions_0 = models[0].predict(X_test)
categ_acc_test_set = tensorflow.keras.metrics.CategoricalAccuracy()
_ = categ_acc_test_set.update_state(y_test, test_set_predictions_0)
print('best_model_0_eval_set_acc: {}'.format(categ_acc_test_set.result().numpy()))


# %%[markdown]
### Hasta aquí hemos podido utilizar un RANDOM_SEARCH hiperparametrizador keras tuner
#### ToDo: acceder a los pesos y arquitectura del modelo


# %%
