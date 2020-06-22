# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Etiquetas para revisar y ordenar luego el contenido:
# * #Unit_test
# * #Hipothesis
# * #To_check
# * #Prio
# * #Nice_to_have
# * #Data_quality_check
# %% [markdown]
#### Fase exploratoria del dataset, donde se mirarán aspectos como:
# * calidad del dataset (missing values? valores atípicos que no tengan sentido con la naturaleza del atributo? cantidad suficiente de histórico? ...)
# * representación gráfica de las series temporales para algunos clientes; de aquí esperamos obtener alguna intuición 
# * análisis estadístico descriptivo inicial
# * generación de algunas hipótesis que puedan ser útiles para el enfoque a negocio
#### Transformaciones del dataset, por ejemplo:
# * Imputación de valores nulos por algún valor que no afecte a la integridad del dataset
# * Re-escalado de valores (especialmente importante para algunos algoritmos que se pudieran emplear, menos relevante para otros como los basados en árboles de decisión)
# * Re-estructuración del dataset: según cómo enfoquemos el modelo a construir, será necesario modificar la estructura inicial del dataset (resampleo? generación de nuevos atributos? etc)
# * Comprobaciones del tipo: presenta nuestra serie temporal autocorrelación (univariable)? es ruido blanco? es estacionaria (más importante según el método que apliquemos)? 
#### Fase modelado:
# * enfoque de posibles formas de modelar un predictor, con especial atención al interés en predecir futuros valores de movimeintos de ciertas categorías de interés 
# * perfilar usuarios en segmentos que puedan ser de utilidad, tanto para la construcción de un modelo como para la interpretación de negocio (hay tipos similares de clientes?)
# * posibles casos de uso que podría surgir de todo el proceso de análisis y modelado de este dataset
#### Estrategia de validación:
# * podría ser útil pensar en alguna métrica de validación (a parte de las genéricas a utilizar cuando modelamos) que se ajuste a lo que perseguimos de cara a optimizar la utilidad de nuestros modelos

# %%
class ExploreDataset():
    def __init__(self):
        self.dataset_ = None

    def read_dataset_from_csv(self, path, sep=',', number_of_rows=None):
        """Lee los datos y los almacena en un dataframe; otra versión de este método incluiría posibilidad de extraer otro orden de magnitud de datos con PySpark, Dask o Vaex

           Args:
               path ([string]): ruta de acceso al archivo 
               sep ([string]): separador de valores en el csv
               number_of_rows ([int]): número deseado de registros en caso de querer limitar la extracción
           Returns:
               Pandas dataframe    
        """
        try:
            import pandas as pd

            if number_of_rows is not None: 
                retrieved_df = pd.read_csv(filepath_or_buffer=path, sep=sep, nrows=number_of_rows)
            else:
                retrieved_df = pd.read_csv(filepath_or_buffer=path, sep=sep)

            self.dataset_=retrieved_df

            return retrieved_df
        except Exception as exc:
            #log error with logger
            print(exc)
            return exc

    def generate_date(self, year=None, month=None, day=28):
        """Crea un valor de fecha tipo datetime

           Args:
               year ([int]): año 
               month ([int]): mes
               day ([int]): día
           Returns:
               Fecha de tipo datetime    
        """
        try:
            from datetime import datetime

            return datetime(year, month, day)
            
        except Exception as exc:
            #log error with logger
            print(exc)
            return exc

    def add_date_attribute(self, year_column_name, month_column_name=None, day_column_name=28):
        try:
            """Crea nuevo atribute date para visualizar la serie entre otras posibles utilidades
            
               Args:
                   year_column_name ([string]): nombre del atributo año
                   month_column_name ([string]): nombre del atributo mes 
                   day_column_name ([string]): nombre del atributo día
                    
               Returns:
               
            """
            import pandas as pd 

            self.dataset_['date'] = pd.Series(self.dataset_.index).apply(lambda idx: self.generate_date(self.dataset_.iloc[idx][year_column_name], 
                                        self.dataset_.iloc[idx][month_column_name])) 

        except Exception as exc:
            #log error with logger
            print(exc)
            return exc


    def plot_timeseries_values(self, attribute_to_plot, category_type='all', mask_attribute_name=None, mask_attribute_value=None, date_column_name=None, graph_title=None):
        """Representa la serie temporal de los valores correspondientes a: 
           - atributo indicado del dataframe, o
           - valores del atributo indicado para los registros que cumplan el filtro indicado
            
           Args:
               attribute_to_plot ([string]): nombre del atributo a representar 
               category_type ([string]): si 'all' consumo agregado, si no es 'all' debería ser un tipo de categoría válido  
               attribute_name ([string]): nombre de la columna a representar 
               attribute_mask_value ([]): valor condición para extraer los registros deseados
               date_column_name ([string]): nombre de la columna con los valores de fecha
               graph_title ([string]): nombre de la gráfica a representar
           Returns:
               
        """
        try:
            import plotly.express as px
            
            dataset = self.dataset_
            if (mask_attribute_name is not None)&(mask_attribute_value is not None):
                client_mask = self.dataset_[mask_attribute_name]==mask_attribute_value
                dataset=dataset[client_mask]

            if category_type is not None:
                category_mask = self.dataset_['categoryDescription']==category_type
                dataset_by_category = dataset[category_mask]
                sorted_dataset = dataset_by_category.sort_values(by=date_column_name, ascending=True)
            else:
                sorted_dataset = dataset.sort_values(by=date_column_name, ascending=True)
            
            fig = px.line(sorted_dataset, x=date_column_name, y=attribute_to_plot, title=graph_title)
            fig.show()

        except Exception as exc:
            #log error with logger
            print(exc)
            return exc

    def plot_bar_values(self, dataset, x_attribute, y_attribute):
        """Representa como histograma los ratios de aparición de cada tipo de movimiento  
            
           Args:
               dataset ([string]): dataset que contiene los atributos de interés
               x_attribute ([string]): atributo eje x
               y_attribute ([string]): atributo eje y 
           Returns:
               
        """
        try:
            import plotly.express as px

            fig = px.bar(dataset, x=x_attribute, y=y_attribute)
            fig.show()

        except Exception as exc:
            #log error with logger
            print(exc)
            return exc

    def check_if_gaussian(self, dataset, selected_attributes):
        """Comprueba si los atributos indicados del dataset siguen una distribución gausiana  
            
           Args:
               dataset ([string]): dataset que contiene los atributos de interés
               selected_attributes ([string]): nombre de los atributos a evaluar
           Returns:
               listas de nombres de atributos con distribución gaussiana y no gaussiana según el test de Saphiro  
        """
        try:
            gaussian_attributes = []
            non_gaussian_attributes = []
            for col in selected_attributes:
                # normality test
                stat, p = shapiro(dataframe[col].values)
                print('Statistics=%.3f, p=%.3f' % (stat, p))
                # interpret
                alpha = 0.05
                if p > alpha:
                    print('Sample looks Gaussian (fail to reject H0)')
                    gaussian_attributes.append(col)
                else:
                    print('Sample does not look Gaussian (reject H0)')
                    non_gaussian_attributes.append(col)

            return gaussian_attributes, non_gaussian_attributes

        except Exception as exc:
            #log error with logger
            print(exc)
            return exc
    
#%%
class ProcessDataset():
    def __init__(self):
        self.dataset_ = None
        self.supervised_format_dataset_ = None

    def check_if_nan(self, x):
        """Comprueba posibles valores ausentes

           Args:
               x ([number]): value
           Returns:
               boolean value    
        """
        try:
            import numpy as np 
            import math

            is_nan = math.isnan(x)             

            return is_nan
        except Exception as exc:
            #log error with logger
            print(exc)
            return exc   

    def impute_missing_value(self, dataframe, value_to_set):
        """Sustituye valores nulos del dataframe por el valor indicado

           Args:
               x ([number]): valor a setear
               dataframe ([dataframe]): dataframe con valores nulos 
           Returns:
               dataframe sin valores nulos
        """
        try:
            import pandas as pd

            dataframe=dataframe.applymap(lambda x: value_to_set if pd.isna(x) else x)
            return dataframe
            
        except Exception as exc:
            #log error with logger
            print(exc)
            return exc

    def build_client_dataset(self, client_ID):
        """Construye el dataset de entrenamiento para un cliente dado

            Args:
                client_ID ([int]): identificador del cliente
            Returns:
                dataset del cliente con tantos registros como meses tenga su histórico, con las categorías 
                de movimeintos como atributos 
        """
        try:
            import pandas as pd

            client_mask = self.dataset_['associatedAccountId']==client_ID
            this_client_ds = self.dataset_[client_mask]
            this_client_ds_sorted = this_client_ds.sort_values(by='date', ascending=True)  
            this_client_ds_sorted.set_index('date', inplace=True)
            # hasta aquí tenemos el dataset del histórico de este cliente
            
            transp_ds_this_client_all=pd.DataFrame()
            # ahora, para cada fecha en this_client_ds_sorted, se forma el dataset con la estructura deseada:  # estos buclkes 'for' se pueden "eficientar"
            for this_date in this_client_ds_sorted.index.unique():
                date_mask = this_client_ds_sorted.index==this_date
                these_client_date_spents_df = this_client_ds_sorted[date_mask].set_index('categoryDescription')[['monthlySpent']]
                transp_these_client_date_spents_df = these_client_date_spents_df.T
                transp_these_client_date_spents_df.index=pd.Series(this_date)
                transp_these_client_date_spents_df['associatedAccountId']=int(this_client_ds_sorted[date_mask]['associatedAccountId'][0])
                transp_these_client_date_spents_df['year']=this_client_ds_sorted[date_mask]['year'][0]
                transp_these_client_date_spents_df['month']=this_client_ds_sorted[date_mask]['month'][0]
                # añadimos al dataset global de este cliente
                transp_ds_this_client_all=transp_ds_this_client_all.append(transp_these_client_date_spents_df)

            # confirmamos que se trata de un solo cliente 
            assert len(transp_ds_this_client_all['associatedAccountId'].unique())==1 
            return transp_ds_this_client_all
            
        except Exception as exc:
            #log error with logger
            print(exc)
            return exc

    def series_to_supervised(self, dataset, n_in, n_out=1):
        #source: Jason Brownlee's time series book
        try:
            import pandas as pd 
            df = pd.DataFrame(dataset)
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

        except Exception as exc:
                #log error with logger
                print(exc)
                return exc


class IteratorHelper():
    '''
    Ejemplo que, a partir de un dataframe original de 2 columnas, genera una lista de todas las posibles combinaciones 
    de dichas columnas; una vez obtenidas esas combinaciones, trabajamos con todas las posibles combinaciones de los 
    dos arrays (cliente y producto en este caso), en lugar de realizar un bucle 'for' anidado
    '''
    def __init__(self):
        import pandas as pd
        import itertools

        self

    def make2DCartesianProduct(self, array_x, array_y, x_name='x', y_name='y'):
        try:
            return pd.DataFrame.from_records(itertools.product(array_x.reshape(-1, ), array_y.reshape(-1, )), 
                                            columns=[x_name, y_name])
        except Exception as exc:
            #log error with logger
            print(exc)
            return exc

    def make4DCartesianProduct(self, array_x, array_y, array_z, x_name='x', 
                               y_name='y', z_name='z'):
        try:
            return pd.DataFrame.from_records(itertools.product(array_x.reshape(-1, ), array_y.reshape(-1, ), 
                                            array_z.reshape(-1, )), columns=[x_name, y_name, z_name])
        except Exception as exc:
            #log error with logger
            print(exc)
            return exc

# %%
file_path = r'.\data\movementsSample.csv'

explorer_obj = ExploreDataset()
explorer_obj.read_dataset_from_csv(file_path)


# %%
print('categorías de movimientos: {}'.format(explorer_obj.dataset_.categoryDescription.unique()))

# %% [markdown]
# #Hipótesis --> habrá unas categorías cuya distribución temporal sea mucho más constante que otras, por ejemplo: se espera que la categoría 'MENAJE DEL HOGAR Y ELECTRÓNICA' no presente un claro patrón de consumo como sí lo deberíamos ver en 'GAS Y ELECTRICIDAD' <p>
# #Prio --> calcular el techo de gasto mediano (y su std) para cada cliente; en base a este valor, podremos ajustar las estimaciones de consumo de un usuario a final de mes al menos para los que presenten un gasto uniforme (esto podría ser otro criterio de segmentación de tipo de clientes)
# 
# %% [markdown]
# #Data_quality_check --> tenemos el mismo número de IDs de categorías que de descripciones?

# %%
assert len(explorer_obj.dataset_['categoryDescription'].unique())==len(explorer_obj.dataset_['categoryId'].unique())


# %%
print('número de valores distintos de categoryDescription : {} y número de valores distintos de categoryId : {}'.format(len(explorer_obj.dataset_['categoryDescription'].unique()), len(explorer_obj.dataset_['categoryId'].unique()))) 

# %% [markdown]
# ### Comprobamos qué categoryId nos sobra; para ello, vemos las combinaciones de description e ID:

# %%
category_desc__ID_combinations = explorer_obj.dataset_[['categoryId', 'categoryDescription']].drop_duplicates()
#category_desc__ID_combinations.categoryId.value_counts()
category_desc__ID_combinations.categoryDescription.value_counts()

# %% [markdown]
# #### Vemos que la descripción duplicada corresponde a 'SIN CLASIFICAR', por lo que no se trata de una duplicidad a corregir
# #### Podría ser interesante mirar la cuantía de movimientos no clasificados, en caso de que sea alta y de posible interés a identificar 

# %%
# añadimos este dataset de combinaciones a nuestro objeto exploratorio:
explorer_obj.category_desc__ID_combinations = category_desc__ID_combinations

if explorer_obj.category_desc__ID_combinations.to_dict==category_desc__ID_combinations.to_dict:
    category_desc__ID_combinations=None

# %% [markdown]
# ### Representamos los valores de alguna serie temporal

# %%
#Unit_test 
from datetime import datetime

assert datetime(2017, 1, 1, 0, 0)==explorer_obj.generate_date(2017, 1, 1) 

# %%
#explorer_obj.dataset_['date'] = explorer_obj.add_date_attribute(year_column_name='year', month_column_name='month')
import pandas as pd 
import time

init_time = time.time()
explorer_obj.dataset_['date'] = pd.Series(explorer_obj.dataset_.index).apply(lambda idx: explorer_obj.generate_date(explorer_obj.dataset_.iloc[idx]['year'], explorer_obj.dataset_.iloc[idx]['month'])) 
#Nice_to_have --> EFICIENTAR

print('process time: {}'.format((time.time() - init_time)))
explorer_obj.dataset_.tail(5)

# %%
nominas_mask = explorer_obj.dataset_['categoryDescription']=='NÓMINAS'
dataset_nominas = explorer_obj.dataset_[nominas_mask]
assert len(dataset_nominas)>0
print('número de registros con tipo de movimiento ''NOMINA'': {}'.format(len(dataset_nominas)))

# %% [markdown]
# ### Contamos el porcentaje de clientes que tienen algún movimiento para cada categoría: 
# %%
categories = explorer_obj.category_desc__ID_combinations.categoryDescription.values
clients_number = len(explorer_obj.dataset_.associatedAccountId.unique())
category_freqs_df = pd.DataFrame(columns=['category', 'usage_ratio'])
for category in categories:
    category_mask = explorer_obj.dataset_['categoryDescription']==category
    dataset_with_category_type = explorer_obj.dataset_[category_mask]
    category_freqs_df = category_freqs_df.append({'category': category, 
            'usage_ratio': len(dataset_with_category_type)/clients_number}, ignore_index=True)

#%%
# HACER ESTO CON EL MÉTODO DEFINIDO EN SU CLASE
import plotly.express as px

explorer_obj.category_freqs_df_ = category_freqs_df
fig = px.bar(explorer_obj.category_freqs_df_, x='category', y='usage_ratio')
fig.show()

#%% libero memoria
category_freqs_df=None


# %% [markdown]
### #Nice_to_have 
################
'''
Y ahora calculamos dicho "ratio de uso" normalizado con el importe medio de los correspondientes movimientos; 
esto nos podría dar una idea de la relevancia de unas categorías de movimientos sobre otras, ya que no es lo 
mismo 3 movimientos de la categoría 'GAS Y ELECTRICIDAD' que 3 de la categoría 'VIAJES HOTELES Y LÍNEAS AÉREAS', 
esto nos podría ayudar a dar pesos específicos de importancia por cada categoría de movimiento...
'''
median_spents_by_category = explorer_obj.dataset_[['categoryDescription', 'monthlySpent']].groupby(by=['categoryDescription']).median()
median_spents_by_category.rename
explorer_obj.dataset_.merge(median_spents_by_category, how='right', on='categoryDescription')
############

# %% [markdown]
# represento la serie temporal de un cliente para un producto:
#client_ID=43396928
#explorer_obj.plot_timeseries_values(attribute_to_plot="monthlySpent", mask_attribute_name='associatedAccountId', mask_attribute_value=client_ID, date_column_name='date', graph_title='Nóminas cliente {}'.format(client_ID))

#%%
import plotly.express as px

client_ID=43396928
client_ID=dataset_nominas['associatedAccountId'].unique()[39]

client_mask = dataset_nominas['associatedAccountId']==client_ID
sorted_client_nominas = dataset_nominas[client_mask].sort_values(by='date', ascending=True)
fig = px.line(sorted_client_nominas, x="date", y="monthlySpent", title='Nóminas cliente {}'.format(client_ID))

fig.show()

# %% [markdown]
'''
Esos picos parecen indicar pagas extras. En los casos donde vemos esos picos pero no coinciden
con los meses esperados (junio y diciembre) y/o no son semestrales, podríamos hacer uso de análisis 
de texto del campo 'concepto' (si lo hubiera) correspondiente a tal movimiento, por ejemplo el cliente
43396928
'''

# %%
client_ID=38361820 #29292026
client_mask = dataset_nominas['associatedAccountId']==client_ID
sorted_client_nominas = dataset_nominas[client_mask].sort_values(by='date', ascending=True)
fig = px.line(sorted_client_nominas, x="date", y="monthlySpent", title='Nóminas cliente {}'.format(client_ID))
fig.show()

# %% [markdown]
#Nice_to_have
#### anomalía gordísima (generar atributo de anomalías univariables para cada registro de cada cliente en cada producto?)

#%%
# Y si ploteamos la suma de todos los movimientos para este cliente que puedan relacionarse con mayor gasto?
categorias_gastos = ['PRODUCTOS Y SERVICIOS DIGITALES', 'MODA', 'SUPERMERCADOS Y ALIMENTACIÓN', 
                     'AUTOPISTAS GASOLINERAS Y PARKINGS', 'RESTAURACIÓN', 'OCIO', 'COMPRA ONLINE']
client_id = 38361820 #29292026
client_mask = explorer_obj.dataset_['associatedAccountId']==client_id
this_client_ds = explorer_obj.dataset_[client_mask]
gastos_mask = [gasto in categorias_gastos for gasto in this_client_ds['categoryDescription'].values]
sorted_client_movs = this_client_ds[gastos_mask].sort_values(by='date', ascending=True)
fig = px.line(sorted_client_movs, x="date", y="monthlySpent", title='Suma de movimientos ocio cliente {}'.format(client_id))
fig.show()

#%%[markdown]
#Hipothesis
'''
Se aprecian máximos locales en torno a los meses de julio y diciembre--> makes sense, esto habría que corroborarlo con 
un valor relativamente alto de correlación entre tipo de gastos e ingresos <p>

En cambio para el 38361820 no existe esa correlación de picos de consumo en las categorías consideradas VS pagas extras
Vamos a probar si 'TRANSFERENCIAS DE ENTRADA' podría tener esa correlación en caso de tratarse de ahorro?
'''
#%%
categoria_transf_entrada = ['TRANSFERENCIAS DE ENTRADA']
client_id = 38361820 #29292026
client_mask = explorer_obj.dataset_['associatedAccountId']==client_id
this_client_ds = explorer_obj.dataset_[client_mask]
gastos_mask = [gasto in categoria_transf_entrada for gasto in this_client_ds['categoryDescription'].values]
sorted_client_movs = this_client_ds[gastos_mask].sort_values(by='date', ascending=True)
fig = px.line(sorted_client_movs, x="date", y="monthlySpent", title='Transferencias de entrada {}'.format(client_id))
fig.show()

#%%
'''
Vemos que el cliente 38361820 tiene movimientos de 'TRANSFERENCIAS DE ENTRADA' (en principio de bajo importe) 
hasta junio de 2019 mientras que los de nómina acabaron en 2018. Un atributo extra podría ser la variación del
número de tipos de movimiento en los últimos x meses con esta entidad bancaria, con peso específico por importe asociado; 
esto podría dar información para predicción de fuga de clientes o de presencia en la entidad
'''

#%%
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=2, cols=2, subplot_titles=("SUPERMERCADOS Y ALIMENTACIÓN", "GAS Y ELECTRICIDAD", "MODA", "RESTAURACIÓN"))

fig.add_trace(
    go.Scatter(x=client_38361820_history_set.date, y=client_38361820_history_set["SUPERMERCADOS Y ALIMENTACIÓN"], 
            mode='lines+markers+text'), row=1, col=1
)
fig.add_trace(
    go.Scatter(x=client_38361820_history_set.date, y=client_38361820_history_set["GAS Y ELECTRICIDAD"], 
            mode='lines+markers+text'), row=1, col=2
)

fig.add_trace(
    go.Scatter(x=client_38361820_history_set.date, y=client_38361820_history_set["MODA"], 
            mode='lines+markers+text'), row=2, col=1
)

fig.add_trace(
    go.Scatter(x=client_38361820_history_set.date, y=client_38361820_history_set["RESTAURACIÓN"], 
            mode='lines+markers+text'), row=2, col=2
)

fig.update_layout(margin={"r":10,"t":60,"l":10,"b":10}, height=600, width=710, showlegend=False, paper_bgcolor="#EBF2EC") 
fig.show()


#%%[markdown]
#### Construyo el dataset basado en los lag values de cada tipo de gasto, así:
# * metemos 0 en los meses donde un cliente no tiene movimiento en un tipo de gasto
# * pasamos el check de missing values
# * mencionamos que no vamos a aplicar criterio de anomalía basada en importe en ppio
# * paso el profiling apra generar el EDA report, y reviso corr matrix
# * para esto anterior, paso el check de normal distribution para coger o no la pearson corr coeff u otra no param.
# * para la variable de interés, es white noise?
# * para cada variable temporal, son estacionarias?
# * checkear la distribución de residuos tras hacer el forecast and check if white noise?
# * para abordar el problema p >> n, miraremos: selección de atributos (en ppio por correlaciones) y/o PCA
# * mencionar que dependiendo de la antelación con la que necesitemos predecir posibles gastos de movimientos, necesitaríamos crear como variable objetivo un single-step (a un mes), o etiquetar con el de x meses a futuro, o hacer un multi-step forecast 
# * mencionar que podríamos probar a agrupar categorías por super tipos de categorías

# %% [markdown]
#### Empezaría por una segmentación basada en un criterio sencillo y útil; posteriormente haría el multivariable
#### Podría segmentar clientes por tipo de ingresos? Los que presenten patrones más repetitivos con picos en torno a junio y diciembre podrían ser asalariados, mientras que otros como éste podrían ser autónomos etc?
#### Pensar en categorizar tipos de productos por relevancia: podría ser por cuantía mediana de movimientos de esa categoría, o por número agregado entre todos los clientes
#### De cara a modelar, nuestros modelos podrían validarse en base a que los intervalos de confianza de las variables predichas no excedieran el techo de gasto o movimientos (si lo hubiera)

#%%[markdown]
#### Formamos el dataset traspuesto generando un atributo por cada tipo de movimiento
# * por cada año-mes, podemos ver la evolución de tipos de movimientos; por ejempo el cliente 38361820 va aumentando el tipo de movimientos en la entidad   
# *
# *

#%%
import pandas as pd 

explorer_obj.dataset_['date'] = pd.Series(explorer_obj.dataset_.index).apply(lambda idx: explorer_obj.generate_date(explorer_obj.dataset_.iloc[idx]['year'], explorer_obj.dataset_.iloc[idx]['month'])) 

#%%
client_id = explorer_obj.dataset_['associatedAccountId'][25] #38361820
processor_obj = ProcessDataset()
#%%
processor_obj.dataset_ = explorer_obj.dataset_
#%%
this_client_history_set = processor_obj.build_client_dataset(client_id)
#%%
this_client_history_set = processor_obj.impute_missing_values(this_client_history_set, 0)
len(this_client_history_set)

#%%[markdown]
# * Queremos ver la distribución de número de meses durante los que los clientes tienen movimientos con el banco
# * Esto es relevante de cara a conocer con qué históricos contamos por cliente
#%%[markdown]
# * Una vez tengamos los datasets de cada cliente, podríamos hacer el profiling para entre otras cosas enconrtar correlaciones
# * Por cada serie de interés a predecir, podríamos comprobar posible white noise (pero recordar que puede intentarse regresión sin estructura temporal)
# * Por ahora intentamos baseline VS LSTM en un cliente con suficinete histórico

#%%[markdown]
#### Pasamos un corr coeff al histórico de un cliente
client_id = this_client_history_set['associatedAccountId'][1] # 38361820
client_id
#%%
processor_obj = ProcessDataset()
processor_obj.dataset_ = explorer_obj.dataset_
this_client_history_set = processor_obj.build_client_dataset(client_id)
this_client_history_set
#%%
this_client_history_set = processor_obj.impute_missing_values(this_client_history_set, 0)
this_client_history_set
#%%
# source: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")
# Generate a large random dataset
import numpy as np 

rs = np.random.RandomState(33)
columns_desired = list(this_client_history_set.columns)
columns_desired.remove('associatedAccountId')

d = this_client_history_set[columns_desired]
# Compute the correlation matrix
corr = d.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#%%[markdown]
#### Ahora para todos los clientes para que dichas correlaciones puedan ser significativas:
all_clients_history_set=pd.DataFrame()
clients_with_errors=list()
history_length_per_client_dict = {}
for client_id in explorer_obj.dataset_['associatedAccountId'].unique(): 
    try:
        processor_obj = ProcessDataset()
        processor_obj.dataset_ = explorer_obj.dataset_
        this_client_history_set = processor_obj.build_client_dataset(client_id)
        this_client_history_set = processor_obj.impute_missing_values(this_client_history_set, 0)
        all_clients_history_set = all_clients_history_set.append(this_client_history_set)
        history_length_per_client_dict[client_id] = len(all_clients_history_set)
    except Exception as exc:
        #clients_with_errors=clients_with_errors.append(client_id)
        print('client {} gave an error: {}'.format(client_id, exc))
        pass

history_length_per_client_dict

#%%[markdown]
# * cogería sólo los datasets de los clientes con más registros mensuales para busar mayor representatividad (info en history_length_per_client_dict)
import seaborn as sns, numpy as np
sns.set()

values = [value for key, value in history_length_per_client_dict.items()]
ax = sns.distplot(values)

#%%
selected_client_IDs = [key for key, value in history_length_per_client_dict.items() if value > 15000]

#%%
selected_clients_history_set=pd.DataFrame()
for client_ID in selected_client_IDs:
    client_mask = all_clients_history_set['associatedAccountId']==client_ID
    client_ID_df = all_clients_history_set[client_mask]
    selected_clients_history_set = selected_clients_history_set.append(client_ID_df)

#%%[markdown]
#### Obtenemos también los valores medianos (no medios) de cada atributo por cada cliente, para estudiar luego posibles correlaciones:
selected_clients_columns = list(selected_clients_history_set.columns)
selected_clients_columns.remove('associatedAccountId')

selected_clients_history_set = processor_obj.impute_missing_value(selected_clients_history_set, 0)

selected_clients_median_values = selected_clients_history_set.groupby(by='associatedAccountId')[selected_clients_columns].median()

#%%[markdown]
#### Comprobamos si los atributos considerados siguen una distribución normal
### Añadimos otro test de normalidad:

client_id = selected_clients_history_set['associatedAccountId'][5]
this_client_mask = selected_clients_history_set['associatedAccountId'] == client_id 
this_client_ds = selected_clients_history_set[this_client_mask]

#%%[markdown]
#### escogemos los atributos que no suponen un valor medio = 0
non_zero_spents_mask = selected_clients_median_values.loc[client_id] > 0
this_client_ds[non_zero_spents_mask.values]

#%%
gaussian_attributes, non_gaussian_attributes = explorer_obj.check_if_gaussian(this_client_ds,
                                                    selected_clients_columns)
#%%
print('gaussian_attributes: {}'.format(gaussian_attributes))
print('non_gaussian_attributes: {}'.format(non_gaussian_attributes))

#%%[markdown]
#### Probamos otro cliente:
client_id = selected_clients_history_set['associatedAccountId'][25]
this_client_mask = selected_clients_history_set['associatedAccountId'] == client_id 
this_client_ds = selected_clients_history_set[this_client_mask]
this_client_ds

#%%[markdown]
#### escogemos los atributos que no suponen un valor medio = 0
non_zero_selected_attributes = ['TRANSFERENCIAS DE ENTRADA', 'OPERACIONES CAJERO']

#%%
gaussian_attributes, non_gaussian_attributes = explorer_obj.check_if_gaussian(this_client_ds,
                                                    non_zero_selected_attributes)
print('gaussian_attributes: {}'.format(gaussian_attributes))
print('non_gaussian_attributes: {}'.format(non_gaussian_attributes))

#%%[markdown]
#### Parece que las distribuciones de los atributos no son gaussianas, por lo que emplearemos el método Kendall para distribuciones no paramétricas
d = this_client_ds[non_zero_selected_attributes]
# Compute the correlation matrix
corr = d.corr(method='kendall')
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#%%
d = this_client_ds[non_zero_selected_attributes]
# Compute the correlation matrix
corr = d.corr(method='spearman')
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


#%%[markdown]
'''
Vamos a crear atributos a partir de los valores pasados de cada uno de los existentes; en una estructura de serie temporal,
esperaríamos encontrar cierta correlación entre valores presentes y pasados al menos si se aprecia cierta repetición según 
la frecuencia de muestreo, mensual en este caso
'''
'''
def series_to_supervised(dataset, n_in, n_out=1):
    #source: Jason Brownlee's time series book
    try:
        import pandas as pd 
        df = pd.DataFrame(dataset)
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

    except Exception as exc:
            #log error with logger
            print(exc)
            return exc
'''
#%%[markdown]
#### Pruebo con el cliente con ID
client_id = selected_client_IDs[10]
this_client_mask = selected_clients_history_set['associatedAccountId'] == client_id 
this_client_ds = selected_clients_history_set[this_client_mask]

this_client_ds['date']=this_client_ds.index
fig = px.line(this_client_ds, x="date", y="NÓMINAS", title='Nóminas cliente {}'.format(client_ID))
fig.show()

#%%[markdown]
#### escogemos los atributos que no suponen un valor medio = 0

this_client_median_values = this_client_ds.groupby(by='associatedAccountId')[selected_clients_columns].median()
non_zero_spents_mask = this_client_median_values[0] > 0
non_zero_spents_mask
#%%
desired_columns_client = this_client_median_values.columns[non_zero_spents_mask.values[0]]
this_client_ds.columns[desired_columns_client]

#%%[markdown]
#### probamos a generar los atributos que son 1 y 2 meses de valores anteriores
#series_to_supervised(this_client_ds[['TRANSFERENCIAS DE SALIDA', 'COMPRA ONLINE',
#       'GAS Y ELECTRICIDAD']], 3, n_out=1)
data_sup=series_to_supervised(this_client_ds[['TRANSFERENCIAS DE SALIDA']], 2, n_out=1)
series_to_supervised_df = pd.DataFrame(columns=['transf_sal_past_2', 'transf_sal_past_1', 
                                                'transf_sal_past_0'], data=data_sup)

#%%
data_sup=series_to_supervised(this_client_ds[['TRANSFERENCIAS DE SALIDA', 'COMPRA ONLINE']], 2, n_out=1)
data_sup
#%%
series_to_supervised_df = pd.DataFrame(columns=['transf_sal_past_2', 'online_shop_past_2',
                                                'transf_sal_past_1', 'online_shop_past_1',
                                                'transf_sal_past_0', 'online_shop_past_0'],
                                                data=data_sup)

series_to_supervised_df

#%%[markdown]
#### Y ahora tenemos mayor correlación?
d = series_to_supervised_df
# Compute the correlation matrix
corr = d.corr(method='pearson')
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#%%[markdown]
### Check: lo está haciendo bien?
fig = px.line(this_client_ds, x="date", y="COMPRA ONLINE", 
              title='COMPRA ONLINE cliente {}'.format(client_ID))
fig.show()

#%%[markdown]
#### Podríamos intentar crear categorías de movimientos que engloben a varias de las ya existentes

#%%[markdown]
#### Una vez realizados estos exploratorios y checks varios sobre la naturaleza de nuestros datos, intentamos agrupar a los clientes por tipos

#####################
#%%[markdown]
#### ACF/PACF por variable de interés
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

# create lagged dataset
values = this_client_ds['TRANSFERENCIAS DE SALIDA']
values = pd.DataFrame(values)
autocorrelation_plot(values)
pyplot.show()
#####################


#%%[markdown]
# * Modelo baseline naive: de persistencia
naive_m_dataframe = series_to_supervised_df[['transf_sal_past_1', 'transf_sal_past_0']]
naive_m_dataframe

#%%
# split into train and test sets
X = naive_m_dataframe.values
train_size = int(len(X) * 0.7)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# persistence model
predictions = [x for x in test_X]
predictions
#%%
# calculate residuals
residuals = [test_y[i]-predictions[i] for i in range(len(predictions))]
residuals = pd.DataFrame(residuals)
autocorrelation_plot(residuals)
pyplot.show()
print(residuals)


#%%[markdown]
# * COMO SIGUIENTES MODELOS PODRÍAMOS PROBAR ARIMA MULTIVARIABLE, 
# * MODELOS DE REGRESIÓN COMO Support-Vector-Regressor, Decission-tree-regressor... con el dataset transformado a formato supervisado
# * PROBAMOS LSTM MULTIVARIATE:

desired_categories = ['TRANSFERENCIAS DE SALIDA', 'COMPRA ONLINE', 'GAS Y ELECTRICIDAD'] 
fig = make_subplots(rows=4, cols=1, subplot_titles=(['TRANSFERENCIAS DE ENTRADA', 'TRANSFERENCIAS DE SALIDA', 
                                                    'COMPRA ONLINE', 'SUPERMERCADOS Y ALIMENTACIÓN']))

fig.add_trace(
    go.Scatter(x=this_client_ds.date, y=this_client_ds['TRANSFERENCIAS DE ENTRADA'], 
            mode='lines+markers+text'), row=1, col=1
)
fig.add_trace(
    go.Scatter(x=this_client_ds.date, y=this_client_ds['TRANSFERENCIAS DE SALIDA'], 
            mode='lines+markers+text'), row=2, col=1
)
fig.add_trace(
    go.Scatter(x=this_client_ds.date, y=this_client_ds['COMPRA ONLINE'], 
            mode='lines+markers+text'), row=3, col=1
)
fig.add_trace(
    go.Scatter(x=this_client_ds.date, y=this_client_ds['SUPERMERCADOS Y ALIMENTACIÓN'], 
            mode='lines+markers+text'), row=4, col=1
)
fig.update_layout(margin={"r":10,"t":60,"l":10,"b":10}, height=600, width=710, showlegend=False, paper_bgcolor="#EBF2EC") 
fig.show()

#%%
client_id = selected_client_IDs[30]
this_client_history_set = processor_obj.build_client_dataset(client_id)
#%%
this_client_history_set = processor_obj.impute_missing_value(this_client_history_set, 0)
len(this_client_history_set)
#%%
this_client_history_set.tail()

#%%[markdown]
# * checkeamos el total de movimientos en los últimos meses:
cumulative_spents = pd.DataFrame({})
#for date in client_38361820_history_set[date_mask].index:
for date in this_client_history_set.index:
    cols = list(this_client_history_set.columns)
    no_category_cols = ['year', 'month', 'associatedAccountId']
    for no_cat in no_category_cols:
        cols.remove(no_cat)
    
    sum_spents_this_month = this_client_history_set.loc[date][cols].sum()
    cumulative_spents = cumulative_spents.append({'date': date, 
            'total_spents': sum_spents_this_month}, ignore_index=True)
#%%
import plotly.express as px 

fig = px.line(cumulative_spents, x="date", y="total_spents", 
                title='tendencia de gasto úlimos meses'.format(category, client_ID))
fig.show()

#%%[markdown]
# * UNA TENDENCIA DESCENDENTE DE LA SUMA DE MOVIMIENTOS PODRÍA DAR INFORMACIÓN SOBRE POSIBLE FUGA DE CLIENTES
# * ESTE CLIENTE PARECE BASTANTE ESTABLE CON LA ENTIDAD BANCARIA

#%%[markdown]
client_38361820_desired_hist = client_38361820_desired_hist[cols+['month']]
client_38361820_desired_hist
#%%[markdown]
# * esto se está haciendo ahora para un cliente; luego se podrá hacer para varios clientes de un segmento
dataset_vals = client_38361820_desired_hist.values
train_fraction = int(0.8*len(client_38361820_desired_hist)) 

#dataset_mean = dataset_vals[:train_fraction].mean(axis=0)
#dataset_std = dataset_vals[:train_fraction].std(axis=0)
#dataset = (dataset_vals-dataset_mean)/dataset_std

scaled_client_38361820_desired_hist = client_38361820_desired_hist
for attr in client_38361820_desired_hist.columns:
    attr_mean = client_38361820_desired_hist[:train_fraction].mean()
    print('attr_mean: {}'.format(attr_mean))
    attr_std = client_38361820_desired_hist[:train_fraction].std()
    print('attr_std: {}'.format(attr_std))
    scaled_client_38361820_desired_hist[attr] = (scaled_client_38361820_desired_hist[attr]-attr_mean)/attr_std


#%%
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

past_history = 6 #6 meses pasados
future_target = 1 #a un mes vista
STEP = 1

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                   train_fraction, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                               train_fraction, None, past_history,
                                               future_target, STEP,
                                               single_step=True)
#%%
import tensorflow as tf

BUFFER_SIZE = 10000
BATCH_SIZE = 256

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

#%%
single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32,input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

#%%
for x, y in val_data_single.take(1):
  print(single_step_model.predict(x).shape)

#%%
import time

mis_preds = []
true_values = []
init_time = time.time()
for x, y in val_data_single.take(300):
    true_values.append(y[0].numpy())
    mis_preds.append(single_step_model.predict(x)[0][0])

single_step_model_mae_b4_fit = tf.keras.metrics.mae(true_values, mis_preds).numpy()
print('single_step_model_mae_b4_fit: {}'.format(single_step_model_mae_b4_fit))





#%%[markdown]
#### ME INTERESA CUANTIFICAR LA POSIBLE PREDICTIBILIDAD DE LOS GASTOS POR SEPARADO, PODRÍA:
# * REVISAR CORRELACIONES CON LAG VALUES DE OTROS GASTOS (ESTO ES, LA CORR. PLOT PERO CON MÁS ATRIBUTOS)
# * SEGMENTAR CLIENTES EN ABSE A SUS GASTOS MEDIOS POR CATEGORÍA Y VER ESTO



#%%
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t', 't+1']
# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# persistence model
predictions = [x for x in test_X]
# calculate residuals
residuals = [test_y[i]-predictions[i] for i in range(len(predictions))]
residuals = pd.DataFrame(residuals)
autocorrelation_plot(residuals)
pyplot.show()

#%%[markdown]
#### Podría haber 


#%%[markdown]
#### WHITE NOISE CHECK en la variable categoría de gasto de interés

#%%[markdown]
#### BASELINE MODEL: se trata de un modelo naive que predice basado en el valor del mes anterior
# persistence model
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

predictions = [x for x in test_X]

#%%[markdown]
#### ARIMA univariable: re-escalar los datos +  aplicar ARIMA multivariable

#%%[markdown]
#### LSTM MULTIVARIATE: mencionar la importancia de añadir confidence/prediction intervals 











#%%
#################
category = 'AUTOPISTAS GASOLINERAS Y PARKINGS' 
#represento la serie temporal de un cliente para un producto:
category_mask = explorer_obj.dataset_['categoryDescription']==category
dataset_category = explorer_obj.dataset_[category_mask]
assert len(dataset_category)>0
print(len(dataset_category))

# %% [markdown]
# #### como era de esperar, más clientes tienen este tipo de movimiento que nómina domiciliada (makes sense, business testing)

# %%
import plotly.express as px 

client_ID=dataset_category.iloc[10]['associatedAccountId']
client_mask = dataset_category['associatedAccountId']==client_ID
sorted_client_category = dataset_category[client_mask].sort_values(by='date', ascending=True)
fig = px.line(sorted_client_category, x="date", y="monthlySpent", title='{} cliente {}'.format(category, client_ID))
fig.show()


# %%
fig = px.scatter(sorted_client_category, x="date", y="monthlySpent", title='{} cliente {}'.format(category, client_ID))
fig.show()

# %% [markdown]
# #### esto parece indicar estacionalidad correspondiente a vacaciones (en torno a julio-agosto de cada año) --> probar DT-regressor, ARIMA, CNN, LSTM, etc
# %% [markdown]
# #### en estas regresiones temporales multivariables podríamos añadir como atributos la ocurrencia o no de movimientos anómalos de otras categorías relacionadas

# %%
category = 'VEHíCULOS Y REPARACIONES'
#represento la serie temporal de un cliente para un producto:
category_mask = explorer_obj.dataset_['categoryDescription']==category
dataset_category = explorer_obj.dataset_[category_mask]
assert len(dataset_category)>0
print(len(dataset_category))


# %%
client_ID=dataset_category.iloc[10]['associatedAccountId']
client_mask = dataset_category['associatedAccountId']==client_ID
sorted_client_category = dataset_category[client_mask].sort_values(by='date', ascending=True)
fig = px.scatter(sorted_client_category, x="date", y="monthlySpent", title='{} cliente {}'.format(category, client_ID))
fig.show()

# %% [markdown]
# #### CATEGORIZAR LAS SUBCATEGORIAS EN CATEGORÍAS SUPERIORES; ASÍ AUMENTAMOS ADEMÁS LOS DATOS DE MOVIMIENTOS PARA CATEGORÍAS COMO LA ANTERIOR QUE TIENEN POCOS MOVIMIENTOS
# %% [markdown]
# #### FUTURIBLE: INTENTAR PENSAR TB EN APLICAR NLP A LOS TEXTOS DEL CAMPO CONCEPTO. PODRÍA INFORMAR DE POSIBLES GASTOS ASOCIADOS DE OTRO TIPO? POR EJEMPLO SI SE VE UN MOVIMIENTO QUE INDIQUE "GUARDERÍA" QUIZÁS SE LE PUEDE RECOMENDAR PRODUCTOS RELACIONADOS CON BEBÉS. Esto más bien sería para categorizar los movimientos no clasificados con categoría alguna
# %% [markdown]
# #### HAY CATEGORÍAS QUE PODRÍAN PERTENECER A MÁS DE UNA SUPERCATEGORÍA; POR EJEMPLO, GASTOS DE VIAJES PUEDE SER TANTO PARA TEMAS DE OCIO COMO TEMAS DE TRABAJO, PENSAR EN ESTO Y VALIDAR SI ES POSIBLE CON CORRELACIONES EN EL EDA  

# %%
explorer_obj.dataset_[['associatedAccountId', 'categoryId', 'year', 'month', 'monthlySpent']]


# %%
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data)


# %%
import seaborn as sns #; sns.set()

ax = sns.heatmap(explorer_obj.dataset_[['associatedAccountId', 'categoryId', 'year', 'month', 'monthlySpent']].iloc[:1000].values)

# %% [markdown]
# #### PROBAR A REALIZAR LA MISMA PRUEBA SOBRE UN DATASET SENCILLO A MODO UNIT TEST
# %% [markdown]
# ### PERFILAR ADEMÁS DE TIPOS DE GASTO, LOS TIPOS DE CLIENTE SEGÚN: 
# * ESTACIONALIDAD EN SU CONSUMO
# * NÚMERO DE MOVIMIENTOS REGISTRADOS EN CIERTO TIPO DE CONSUMO
# * REGULARIDAD EN LA FORMA DE CONSUMIR DICHA CATEGORÍA
# * ... 
# %% [markdown]
# # MODELO QUE SE ME OCURRE DE FORECASTING: MULTIVARIATE DATASET POR CDA CLIENTE (O TIPO DE CLIENTES SI HAGO SEGMENTACIÓN):
# * Por cada cliente, tengo para cada producto un valor de consumo en cada registro temporal, por lo que la entrada sería multivariable con objetivo de predecir el consumo en el producto requerido
# * Puedo intentar que las entrada multivariable no sea con cada tipo, sino con super tipos de productos
# * En caso de hacer segmentación de clientes, en lugar de un modelo por cliente, podría ser un modelo por tipo de cliente, donde este perfilado de cliente sería en base a: 
# - cómo de constante es en sus consumos: su patrón de consumo definido con la std de de las std de franjas temporalaes (divide el histórico en cuatrimestres por ej) 
# - qué estacionalidad tiene en ciertas categorías: por ej., en combustible tiene semestral coincidiendo con vacas? 
# 

# %%


