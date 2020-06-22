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


    def plot_timeseries_values(self, attribute_to_plot, category_type='all', mask_attribute_name=None, mask_attribute_value=None, 
                                date_column_name=None, graph_title=None):
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
            return fig

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
            from scipy.stats import shapiro

            gaussian_attributes = []
            non_gaussian_attributes = []
            for col in selected_attributes:
                # normality test
                stat, p = shapiro(dataset[col].values)
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
# * se intuye que habrá unas categorías cuya distribución temporal sea mucho más constante que otras, por ejemplo: se espera que la categoría 'MENAJE DEL HOGAR Y ELECTRÓNICA' no presente un claro patrón de consumo como sí lo deberíamos ver en 'GAS Y ELECTRICIDAD' 

#%%[markdown]
# * Data_quality_check --> tenemos el mismo número de IDs de categorías que de descripciones?

# %%
assert len(explorer_obj.dataset_['categoryDescription'].unique())==len(explorer_obj.dataset_['categoryId'].unique()), \
    print('número de valores distintos de categoryDescription : {} y número de valores distintos de categoryId : {}'.format(len(explorer_obj.dataset_['categoryDescription'].unique()), len(explorer_obj.dataset_['categoryId'].unique()))) 

#%%[markdown]
#### Comprobamos qué categoryId nos sobra; para ello, vemos las combinaciones de description e ID:

# %%
category_desc_ID_combinations = explorer_obj.dataset_[['categoryId', 'categoryDescription']].drop_duplicates()
category_desc_ID_combinations.categoryDescription.value_counts()

#%%[markdown]
# * Vemos que la descripción duplicada corresponde a 'SIN CLASIFICAR', por lo que no se trata de una duplicidad a corregir
# * Podría ser interesante mirar la cuantía de movimientos no clasificados, en caso de que sea alta y de posible interés a identificar 

#%%[markdown]
# * añadimos este dataset de combinaciones a nuestro objeto exploratorio:
explorer_obj.category_desc_ID_combinations = category_desc_ID_combinations

if explorer_obj.category_desc_ID_combinations.to_dict()==category_desc_ID_combinations.to_dict():
    category_desc_ID_combinations=None

#%%[markdown]
#### Representamos los valores de alguna serie temporal

#%%[markdown]
# * Añadimos el atributo 'date' de forma que nos facilite la representación gráfica en formato de serie temporal:
from datetime import datetime

assert datetime(2017, 1, 1, 0, 0)==explorer_obj.generate_date(2017, 1, 1) 

#%%
import pandas as pd 

explorer_obj.add_date_attribute('year', 'month', day_column_name=28)
explorer_obj.dataset_.tail(5)

#%%
nominas_mask = explorer_obj.dataset_['categoryDescription']=='NÓMINAS'
dataset_nominas = explorer_obj.dataset_[nominas_mask]
assert len(dataset_nominas)>0
print('número de registros con tipo de movimiento ''NOMINA'': {}'.format(len(dataset_nominas)))

# %% [markdown]
#### Contamos el porcentaje de clientes que tienen algún movimiento para cada categoría: 
# %%[markdown]
#### este código iría en una función de la clase de exploratorio
categories = explorer_obj.category_desc_ID_combinations.categoryDescription.values
clients_number = len(explorer_obj.dataset_.associatedAccountId.unique())
category_freqs_df = pd.DataFrame(columns=['category', 'usage_ratio'])
for category in categories:
    category_mask = explorer_obj.dataset_['categoryDescription']==category
    dataset_with_category_type = explorer_obj.dataset_[category_mask]
    category_freqs_df = category_freqs_df.append({'category': category, 
            'usage_ratio': len(dataset_with_category_type)/clients_number}, ignore_index=True)

#%%
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

# %%[markdown]
#### este código iría en una función de la clase de exploratorio
import plotly.express as px

client_ID=explorer_obj.dataset_['associatedAccountId'].unique()[77]
client_mask = explorer_obj.dataset_['associatedAccountId']==client_ID
client_df = explorer_obj.dataset_[client_mask]
client_nominas_mask = client_df['categoryDescription']=='NÓMINAS'
client_nominas_df = client_df[client_nominas_mask]
sorted_client_nominas = client_nominas_df.sort_values(by='date', ascending=True)
fig = px.line(sorted_client_nominas, x="date", y="monthlySpent", title='Nóminas de cliente {}'.format(client_ID))
fig.show()

#%%
client_ID=explorer_obj.dataset_['associatedAccountId'].unique()[40]
client_mask = explorer_obj.dataset_['associatedAccountId']==client_ID
client_df = explorer_obj.dataset_[client_mask]
client_nominas_mask = client_df['categoryDescription']=='NÓMINAS'
client_nominas_df = client_df[client_nominas_mask]
sorted_client_nominas = client_nominas_df.sort_values(by='date', ascending=True)
fig = px.line(sorted_client_nominas, x="date", y="monthlySpent", title='Nóminas de cliente {}'.format(client_ID))

fig.show()

client_ID=explorer_obj.dataset_['associatedAccountId'].unique()[88]
client_mask = explorer_obj.dataset_['associatedAccountId']==client_ID
client_df = explorer_obj.dataset_[client_mask]
client_nominas_mask = client_df['categoryDescription']=='NÓMINAS'
client_nominas_df = client_df[client_nominas_mask]
sorted_client_nominas = client_nominas_df.sort_values(by='date', ascending=True)
fig = px.line(sorted_client_nominas, x="date", y="monthlySpent", title='Nóminas de cliente {}'.format(client_ID))

fig.show()

#%%
# liberamos memoria:
client_nominas_df=client_df=client_nominas_mask=client_mask=None

# %% [markdown]
'''
Esos picos parecen indicar pagas extras. En los casos donde vemos esos picos pero no coinciden
con los meses esperados (junio y diciembre) y/o no son semestrales, podríamos hacer uso de análisis 
de texto del campo 'concepto' (si lo hubiera) correspondiente a tal movimiento
'''

# %%
client_ID=38361820 
client_mask = dataset_nominas['associatedAccountId']==client_ID
sorted_client_nominas = dataset_nominas[client_mask].sort_values(by='date', ascending=True)
fig = px.line(sorted_client_nominas, x="date", y="monthlySpent", title='Nóminas cliente {}'.format(client_ID))
fig.show()

#%% [markdown]
#### podría ser interesante generar atributo con elñ score de anomalía univariable por cada registro de cada cliente en cada producto?)

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
Se aprecian máximos locales en torno a los meses de julio y diciembre para lo que parecen ser asalariados. En cambio para otros cleintes 
no existe ese patrón; esto nos ayudaría a perfilar clientes por tipo de sueldo (asalariados, autónomos, etc)
'''
#%%
categoria_transf_entrada = ['TRANSFERENCIAS DE ENTRADA']
client_id = 38361820 
client_mask = explorer_obj.dataset_['associatedAccountId']==client_id
this_client_ds = explorer_obj.dataset_[client_mask]
gastos_mask = [gasto in categoria_transf_entrada for gasto in this_client_ds['categoryDescription'].values]
sorted_client_movs = this_client_ds[gastos_mask].sort_values(by='date', ascending=True)
fig = px.line(sorted_client_movs, x="date", y="monthlySpent", title='Transferencias de entrada {}'.format(client_id))
fig.show()

#%%
'''
Vemos que el cliente 38361820 tiene movimientos de 'TRANSFERENCIAS DE ENTRADA' (en principio de bajo importe) 
hasta junio de 2019 mientras que los de nómina acabaron en 2018. Un atributo extra podría construirse modelando la tendencia, si la hubiera, 
de ciertas categorías de interés con una ventana móvil de x meses; esto podría dar información para predicción de fuga de clientes?
'''

#%%
'''
from plotly.subplots import make_subplots
import plotly.graph_objects as go

sorted_client_movs = this_client_ds.sort_values(by='date', ascending=True)

fig = make_subplots(rows=2, cols=2, subplot_titles=("TRANSFERENCIAS DE ENTRADA", "GAS Y ELECTRICIDAD", "MODA", "RESTAURACIÓN"))

fig.add_trace(
    go.Scatter(x=sorted_client_movs.date, y=sorted_client_movs["TRANSFERENCIAS DE ENTRADA"], 
            mode='lines+markers+text'), row=1, col=1
)
fig.add_trace(
    go.Scatter(x=sorted_client_movs.date, y=sorted_client_movs["GAS Y ELECTRICIDAD"], 
            mode='lines+markers+text'), row=1, col=2
)

fig.add_trace(
    go.Scatter(x=sorted_client_movs.date, y=sorted_client_movs["MODA"], 
            mode='lines+markers+text'), row=2, col=1
)

fig.add_trace(
    go.Scatter(x=sorted_client_movs.date, y=sorted_client_movs["RESTAURACIÓN"], 
            mode='lines+markers+text'), row=2, col=2
)

fig.update_layout(margin={"r":10,"t":60,"l":10,"b":10}, height=600, width=710, showlegend=False, paper_bgcolor="#EBF2EC") 
fig.show()
'''

#%%[markdown]
#### Construyo el dataset basado en los lag values de cada tipo de gasto, así:
# * metemos 0 en los meses donde un cliente no tiene movimiento en un tipo de gasto
# * pasamos el check de missing values
# * formamos el dataset traspuesto generando un atributo por cada tipo de movimiento
# * para encontrar posibles correlaciones, aplicamos el check de normal distribution para coger o no la pearson corr coeff u otra no paramétrica
# * para la variable de interés, es white noise? Aunque esto aplica a la serie de la categoría y no en principio a su relación con el resto
# * para cada variable temporal, son estacionarias?
# * se puede checkear la distribución de residuos tras hacer el forecast para comprobar si es ruido blanco y podría haber margen de mejora en el modelo
# * para abordar el problema p >> n, miraremos: selección de atributos (en ppio por correlaciones) y/o PCA (aunque esto podría restar interpretabilidad a posteriori)
# * dependiendo de la antelación con la que necesitemos predecir posibles gastos de movimientos, necesitaríamos crear un "single-step forecast" (a un mes), o etiquetar con x meses a futuro, o hacer un multi-step forecast 
# * podríamos probar a agrupar categorías por super tipos de categorías para reducir la relación numero_filas-número_columnas del dataset

# %% [markdown]
#### Empezaría por una segmentación basada en un criterio sencillo y útil; posteriormente haría el multivariable basado por ejemplo en k-means (con métrica euclídea y sin el problema de tener variables discontínuas)
#### Pensar en categorizar tipos de productos por relevancia: podría ser por cuantía mediana de movimientos de esa categoría, o por número agregado entre todos los clientes
#### De cara a modelar, nuestros modelos podrían validarse en base a que los intervalos de confianza de las variables predichas no excedieran el techo medio (mediano) de gasto o movimientos (si lo hubiera)

#%%[markdown]
# * Queremos ver la distribución de número de meses durante los que los clientes tienen movimientos con el banco
# * Esto es relevante de cara a conocer con qué históricos contamos por cliente

#%%[markdown]
### Posibles enfoques a la hora de modelar el predictor:
#### Baseline model basado simplemente en persistir el valor anterior de la serie
#### Enfoque de histórico individual por cliente --> handicap: tendríamos tantos modelos como clientes (753) además de no contar con suficiente histórico para muchos ed ellos
#### Enfoque de histórico por segmento de cliente --> esto resolvería en parte (dependiendo de cuántos clusters tuviéramos) el problema del número de modelos = número de clientes;
#### en cambio para el anterior enfoque tendríamos para cada valor 'date' varios valores, por lo que: 
# * se podrían obtener los valores medianos de cada categoría en cada valor de tiempo, o
# * se podría convertir nuestro dataset en formato "supervisado" donde no sería necesario el orden temporal, convirtiéndolo en un problema de regresión multivariable   

# %%[markdown]
'''
#### este código iría en una función de la clase de exploratorio
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
'''
#%%[markdown]
# * formamos el dataset transponiendo los valroes de las categorías como atributos: 
all_clients_history_set=pd.DataFrame()
clients_with_errors=list()
history_length_per_client_dict = {}
for client_id in explorer_obj.dataset_['associatedAccountId'].unique(): 
    try:
        processor_obj = ProcessDataset()
        processor_obj.dataset_ = explorer_obj.dataset_
        this_client_history_set = processor_obj.build_client_dataset(client_id)
        this_client_history_set = processor_obj.impute_missing_value(this_client_history_set, 0)
        all_clients_history_set = all_clients_history_set.append(this_client_history_set)
        history_length_per_client_dict[client_id] = len(all_clients_history_set)
    except Exception as exc:
        #este error iría a un logger de errores
        print('client {} gave an error: {}'.format(client_id, exc))
        pass

history_length_per_client_dict.tail(10)

#%%[markdown]
# * cogería sólo los datasets de los clientes con más registros mensuales para busar mayor representatividad (info en history_length_per_client_dict)
import seaborn as sns, numpy as np
sns.set()

values = [value for key, value in history_length_per_client_dict.items()]
ax = sns.distplot(values)

#%%
selected_client_IDs = [key for key, value in history_length_per_client_dict.items() if value > 10000]

#%%[markdown]
# * formamos el dataset con el histórico de los clientes con un histórico aceptado como suficiente
selected_clients_history_set=pd.DataFrame()
for client_ID in selected_client_IDs:
    client_mask = all_clients_history_set['associatedAccountId']==client_ID
    client_ID_df = all_clients_history_set[client_mask]
    selected_clients_history_set = selected_clients_history_set.append(client_ID_df)

#%%[markdown]
#### Obtenemos también los valores medianos de cada atributo por cada cliente, para estudiar luego posibles correlaciones:
selected_clients_columns = list(selected_clients_history_set.columns)
selected_clients_columns.remove('associatedAccountId')

selected_clients_history_set = processor_obj.impute_missing_value(selected_clients_history_set, 0)
selected_clients_history_set['date']=selected_clients_history_set.index
selected_clients_median_values = selected_clients_history_set.groupby(by=['date', 'associatedAccountId'])[selected_clients_columns].median()
#%%
selected_clients_history_set=selected_clients_history_set.drop(['date'], axis=1)

#%%[markdown]
#### Comprobamos si los atributos considerados siguen una distribución normal
### Añadimos otro test de normalidad:
explore_obj = ExploreDataset()
gaussian_attributes, non_gaussian_attributes = explore_obj.check_if_gaussian(selected_clients_median_values, selected_clients_columns)


#%%[markdown]
#### Probamos con el histórico de un cliente para abordar un forecast model:
client_id = selected_clients_history_set['associatedAccountId'][25]
this_client_mask = selected_clients_history_set['associatedAccountId'] == client_id 
this_client_ds = selected_clients_history_set[this_client_mask]
this_client_mask=None
this_client_ds

#%%[markdown]
# * escogemos atributos que no tiene de valor medio = 0
columns_to_select = list(this_client_ds.columns)
non_zero_selected_attributes_mask = this_client_ds[columns_to_select].groupby(by=['associatedAccountId']).mean() > 0
non_zero_attrs = list(non_zero_selected_attributes_mask.columns[non_zero_selected_attributes_mask.values[0]])
non_zero_attrs.remove('month')

#%%
gaussian_attributes, non_gaussian_attributes = explore_obj.check_if_gaussian(this_client_ds,
                                                    non_zero_attrs)
print('gaussian_attributes: {}'.format(gaussian_attributes))
print('non_gaussian_attributes: {}'.format(non_gaussian_attributes))

#%%[markdown]
#### Parece que las distribuciones de los atributos no son gaussianas, por lo que emplearemos el método Kendall para distribuciones no paramétricas
# * esto iría en una función de la clase de exploratorio
import matplotlib.pyplot as plt

dataset = this_client_ds[non_zero_attrs]
# Compute the correlation matrix
corr = dataset.corr(method='kendall')
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
# * y si probamos a generar los atributos que son valores de 1 y 2 meses anteriores?
# * esto que sigue iría en la clase preprocesadora del dataset
data_sup = processor_obj.series_to_supervised(this_client_ds[['TRANSFERENCIAS DE SALIDA']], 2, n_out=1)
series_to_supervised_df = pd.DataFrame(columns=['transf_sal_past_2', 'transf_sal_past_1', 
                                                'transf_sal_past_0'], data=data_sup)
series_to_supervised_df

#%%[markdown]
# * lo anterior se podría aplicar a cada uno de los atributos (categorías de gasto) consideradas, creando n_attributos*n_lags atributos finales
# * este dataset resultante, etiquetando como variable objetivo los valores del último 'date_time' de la categoría a predecir, se podría utilizar con algoritmos de machine learning sin necesidad de orden temporal 

#%%
'''
data_sup=series_to_supervised(this_client_ds[['TRANSFERENCIAS DE SALIDA', 'COMPRA ONLINE']], 2, n_out=1)
data_sup
#%%
series_to_supervised_df = pd.DataFrame(columns=['transf_sal_past_2', 'online_shop_past_2',
                                                'transf_sal_past_1', 'online_shop_past_1',
                                                'transf_sal_past_0', 'online_shop_past_0'],
                                                data=data_sup)
'''

#%%[markdown]
##### Y ahora tenemos mayor correlación del atributo 'TRANSFERENCIAS DE SALIDA' con valores pasados del mismo?
dataset = series_to_supervised_df
# Compute the correlation matrix
corr = dataset.corr(method='kendall')
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
#### otra forma de comprobar si la propia variable podría ser explicada por sí misma con valores anteriores es mediante la función de auto-correlación: 
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

# create lagged dataset
values = this_client_ds['TRANSFERENCIAS DE SALIDA']
values = pd.DataFrame(values)
autocorrelation_plot(values)
pyplot.show()

#%%[markdown]
# * no parece haber algunos lags estadísticamente significativos en esta variable
# * Podríamos intentar crear categorías de movimientos que engloben a varias de las ya existentes
# * Una vez realizados estos exploratorios y checks varios sobre la naturaleza de nuestros datos, intentaríamos agrupar a los clientes por tipos


#%%[markdown]
#### Ahora intentamos abordar el modelado del predictor:
# * Comenzamos con un modelo baseline naive: de persistencia
naive_m_dataframe = series_to_supervised_df[['transf_sal_past_1', 'transf_sal_past_0']]
naive_m_dataframe.tail(10)

#%%[markdown]
# * creamos train y test sets
# * creamos el modelo de persistencia del valor anterior 
# * para este modelo no tenemos en cuenta la escala de los atributos 
# * calculamos el mean-squared-error (el código asociado de esto iría en una clase dedicada)
X = naive_m_dataframe.values
train_size = int(len(X) * 0.7)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# persistence model
predictions = [x for x in test_X]

from sklearn.metrics import mean_squared_error

mse_value = mean_squared_error(test_y, predictions)
print('mse_value: {}'.format(mse_value))

#%%[markdown]
# * faltaría añadir intervalo de confianza asociado 

#%%[markdown]
# * COMO SIGUIENTES MODELOS PODRÍAMOS PROBAR ARIMA MULTIVARIABLE, 
# * MODELOS DE REGRESIÓN COMO Support-Vector-Regressor, Decission-tree-regressor... con el dataset transformado a formato supervisado
# * PROBAMOS LSTM MULTIVARIATE:

#%%[markdown]
# * ploteamos variables de interés del dataset escogido (de un cliente por sencillez)

from plotly.subplots import make_subplots
import plotly.graph_objects as go


this_client_ds['date'] = this_client_ds.index

desired_categories = ['TRANSFERENCIAS DE ENTRADA', 'TRANSFERENCIAS DE SALIDA', 
                                                    'COMPRA ONLINE', 'SUPERMERCADOS Y ALIMENTACIÓN', 
                                                    'AUTOPISTAS GASOLINERAS Y PARKINGS', 'NÓMINAS'] 
fig = make_subplots(rows=3, cols=2, subplot_titles=(desired_categories))

fig.add_trace(
    go.Scatter(x=this_client_ds.date, y=this_client_ds['TRANSFERENCIAS DE ENTRADA'], 
            mode='lines+markers+text'), row=1, col=1
)
fig.add_trace(
    go.Scatter(x=this_client_ds.date, y=this_client_ds['TRANSFERENCIAS DE SALIDA'], 
            mode='lines+markers+text'), row=1, col=2
)
fig.add_trace(
    go.Scatter(x=this_client_ds.date, y=this_client_ds['COMPRA ONLINE'], 
            mode='lines+markers+text'), row=2, col=1
)
fig.add_trace(
    go.Scatter(x=this_client_ds.date, y=this_client_ds['SUPERMERCADOS Y ALIMENTACIÓN'], 
            mode='lines+markers+text'), row=2, col=2
)
fig.add_trace(
    go.Scatter(x=this_client_ds.date, y=this_client_ds['AUTOPISTAS GASOLINERAS Y PARKINGS'], 
            mode='lines+markers+text'), row=3, col=1
)
fig.add_trace(
    go.Scatter(x=this_client_ds.date, y=this_client_ds['NÓMINAS'], 
            mode='lines+markers+text'), row=3, col=2
)

this_client_ds=this_client_ds.drop(['date'], axis=1)

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


