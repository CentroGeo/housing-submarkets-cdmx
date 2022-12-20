#############################################################################
#####  KMEANS, TSNE Y AZP

#############################################################################

#### Imports

import libpysal
from h3 import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from spopt.region import AZP
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import PowerTransformer

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import ward, fcluster
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

def id2poly(hex_id):
    boundary = [x[::-1] for x in h3.h3_to_geo_boundary(hex_id)]
    return Polygon(boundary)

#############################################################################
##### Input - output paths

Entradas = ''
Salidas = ''

#############################################################################
#####

#### Input dataset must have all acessibility and geographic variables 
####

DATA_DEPART = pd.read_csv(Entradas + 'DEPART_DIM.csv')

DATA_CASAS = pd.read_csv(Entradas + 'CASA_DIM.csv')

#### Example for apartments dataset

DATA_DPTO = DATA_DEPART.groupby('hex').agg({'superficie':'mean',
                                            'habitacion':'mean',
                                            'baños':'mean',
                                            'Parques_SA':'mean',
                                            'Angel_SA':'mean',
                                            'Teatros_SA':'mean',
                                            'SuperM_SA':'mean',
                                            'Restau_SA':'mean',
                                            'Oficina_SA':'mean',
                                            'MetroBus_SA':'mean',
                                            'Metro_SA':'mean',
                                            'Hospital_SA':'mean',
                                            'GyMFree_SA':'mean',
                                            'GyM_SA':'mean',
                                            'Cinema_SA':'mean',
                                            'EscSecu_SA':'mean',
                                            'EscPr_SA':'mean',
                                            'EscMS_SA':'mean',
                                            'EcoBici_SA':'mean',
                                            'Delitos':'mean',
                                            'Socioeco':'mean'}).reset_index()

LABEL = DATA_DPTO['hex']
DIMENSIONS = DATA_DPTO.drop(['hex'], axis=1)


S_S = PowerTransformer()
Var_Scales = pd.DataFrame(S_S.fit_transform(DIMENSIONS),
                          columns = DIMENSIONS.columns)


#############################################################################
##### KMEANS CLUSTER


def K_model(Dimensiones_tran, numero_min, numero_max, Title):
    ### Input:
    ### -> Variables explicativas transformadas
    ### -> Número mínimo de cluster
    ### -> Número máximo de cluster
    model = KMeans()
    visualizer = KElbowVisualizer( model,
                                   k = (numero_min, numero_max),
                                   timings= True)
    visualizer.fit(Dimensiones_tran)
    visualizer.show(outpath = Title, dpi = 300)


K_model(Var_Scales, 2, 10, 'CURVA DATA')


def K_means(Dimensiones_tran, numero_cluster, semilla):
    ### Input:
    ### -> Variables explicativas transformadas
    ### -> Número de Cluster
    ### -> Número de semilla en random-state
    kmeans = KMeans(n_clusters = numero_cluster, random_state = semilla)
    kmeans.fit(Dimensiones_tran)
    clusters = kmeans.predict(Dimensiones_tran)
    ### Output:
    ### -> Valores de cluster
    return (clusters)

Clusters_K = K_means(Var_Scales, 5, 42)

#############################################################################
#####  Ward CLUSTER

def T_NSE(Dimensiones_tran, Numero_compo, Semilla, Perplexity):
    ### Input:
    ### -> Variables explicativas transformadas
    ### -> Número de componentes debe ser inferior a 4 (algoritmo barnes_hut, basado en quad-tree y oct-tree)
    ### -> Número de semilla en random-state
    ### -> Perplexity o valores de perplejidad en el rango (5 - 50) sugeridos por van der Maaten & Hinton
    tsn = TSNE(n_components = Numero_compo, random_state = Semilla, perplexity = Perplexity)
    res_tsne = tsn.fit_transform(Dimensiones_tran)
    fig = plt.figure(figsize=(25,25))
    ax1 = fig.add_subplot(3,3,1)
    axpl_1 = sns.scatterplot(res_tsne[:,0],res_tsne[:,1]);
    ax2 = fig.add_subplot(3,3,2)
    axpl_2 = sns.scatterplot(res_tsne[:,0],res_tsne[:,2]);
    ### Output:
    ### -> Valores TNSE de las componentes reducidas
    return (res_tsne)

TNSE_Model = T_NSE(Var_Scales, 3, 42, 50)

### Cluster t-SNE :WARD

def Cluster_TNSE(Modelo_tSNE, Cluster_max):
    ### Input:
    ### -> Array correspondiente a cada componente TNSE resultado de reducción dimensional
    ### -> Número de cluster designados bajo el criterio maxclust de Fcluster
    link = ward(Modelo_tSNE)
    Cluster_Values = fcluster(link, t = Cluster_max, criterion = 'maxclust')
    fig = plt.figure(figsize = (25,25))
    ax1 = fig.add_subplot(3,3,1)
    pd.value_counts(Cluster_Values).plot(kind='barh')
    ax2 = fig.add_subplot(3,3,2)
    axpl_2 = sns.scatterplot(x = Modelo_tSNE[:,0], y = Modelo_tSNE[:,1],hue = Cluster_Values, palette="Set2");
    ax2 = fig.add_subplot(3,3,3)
    axpl_2 = sns.scatterplot(x = Modelo_tSNE[:,0], y = Modelo_tSNE[:,2], hue = Cluster_Values, palette="Set2");
    ### Output:
    ### -> Valores de cluster TNSE
    return (Cluster_Values)


TNSE_Cluster = Cluster_TNSE(TNSE_Model, 5)


### CLUSTER (KMEANS + WARD) EN LA TABLA DE DIMENSIONES

DATA_DPTO["K_Means"] = Clusters_K
DATA_DPTO["t_SNE"] = TNSE_Cluster


#############################################################################
#####  AZP CLUSTER

Var_Scales['geometry'] = Var_Scales['hex'].apply(id2poly)
Var_Scales = gpd.GeoDataFrame( Var_Scales,
                               geometry = Var_Scales['geometry'])
Var_Scales.crs = "EPSG:4326"
Var_Scales = Var_Scales.to_crs("EPSG:4326").to_crs(6372).set_index('hex')


COLUMN_DIMEN = ['superficie', 'habitacion', 'baños', 'Parques_SA',
                 'Angel_SA', 'Teatros_SA', 'SuperM_SA', 'Restau_SA', 'Oficina_SA',
                 'MetroBus_SA', 'Metro_SA', 'Hospital_SA', 'GyMFree_SA', 'GyM_SA',
                 'Cinema_SA', 'EscSecu_SA', 'EscPr_SA', 'EscMS_SA',
                 'EcoBici_SA', 'Delitos', 'Socioeco']

#############################################################################
##### AZP CLUSTER

KKN_MATRIX = libpysal.weights.KNN.from_dataframe(Var_Scales, k = 8)

model = AZP(Var_Scales, KKN_MATRIX, COLUMN_DIMEN, 5)
model.solve()

Var_Scales['AZP'] = model.labels_

DATA_ALL = DATA_DPTO.merge( Var_Scales[['hex', 'AZP']],
                            left_on = 'hex',
                            right_on = 'hex',
                            how ='left')

DPTO_CLUSTER_ALL = DATA_DEPART.merge( DATA_ALL[['hex',
                                                'AZP',
                                                'K_Means',
                                                't_SNE']],
                                                left_on = 'hex',
                                                right_on = 'hex',
                                                how = 'left')


#############################################################################
##### Write as csv

DPTO_CLUSTER_ALL.to_csv(Salidas + "DPTO_CLUSTER_ALL.csv", index = False)
