import random
import csv
import tensorflow as tf
from tensorflow import keras
#from tensorflow_model_optimization.python.core.keras.compat import keras
from keras import layers
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import regularizers
#import tensorflow_model_optimization as tfmot
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#come tolgo il commento non carica più il modello .-.
#se metto il commento lo carica , non capisco il problema 
from keras import backend  as K
#allenando bene il modello non cambio pià le immagini , i punti di focus so sempre gli stessi m ail modello non sbaglia , ansi , alcune volte migliora
import copy
import openpyxl
from openpyxl import Workbook
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
def back_position_maxpool_to_back(riga,colonna,ch):
    new_shape = (15, 15, 32)
    pool_size = (2, 2)

    # Calcolo del blocco nella matrice originale
    old_row_start = riga * pool_size[0]
    old_row_end = old_row_start + pool_size[0] - 1

    old_col_start = colonna * pool_size[1]
    old_col_end = old_col_start + pool_size[1] - 1

    old_channel = ch

    # Blocco di elementi originali che contribuiscono
    old_block = {
        "row_range": (old_row_start, old_row_end),
        "col_range": (old_col_start, old_col_end),
        "channel": old_channel
    }
    return old_block

def back_position_Conv_to_back(riga,colonna,ch):
  
    kernel_size = 3
    stride = 1
    padding = 0

    # Calcolo del blocco nella matrice originale
    old_row_start = riga * stride
    old_row_end = old_row_start + kernel_size - 1

    old_col_start = colonna * stride
    old_col_end = old_col_start + kernel_size - 1

    old_channel = ch

    # Blocco di elementi originali che contribuiscono
    old_block = {
        "row_range": (old_row_start, old_row_end),
        "col_range": (old_col_start, old_col_end),
        "channel": old_channel
    }

    print(f"Il blocco di elementi originali che contribuiscono: {old_block}")

    return old_block


def back_from_most_important(loaded_model,numeroscelto,x_test_normalized):
          


    input_test = x_test_normalized[numeroscelto]#l'unico che trova in maniera corretta 109
    newarr = input_test.reshape(1,32,32,3)  

    get_last_layer_output =K.function([loaded_model.layers[0].input],[loaded_model.layers[2].output])# layer2 sarebbe il secondo conv2d m, non mi serve il successivo maxpooling
    layer_output = get_last_layer_output(newarr)[0]

    _,dimx,dimy,dimn=layer_output.shape
    max = 0
    n = x = y = 0
    for n in range(dimx):                                     
        for x in range(dimy):
            for y in range(dimn):
                if (max < layer_output[0,n,x,y]):
                    max=layer_output[0,n,x,y]  
                    saveN= n  
                    saveX= x
                    savey= y

        #print("il valore più altro trovato 3(secondo conv2d)e' ",max,"in posizione ",saveN , saveX, savey)

    
        new_block = back_position_Conv_to_back(saveN,saveX,savey)
        row_start = new_block["row_range"][0]
        row_end = new_block["row_range"][1]+1

        col_start = new_block["col_range"][0]
        col_end = new_block["col_range"][1]+1
        new_block_maxpool=[]
        for i in range (row_start,row_end):
          for j in range(col_start,col_end):
                result=back_position_maxpool_to_back(i, j,savey)
                new_block_maxpool.append(result)

  #     print (len(new_block_maxpool))


        mask = np.zeros((30,30))
        for i in range (len(new_block_maxpool)):
                row_start = new_block_maxpool[i]["row_range"][0]
                row_end = new_block_maxpool[i]["row_range"][1]+1

                col_start = new_block_maxpool[i]["col_range"][0]
                col_end = new_block_maxpool[i]["col_range"][1]+1
            
                for i in range (row_start,row_end):
                    for j in range(col_start,col_end):
                        mask[i][j]=1
        
        # cont=0
        # for i in range (30):
        #     for j in range(30):
        #         if mask[i][j]==1:
        #             cont=cont+1
        #             print("i vaoli di interesse si trovano in posizione (",i,",",j,")")
        
    return mask


def give_foglio ( loaded_model,numeroscelto,x_test_normalized):
    
   

    input_test = x_test_normalized[numeroscelto]#l'unico che trova in maniera corretta 109
    newarr = input_test.reshape(1,32,32,3)  
    input_test = x_test_normalized[numeroscelto]#l'unico che trova in maniera corretta 109
    newarr = input_test.reshape(1,32,32,3)                

    get_1st_layer_output =K.function([loaded_model.layers[0].input],[loaded_model.layers[0].output])# ricordo che sto K.function
    layer_output = get_1st_layer_output(newarr)[0]# non so perché ci vuole sto zero
    

    righe= layer_output.shape[1] * layer_output.shape[2]
    colonne= layer_output.shape[3]
    foglio = np.zeros((righe,colonne))

    # Usa la funzione reshape per ottenere la matrice desiderata
    foglio = layer_output.reshape(righe, colonne)


    return foglio 

def clustering(data,mask):
         # Normalizzazione dei dati
    df = pd.DataFrame(data)
    scaled_df = StandardScaler().fit_transform(df)

    # Definisci il numero di cluster
    n_clusters = 10

    # Crea un'istanza dell'algoritmo K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Esegui il clustering sui dati
    kmeans.fit(scaled_df)

    # Ottieni le etichette dei cluster per ogni elemento
    labels = kmeans.labels_

    # Trasforma il vettore in una matrice in base alle dimensioni di mask
    mask_array = mask
    rows, cols = mask_array.shape
    matrice = labels[:rows*cols].reshape(rows, cols) #matrice delle labels

    # Calcola l'importanza dei cluster in base alla matrice mask
    importance = np.zeros(n_clusters)

    for i in range(rows):
        for j in range(cols):
            if mask_array[i, j] == 1:
                cluster_label = matrice[i, j]
                importance[cluster_label] += 1

    print(importance)
    # Trova il cluster più importante
    most_important_cluster = np.argmax(importance)

    # Stampa il cluster più importante
    print(f"Il cluster più importante è il cluster {most_important_cluster+1} con importanza: {importance[most_important_cluster]}")

    # Ottieni le posizioni degli elementi del cluster più importante
    positions_most_important_cluster = [(i, j) for i in range(rows) for j in range(cols) if matrice[i, j] == most_important_cluster]

    # # Stampa le posizioni degli elementi del cluster più importante
    # print("Posizioni degli elementi del cluster più importante:")
    # for pos in positions_most_important_cluster:
    #     print(pos)
    positions_array = np.array(positions_most_important_cluster).flatten()







    #sta cosa dovrei salvarla su file e poi darla in input al back layer , qua stiamo al convd2 , i dati che mi sevno solo dell'input quindi non sono solo quest ele posizioni corrette
    # Stampa gli elementi dell'array separati da virgola

    return positions_array
if __name__ == '__main__':
    # Loading CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


    #print(x_test.shape)#stampa (10000,32,32,3) ovvero 10000 elementi di (32,32,3)
    #print(x_test[0])
    # Preprocess the data
    x_train_normalized = x_train.astype('float32')
    x_test_normalized = x_test.astype('float32')
    x_train_normalized /= 255
    x_test_normalized /= 255

    
    #Convert class vectors to binary class matrices
    y_train_normalized = keras.utils.to_categorical(y_train, 10)
    y_test_normalized = keras.utils.to_categorical(y_test, 10)
   
    # Set up data augmentation
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')
                    #----------------------caricare il modello --------------------

    loaded_model = keras.models.load_model("C:/Users/Dell/Min8.h5")
    loaded_model.summary()
    loaded_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    
 
    numeroscelto=9237

    mask= back_from_most_important(loaded_model,numeroscelto,x_test_normalized)
    data = give_foglio ( loaded_model,numeroscelto,x_test_normalized)
    vector_importante = clustering(data,mask)

 #   print("Posizioni degli elementi del cluster più importante come vettore:")

    print(vector_importante)
        
    output = ', '.join(map(str, vector_importante))

    # Stampa il risultato
    print(output)
   