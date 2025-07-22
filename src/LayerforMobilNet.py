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
from load_dataset import load_test_dataset
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

optimizers = {"ResNet50DvsC": 'SGD', "MobilNet": tf.keras.optimizers.Adam(), "ResNet24": tf.keras.optimizers.Adam()}
losses = {"ResNet50DvsC": 'sparse_categorical_crossentropy', "MobilNet": tf.keras.losses.categorical_crossentropy, "ResNet24": tf.keras.losses.categorical_crossentropy}
        #queste liste per ogni rete dicono quali ottimizzatori e quali loss utilizzare 

  


def create_mask2(x_test,loaded_model,numeroscelto,pixel_indices,last_layer):
     #secondom me è logicamente sbagliato perché dovrebbe eessere un tensore, ma sti cazzi
   
    print (len(pixel_indices[0]), pixel_indices)
    print(pixel_indices[0][0],pixel_indices[0][1])
  
    livelloattuale=last_layer
     
    input_test = x_test[numeroscelto]#l'unico che trova in maniera corretta 109
    img_x=input_test.shape[0]
    img_y=input_test.shape[1]
    img_ch=input_test.shape[2]
    newarr = input_test.reshape(1,img_x,img_y,img_ch)  
    posizioni_da_salvare = set()    

    #     layer_attuale= loaded_model.layers[livelli]
    #     layer_config = layer_attuale.get_config()
    #     layer_attuale_shape= layer_attuale.output_shape#check che tutto vada bene
    #     print(f"Configurazioni del layer '{layer_attuale.__class__.__name__}':")

    #     vettore_massimi,vettore_posizioni_max=funzione_cerca_max(newarr,livelli,layer_attuale_shape,posizioni_da_salvare)  #funzione che mi permette di identificare gli elementi di maggior valore all'interno delle fibre

    for index in range(0,pixel_indices.shape[0],1):
        r=pixel_indices[index]
       # c=pixel_indices[0][index+1]
        posizioni_da_salvare.add(tuple(r))
    
    for livelli in range(last_layer,0, -1):

        layer_attuale= loaded_model.layers[livelli]
        layer_config = layer_attuale.get_config()
        nome_layer=layer_attuale.__class__.__name__
        layer_attuale_shape= layer_attuale.output_shape
        print(f"Configurazioni del layer '{layer_attuale.__class__.__name__}':")
        print(layer_config)

        layer_livello_precedente=loaded_model.layers[livelli-1]
        layer_precedente_shape= layer_livello_precedente.output_shape
        print("shape layer precedente",layer_precedente_shape)

        #sta cosa da il tipo
    #1)salvo i primi elementi in "POSIZIONI_DA SALVARE"
    #2)ho un layer conv2d, gli do in ingresso gli elementi di "posizioni_da_salvare", in uscita avrò i nuovi elementi da salvare
        print(nome_layer)
        if livelli == 1:
            creo_tensore_per_apparare=(layer_precedente_shape[0][0],layer_precedente_shape[0][1],layer_precedente_shape[0][2],layer_precedente_shape[0][3])
        else:
            creo_tensore_per_apparare=loaded_model.layers[livelli-1].output_shape

        
        #funzione per la back mapping
        # if(layer_attuale_shape[1] != creo_tensore_per_apparare[1]  and nome_layer!='ZeroPadding2D'):
        #     vettore_massimi,vettore_posizioni_max=funzione_cerca_max(newarr,livelli,layer_attuale_shape,posizioni_da_salvare)  #funzione che mi permette di identificare gli elementi di maggior valore all'interno delle fibre
        #     posizioni_da_salvare=set()#lo svuoto perché ormai prendo solo i nuovi elementi massimi
        #     for index,element in enumerate (vettore_posizioni_max):
        #         posizioni_da_salvare.add(tuple(element))

        elemnti_temporanei=posizioni_da_salvare     
        return_posizioni=esegui_funzione(nome_layer,layer_config,creo_tensore_per_apparare,elemnti_temporanei)#mi ritorna le posizioni in base al layer che sto elaborando
        if return_posizioni != elemnti_temporanei: 
            posizioni_da_salvare=set() # lo svuoto per inserire i nuovi elementi 
            for element in (return_posizioni):
                posizioni_da_salvare.add(tuple(element))
    
    
    riga=creo_tensore_per_apparare[1]
    colonna=creo_tensore_per_apparare[2]
    truemask = np.zeros((riga,colonna))
#------------l'ultima operazione sarà il ritorno della maschera delle posizioni 

    for elements in posizioni_da_salvare:
        row, col = elements
        truemask[row][col]=1


   
    return truemask

def max_pool_to_back(layer_config,layer_precedente_shape,dati_in_ingresso):
    pool_size = layer_config.get('pool_size',(3,3))
    stride = layer_config.get('strides', (1, 1))  # Imposta un valore di default se 'strides' non è presente
    #kernel_size = layer_config.get('kernel_size', (3, 3))  # Esempio di recupero di 'kernel_size' con valore di default
    #dilation = layer_config.get('dilation', (1, 1))  # Esempio di recupero di 'dilation' con valore di default
    padding = layer_config.get('padding', 'valid') 
    posizioni = []

    for idx,poss in enumerate(dati_in_ingresso):
        riga=poss[0]
        colonna = poss[1]
        if padding == 'valid':
                # Calcolo del blocco nella matrice originale
                old_row_start = riga * stride[0]
                old_col_start = colonna * stride[1]
        elif padding == 'same':
            # Calcolo delle coordinate di partenza tenendo conto della dilazione
            old_row_start = riga * stride[0] - pool_size[0] // 2
            old_col_start = colonna * stride[1] - pool_size[1] // 2
     
    
    old_row_end = old_row_start + pool_size[0] - 1
    old_col_end = old_col_start + pool_size[1] - 1
    # print("print old col end", old_col_end)
    # print("layer_livello_precedente, ", layer_livello_precedente)
    #controlli per non sforare
    if old_col_end>= layer_precedente_shape[2]:
        old_col_end=layer_precedente_shape[2]-1
    if old_row_end>= layer_precedente_shape[1]:
        old_row_end=layer_precedente_shape[1]-1

    if old_row_start < 0 : 
        old_row_start=0 #controlli per non sforare
    if old_col_start < 0 : 
        old_col_start=0    

    for row in range(old_row_start, old_row_end + 1,1):
            for col in range(old_col_start, old_col_end + 1, 1):
                posizioni.append((row, col))
                
    return posizioni
    

def back_position_pool_to_back(layer_config,layer_precedente_shape,dati_in_ingresso):
    pool_size = layer_config.get('pool_size',(3,3))
    stride = layer_config.get('strides', (1, 1))  # Imposta un valore di default se 'strides' non è presente
    #kernel_size = layer_config.get('kernel_size', (3, 3))  # Esempio di recupero di 'kernel_size' con valore di default
    #dilation = layer_config.get('dilation', (1, 1))  # Esempio di recupero di 'dilation' con valore di default
    padding = layer_config.get('padding', 'valid') 
    posizioni = []
    
    for idx,poss in enumerate(dati_in_ingresso):
        riga=poss[0]
        colonna = poss[1]
        if padding == 'valid':
            # Calcolo del blocco nella matrice originale
            old_row_start = riga * stride[0]
            old_col_start = colonna * stride[1]
        elif padding == 'same':
            # Calcolo delle coordinate di partenza tenendo conto della dilazione
            old_row_start = riga * stride[0] - pool_size[0] // 2
            old_col_start = colonna * stride[1] - pool_size[1] // 2
     
    
        old_row_end = old_row_start + pool_size[0] - 1
        old_col_end = old_col_start + pool_size[1] - 1
        # print("print old col end", old_col_end)
        # print("layer_livello_precedente, ", layer_livello_precedente)
        #controlli per non sforare
        if old_col_end>= layer_precedente_shape[2]:
            old_col_end=layer_precedente_shape[2]-1
        if old_row_end>= layer_precedente_shape[1]:
            old_row_end=layer_precedente_shape[1]-1

        if old_row_start < 0 : 
            old_row_start=0 #controlli per non sforare
        if old_col_start < 0 : 
            old_col_start=0    

        for row in range(old_row_start, old_row_end + 1,1):
                for col in range(old_col_start, old_col_end + 1, 1):
                    posizioni.append((row, col))
                
    return posizioni
def back_position_ConvDepth_to_back(layer_config,layer_precedente_shape,dati_in_ingresso):
    stride = layer_config.get('strides', (1, 1))  # Imposta un valore di default se 'strides' non è presente
    kernel_size = layer_config.get('kernel_size', (3, 3))  # Esempio di recupero di 'kernel_size' con valore di default
    dilation = layer_config.get('dilation', (1, 1))  # Esempio di recupero di 'dilation' con valore di default
    padding = layer_config.get('padding', 'valid')
    posizioni = []
     # print(f"nome del livello: {layer_config.__class__.__name__} , dimensione del pool:{layer_config.pool_size} , dimensione dello stride {livelli.strides}: ,dimensione del padding: {livelli.padding}, dytpe : {livelli.dtype}" )# in questo modo ottengo il tipo del layer , quindi applicherò u metodo differente in base al tipo
    for idx,poss in enumerate(dati_in_ingresso):
        riga=poss[0]
        colonna = poss[1]
        if padding == 'valid':
            # Calcolo del blocco nella matrice originale
            old_row_start = riga * stride[0]
            old_col_start = colonna * stride[1]
        elif padding == 'same':
            # Calcolo delle coordinate di partenza tenendo conto della dilazione
            old_row_start = riga * stride[0] - kernel_size[0] // 2
            old_col_start = colonna * stride[1] - kernel_size[1] // 2

        # Calcolo delle posizioni di fine
        old_row_end = old_row_start + (kernel_size[0] - 1) * dilation[0]
        old_col_end = old_col_start + (kernel_size[1] - 1) * dilation[1]

        # Controlli per non sforare
        if old_col_end >= layer_precedente_shape[2]:
            old_col_end = layer_precedente_shape[2] - 1
        if old_row_end >= layer_precedente_shape[1]:
            old_row_end = layer_precedente_shape[1] - 1

        if old_row_start < 0:
            old_row_start = 0  # Controlli per non sforare
        if old_col_start < 0:
            old_col_start = 0

        # Generazione delle tuple per ogni posizione
        
        for row in range(old_row_start, old_row_end + 1, dilation[0]):
            for col in range(old_col_start, old_col_end + 1, dilation[1]):
                posizioni.append((row, col))

    return posizioni

def back_position_Conv_to_back(layer_config,layer_precedente_shape,dati_in_ingresso):
    stride = layer_config.get('strides', (1, 1))  # Imposta un valore di default se 'strides' non è presente
    kernel_size = layer_config.get('kernel_size', (3, 3))  # Esempio di recupero di 'kernel_size' con valore di default
    dilation = layer_config.get('dilation', (1, 1))  # Esempio di recupero di 'dilation' con valore di default
    padding = layer_config.get('padding', 'valid') 
     # print(f"nome del livello: {layer_config.__class__.__name__} , dimensione del pool:{layer_config.pool_size} , dimensione dello stride {livelli.strides}: ,dimensione del padding: {livelli.padding}, dytpe : {livelli.dtype}" )# in questo modo ottengo il tipo del layer , quindi applicherò u metodo differente in base al tipo
    posizioni = []
    for idx,poss in enumerate(dati_in_ingresso):
        riga=poss[0]
        colonna = poss[1]
        if padding == 'valid':
            # Calcolo del blocco nella matrice originale
            old_row_start = riga * stride[0]
            old_col_start = colonna * stride[1]
        elif padding == 'same':
            # Calcolo delle coordinate di partenza tenendo conto della dilazione
            old_row_start = riga * stride[0] - kernel_size[0] // 2
            old_col_start = colonna * stride[1] - kernel_size[1] // 2

        # Calcolo delle posizioni di fine
        old_row_end = old_row_start + (kernel_size[0] - 1) * dilation[0]
        old_col_end = old_col_start + (kernel_size[1] - 1) * dilation[1]

        # Controlli per non sforare
        if old_col_end >= layer_precedente_shape[2]:
            old_col_end = layer_precedente_shape[2] - 1
        if old_row_end >= layer_precedente_shape[1]:
            old_row_end = layer_precedente_shape[1] - 1

        if old_row_start < 0:
            old_row_start = 0  # Controlli per non sforare
        if old_col_start < 0:
            old_col_start = 0

        # Generazione delle tuple per ogni posizione
        
        for row in range(old_row_start, old_row_end + 1, dilation[0]):
            for col in range(old_col_start, old_col_end + 1, dilation[1]):
                posizioni.append((row, col))
                
    return posizioni

def back_position_ZeroPadding_to_back(layer_config, layer_precedente_shape, dati_in_ingresso):
    padding = layer_config.get('padding', ((0, 0), (0, 0)))  # Recupero del padding come tupla di tuple
    padding_riga = padding[0][0]
    padding_colonna = padding[1][0]

    posizioni = []

    for idx, poss in enumerate(dati_in_ingresso):
        riga = poss[0]
        colonna = poss[1]

        # Calcolo del blocco nella matrice originale
        old_row_start = riga - padding_riga
        old_col_start = colonna - padding_colonna

        # Controlli per non sforare
        if old_row_start < 0:
            old_row_start = 0
        if old_col_start < 0:
            old_col_start = 0

        old_row_end = old_row_start
        old_col_end = old_col_start

        if old_col_end >= layer_precedente_shape[1]:
            old_col_end = layer_precedente_shape[1] - 1
        if old_row_end >= layer_precedente_shape[2]:
            old_row_end = layer_precedente_shape[2] - 1

        # Iterazione su tutte le posizioni della griglia considerando la dilatazione
        for row in range(old_row_start, old_row_end + 1,1):
            for col in range(old_col_start, old_col_end + 1, 1):
                posizioni.append((row, col))

    return posizioni


def funzione_cerca_max(loaded_model,newarr,livelli,layer_attuale_shape,elementi_salvati):
    get_last_layer_output =K.function([loaded_model.layers[0].input],[loaded_model.layers[livelli].output])# layer2 sarebbe il secondo conv2d m, non mi serve il successivo maxpooling
    layer_output = get_last_layer_output(newarr)[0] #layer_output è il valore dell'output al layer richiesto
    print(layer_output.shape)


    # Calcolare il numero di elementi totali
    total_elements = layer_attuale_shape[1] * layer_attuale_shape[2]
    percentage = 0.10
    num_max_values = round(total_elements * percentage)
    if num_max_values == 0 : num_max_values = 1
    # Inizializzazione delle variabili
    vett_max = np.full(num_max_values, -np.inf)
    vet_pos_max = np.zeros((num_max_values, 2), dtype=int)
    current_value=0
    mask = np.zeros((layer_attuale_shape[1],layer_attuale_shape[2]))
    
    if len(elementi_salvati) == 0 :
        for a in range(layer_attuale_shape[1]):
            for b in range(layer_attuale_shape[2]):
                current_value = layer_output[0, a, b, :].max()
                if current_value > vett_max.min():
                    min_index = vett_max.argmin()
                    vett_max[min_index] = current_value
                    vet_pos_max[min_index] = [a, b]
    else:
        for elements in elementi_salvati:
                rows, cols = elements
                mask[rows][cols]=1


        # Trova i 10% valori massimi e le loro posizioni
        for a in range(layer_attuale_shape[1]):
            for b in range(layer_attuale_shape[2]):
                current_value = layer_output[0, a, b, :].max()
                if mask[a][b]==1:
                    #  current_value = layer_output[0, a, b]
                    if current_value > vett_max.min():
                        min_index = vett_max.argmin()
                        vett_max[min_index] = current_value
                        vet_pos_max[min_index] = [a, b]


    print("Elementi massimi trovati (10%):", vett_max)
    print("Posizioni degli elementi massimi trovati (10%):",vet_pos_max)
   
    return vett_max,vet_pos_max


def esegui_funzione(nome,layer_config,layer_precedente_shape,dati_in_ingresso):
    switcher = {
        'Conv2D': back_position_Conv_to_back,
        'DepthwiseConv2D': back_position_ConvDepth_to_back,
        "ZeroPadding2D": back_position_ZeroPadding_to_back,
        "AveragePooling2D": back_position_pool_to_back,
        "MaxPooling2D" : back_position_pool_to_back
    }
    func = switcher.get(nome, lambda *args: dati_in_ingresso)
    return func(layer_config,layer_precedente_shape,dati_in_ingresso)

def back_from_most_important(loaded_model,numeroscelto,x_test,last_layer):

    input_test = x_test[numeroscelto]#l'unico che trova in maniera corretta 109
    img_x=input_test.shape[0]
    img_y=input_test.shape[1]
    img_ch=input_test.shape[2]
    newarr = input_test.reshape(1,img_x,img_y,img_ch)  

    get_last_layer_output =K.function([loaded_model.layers[0].input],[loaded_model.layers[-3].output])# layer2 sarebbe il secondo conv2d m, non mi serve il successivo maxpooling
    layer_output = get_last_layer_output(newarr)[0]

    _,dimx,dimy,dimn=layer_output.shape
    max = 0
    n = x = y = 0


    flatten_index = None
    for i, layer in enumerate(loaded_model.layers):
        if layer.__class__.__name__== 'Flatten' :
            flatten_index = i
            break
        if layer.__class__.__name__== 'Dense' :
            Dense_index = i
            break

    if flatten_index is not None:
        last_layer_index=flatten_index-1 #andare prima del flatten
    elif Dense_index is not None:
        last_layer_index=Dense_index-2 #andare prima del dense

    #inizializzazione---> non serve secondo me
    posizioni_da_salvare = set()    #inizializzo la struttura che mi salva le posizioni ad ogni layer

    layer_attuale= loaded_model.layers[last_layer_index]
    layer_config = layer_attuale.get_config()
    layer_attuale_shape= layer_attuale.output_shape#check che tutto vada bene
    print(f"Configurazioni del layer '{layer_attuale.__class__.__name__}':")

    vettore_massimi,vettore_posizioni_max=funzione_cerca_max(loaded_model,newarr,last_layer_index,layer_attuale_shape,posizioni_da_salvare)  #funzione che mi permette di identificare gli elementi di maggior valore all'interno delle fibre
    for index,element in enumerate (vettore_posizioni_max):
        posizioni_da_salvare.add(tuple(element))
    
    for livelli in range(last_layer_index,last_layer, -1):

        layer_attuale= loaded_model.layers[livelli]
        layer_config = layer_attuale.get_config()
        nome_layer=layer_attuale.__class__.__name__
        layer_attuale_shape= layer_attuale.output_shape
        print(f"Configurazioni del layer '{layer_attuale.__class__.__name__}':")
        print(layer_config)

        layer_livello_precedente=loaded_model.layers[livelli-1]
        layer_precedente_shape= layer_livello_precedente.output_shape
        print("shape layer precedente",layer_precedente_shape)

          #sta cosa da il tipo
    #1)salvo i primi elementi in "POSIZIONI_DA SALVARE"
    #2)ho un layer conv2d, gli do in ingresso gli elementi di "posizioni_da_salvare", in uscita avrò i nuovi elementi da salvare
        print(nome_layer)

      #  #funzione per la back mapping
        # if(layer_attuale_shape[1] != layer_precedente_shape[1]  and nome_layer!='ZeroPadding2D'):
        #     vettore_massimi,vettore_posizioni_max=funzione_cerca_max(newarr,livelli,layer_attuale_shape,posizioni_da_salvare)  #funzione che mi permette di identificare gli elementi di maggior valore all'interno delle fibre
        #     posizioni_da_salvare=set()#lo svuoto perché ormai prendo solo i nuovi elementi massimi
        #     for index,element in enumerate (vettore_posizioni_max):
        #         posizioni_da_salvare.add(tuple(element))

        elemnti_temporanei=posizioni_da_salvare     
        return_posizioni=esegui_funzione(nome_layer,layer_config,layer_precedente_shape,elemnti_temporanei)#mi ritorna le posizioni in base al layer che sto elaborando
        if return_posizioni != elemnti_temporanei: 
            posizioni_da_salvare=set() # lo svuoto per inserire i nuovi elementi 
            for element in (return_posizioni):
                posizioni_da_salvare.add(tuple(element))
    
    
    # riga=layer_attuale_shape[1]
    # colonna=layer_attuale_shape[2]
    riga = layer_precedente_shape[1]
    colonna = layer_precedente_shape[2]
    mask = np.zeros((riga,colonna))
#------------l'ultima operazione sarà il ritorno della maschera delle posizioni 

    for elements in posizioni_da_salvare:
        row, col = elements
        mask[row][col]=1


    return mask


def give_foglio ( loaded_model,numeroscelto,x_test_normalized,last_layer):
    
    input_test = x_test_normalized[numeroscelto]
    img_x=input_test.shape[0]
    img_y=input_test.shape[1]
    img_ch=input_test.shape[2]
    newarr = input_test.reshape(1,img_x,img_y,img_ch)  
    

   
               
    #RICORDA HO APPENA MESSO IO '1', PERCHè CON 0 FORSE NON ANDAVA BENE, perché sarebbe l'ouptut dell'input layer, a me serve l'ouptput del prio layer conv
    get_1st_layer_output =K.function([loaded_model.layers[0].input],[loaded_model.layers[last_layer].output])# ricordo che sto K.function
    layer_output = get_1st_layer_output(newarr)[0]# non so perché ci vuole sto zero
    

    righe= layer_output.shape[1] * layer_output.shape[2]
    colonne= layer_output.shape[3]
    foglio = np.zeros((righe,colonne))

    # Usa la funzione reshape per ottenere la matrice desiderata
    foglio = layer_output.reshape(righe, colonne)


    return foglio 

def clustering(data,mask,numeroscelto):
         # Normalizzazione dei dati
    df = pd.DataFrame(data)
    scaled_df = StandardScaler().fit_transform(df)

    # Definisci il numero di cluster
    n_clusters = 10 #con 120 funziona bene

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

    
    # Calcolo delle dimensioni di ogni cluster


    cluster_sizes = np.zeros(n_clusters)
    for i in range(rows):
        for j in range(cols):
            cluster_label = matrice[i, j]
            cluster_sizes[cluster_label] += 1

    # Calcolo dell'importanza dei cluster in base alla matrice mask
    importance = np.zeros(n_clusters)
    for i in range(rows):
        for j in range(cols):
            if mask_array[i, j] == 1:
                cluster_label = matrice[i, j]
                importance[cluster_label] += 1

    # Calcolo del rapporto importanza/dimensione per ogni cluster
    importance_ratio = importance 
    # Trova il cluster con il rapporto importanza/dimensione più alto
    most_important_cluster = np.argmax(importance_ratio)
    max_ratio = importance_ratio[most_important_cluster]

    # Stampa il cluster più importante
    print(f"Il cluster più importante è il cluster {most_important_cluster + 1}  importanza: {max_ratio}")

    # Ottieni le posizioni degli elementi del cluster più importante
    positions_most_important_cluster = [(i, j) for i in range(rows) for j in range(cols) if matrice[i, j] == most_important_cluster]

    # Stampa le dimensioni di ogni cluster
    for cluster_idx in range(n_clusters):
        print(f"Dimensione del cluster {cluster_idx + 1}: {cluster_sizes[cluster_idx]}")

    # Stampa il rapporto importanza/dimensione per ogni cluster
    for cluster_idx in range(n_clusters):
        print(f"Rapporto importanza/dimensione per il cluster {cluster_idx + 1}: {importance_ratio[cluster_idx]}")
   
   

    # Genera i colori per la heatmap-----colori 20 clustering
    def generate_colors(num_colors):
        colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
        return [tuple(color) for color in colors]

    def create_colormap(num_colors):
        colors = generate_colors(num_colors)
        return ListedColormap(colors)

    cmap = create_colormap(len(np.unique(labels)))


    plt.imshow(matrice, cmap=cmap, interpolation='nearest')

    colori = generate_colors(len(np.unique(labels)))
    patches = [Patch(color=colori[i], label=f'Cluster {i+1}') for i in range(len(np.unique(labels)))]
    plt.legend(handles=patches, title='Colori cluster', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
#    plt.savefig("C:/Users/Dell/Desktop/Piante/newtry/clusters/PiantaClusterone{}.png".format(numeroscelto))#da cambiare in base a quale dataset sto utilizzando
    plt.close()
    

    positions_array = np.array(positions_most_important_cluster).flatten()


    return positions_array

#dbscan
# def clustering(data, mask, numeroscelto):
#     # Normalizzazione dei dati
#     df = pd.DataFrame(data)
#     scaled_df = StandardScaler().fit_transform(df)

#     # Crea un'istanza dell'algoritmo DBSCAN
#     dbscan = DBSCAN(eps=0.5, min_samples=5)
#     dbscan.fit(scaled_df)

#     # Ottieni le etichette dei cluster per ogni elemento
#     labels = dbscan.labels_

#     # Trasforma il vettore in una matrice in base alle dimensioni di mask
#     mask_array = mask
#     rows, cols = mask_array.shape
#     matrice = labels[:rows * cols].reshape(rows, cols)

#     # Calcolo delle dimensioni di ogni cluster
#     unique_labels = set(labels)
#     n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
#     cluster_sizes = np.zeros(len(unique_labels))

#     for i in range(rows):
#         for j in range(cols):
#             cluster_label = matrice[i, j]
#             if cluster_label != -1:  # Evita gli outlier
#                 cluster_sizes[cluster_label] += 1

#     # Calcolo dell'importanza dei cluster in base alla matrice mask
#     importance = np.zeros(len(unique_labels))
#     for i in range(rows):
#         for j in range(cols):
#             if mask_array[i, j] == 1:
#                 cluster_label = matrice[i, j]
#                 if cluster_label != -1:  # Evita gli outlier
#                     importance[cluster_label] += 1

#     # Trova il cluster con l'importanza più alta
#     importance_ratio = importance
#     most_important_cluster = np.argmax(importance_ratio)
#     max_ratio = importance_ratio[most_important_cluster]

#     # Stampa il cluster più importante
#     print(f"Il cluster più importante è il cluster {most_important_cluster + 1}, importanza: {max_ratio}")

#     # Ottieni le posizioni degli elementi del cluster più importante
#     positions_most_important_cluster = [
#         (i, j) for i in range(rows) for j in range(cols) if matrice[i, j] == most_important_cluster
#     ]

#     # Stampa le dimensioni di ogni cluster
#     for cluster_idx in range(len(unique_labels)):
#         if cluster_idx != -1:
#             print(f"Dimensione del cluster {cluster_idx + 1}: {cluster_sizes[cluster_idx]}")

#     # Stampa il rapporto importanza/dimensione per ogni cluster
#     for cluster_idx in range(len(unique_labels)):
#         if cluster_idx != -1:
#             print(f"Rapporto importanza/dimensione per il cluster {cluster_idx + 1}: {importance_ratio[cluster_idx]}")

#     # Visualizzazione del clustering con una heatmap
#     def generate_colors(num_colors):
#         colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
#         return [tuple(color) for color in colors]

#     def create_colormap(num_colors):
#         colors = generate_colors(num_colors)
#         return ListedColormap(colors)

#     cmap = create_colormap(len(np.unique(labels)))
#     plt.imshow(matrice, cmap=cmap, interpolation='nearest')

#     colori = generate_colors(len(np.unique(labels)))
#     patches = [Patch(color=colori[i], label=f'Cluster {i+1}') for i in range(len(np.unique(labels)))]
#     plt.legend(handles=patches, title='Colori cluster', loc='center left', bbox_to_anchor=(1, 0.5))
# #    plt.show()
#     plt.savefig("C:/Users/Dell/Desktop/CaniVsDOgghi/new/PiantaClusterone{}.png".format(numeroscelto))#da cambiare in base a quale dataset sto utilizzando
#     plt.close()
    
#     positions_array = np.array(positions_most_important_cluster).flatten()
#     return positions_array


# if __name__ == '__main__':
   
    
#     #carico il modello
#     model_name = "MobilNet.h5"
   
#     loaded_model = tf.keras.models.load_model("C:/Users/Dell/Desktop/partefinalwe/MobilNet.h5")
#     loaded_model.summary()
    
#     x_test, y_test = load_test_dataset(loaded_model.input_shape, model_name)
#     print(x_test.shape)#(501, 224, 224, 3)
#     print(y_test.shape)#(501, 2)
#     casi_di_studio=[]
#     contatore_successo=0
#     contatore_fallimento=0
    
#     array_list = []
#     risultati=[]
#     dim_campioni = x_test.shape[0]
#     for i in range(55,60,1):#dim_campioni
#         numeroscelto=i    
#         input_test = x_test[numeroscelto]
#         dX=input_test.shape[0]
#         dY=input_test.shape[1]
#         dZ=input_test.shape[2]
#         newarr = input_test.reshape(1,dX,dY,dZ)    
#         #carico il dataset cifar10 

#         #IN QUESTO CASO FORSE DEVO RIVEDERE IL CLUSTERING , A ME PARE LO FACCIA AL LIVELLO SBAGLITO, HO PROVATO A CORREGGERE PERò POI MI ESCE SFASATO SULL'IMMAGINE ORIGINALE

#         #----------------------------------input non modificiato---------------------
#         inputunchanged= copy.deepcopy(input_test)

#         previsione1 = loaded_model.predict(newarr) 
#         plt.figure()

#         print ("print input: \n",previsione1)
#         if (np.argmax(y_test[numeroscelto]) != np.argmax(previsione1)):
#             print("previsione sbagliata")
#             continue  
#         else:
#             print("previsione corretta")
#         # plt.imshow(input_test)
#         # plt.show()    

            
#         image_label = np.argmax(previsione1)
#         test_acc =np.max(previsione1)
#         image_label_arr = []
#         image_label_arr.append(image_label)
#         image_label_arr = np.array(image_label_arr)
#         classes = ["Foglia Sana","Malatia 1", "Malattia 2", "Malattia 3"]
#         image_label_arr = tf.keras.utils.to_categorical(image_label_arr, len(classes))
#         inputunchanged= copy.deepcopy(input_test)
#         predicted_label=image_label

#         plt.imshow(input_test)  # To change [-1, 1] to [0,1]
#         image_class = classes[image_label] 
#         plt.title('{} : {:.2f}% Confidence'.format(image_class, test_acc*100))
#         #plt.show()
        
#         print(classes[image_label] + " : " + str(image_label))
#         print(image_label_arr)
#         plt.savefig("C:/Users/Dell/Desktop/Piante/Basic/Input{}.png".format(numeroscelto))
#         plt.close()

    
#         dim_X = input_test.shape[0]
#         dim_y = input_test.shape[1]
#         last_layer=8
#         #ottenimento della maschera, e del vettore importante
#         mask= back_from_most_important(loaded_model,numeroscelto,x_test,last_layer)
#         data = give_foglio ( loaded_model,numeroscelto,x_test,last_layer) #ricordo che sta funzione mi permette di ottenere i risultati del primo layer di convoluzione
#         vector_importante = clustering(data,mask)

#         pixel_vector=vector_importante
#         pixel_indices = pixel_vector.reshape(-1, 2)

#     # Ordina gli elementi in base alla coordinata x (prima colonna) e poi y (seconda colonna)
#         indici_ordinati = np.lexsort((pixel_indices[:, 1], pixel_indices[:, 0]))

#     # Riordina gli elementi in base agli indici ottenuti
#         elementi_ordinati = pixel_indices[indici_ordinati]

#         rivettore = elementi_ordinati.reshape(-1,len(pixel_vector))   
#         mask = create_mask2(numeroscelto,rivettore,last_layer)  # Valori casuali 0 o 1

#         #--------------------------------------------------------------------------------------------------------------#----------------------------------------------------------------------------------------------------------------------------------
#         matrix = mask

#         # Configura la figura e gli assi
#         fig, ax = plt.subplots()

#         # Usa imshow per visualizzare la matrice
#         # cmap='gray' usa una scala di grigi (0=bianco, 1=nero)
#         cax = ax.imshow(matrix, cmap='gray_r', interpolation='none')

#         # Rimuove gli assi
#         ax.axis('off')
#         center_row = (matrix.shape[0] - 224) // 2
#         center_col = (matrix.shape[1] - 224) // 2
#         matrix_224x224 = matrix[center_row:center_row+224, center_col:center_col+224]

#         matrix_expanded = np.stack((matrix, matrix, matrix), axis=-1) 
#        # matrix_expanded = np.stack((matrix, matrix, matrix), axis=-1) 
#         new_im= matrix_expanded +input_test
#         # Mostra l'immagine
        

#         plt.imshow(new_im)  
#         plt.savefig("C:/Users/Dell/Desktop/Piante/CLustpiuImm/inp{}.png".format(numeroscelto))
#         #plt.show()
#         plt.close()
   