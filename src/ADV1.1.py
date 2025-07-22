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
from functools import partial
import csv
import ast
#allenando bene il modello non cambio pià le immagini , i punti di focus so sempre gli stessi m ail modello non sbaglia , ansi , alcune volte migliora
import copy
import time
import concurrent.futures
import socket
import queue 
import multiprocessing
import openpyxl
from openpyxl import Workbook
import codformask
import buckets
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from concurrent.futures import ThreadPoolExecutor, as_completed

from multiprocessing import Process
image_shape = (32, 32, 3)

def get_mse_psnr(a, b):
    max_ab = float(np.nanmax(np.concatenate((a, b))))
    mse = np.nanmean((np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)) ** 2)
    if mse == 0:
        return 0, 100
    return mse, 20 * np.log10(max_ab /(np.sqrt(mse)))

def create_mask2(pixel_indices):
    #secondom me è logicamente sbagliato perché dovrebbe eessere un tensore, ma sti cazzi
    mask = np.zeros((32, 32, 3))
    p=0
    
    bast = False
  #  print (len(pixel_indices[0]), pixel_indices)
   # print(pixel_indices[0][0],pixel_indices[0][1])
    for m in range (32):  #(start,stop,step)--> quandi parto a 26 e finisco a 28, il terminatore non viene mai conteggiato ,  il terminatore non viene mai conteggiato (11,14) righe , (12,15) col sbaglia la predizione ma il val max non cambia
        for n in range(32) :
            if(p == len(pixel_indices[0])):
                bast=True
                break
            if(pixel_indices[0][p] == m).all() and (pixel_indices[0][p+1] == n).all():
        #    print(" n'agg capit se tocca pur eil 17 : ", n)
                mask[m][n][0]=mask[m][n][1]=mask[m][n][2]= 1
               # print(pixel_indices[0][p],pixel_indices[0][p+1])
                p=p+2
               
            
        if bast==True:
            break  

    # for i in range(32):
    #         for x in range(32):
    #             for y in range(3):
    #                 if(  mask[i][x][y]==1):
    #                     contatore= contatore+1

    # print("se non esce  70*3 = errore)", contatore)

    truemask=np.zeros((32, 32))
    for x in range(32):
        for y in range (32):
            if mask[x,y,0]==1:
                for p in range (3):
                    for q in range (3):
                        truemask[x+p,y+q]=1

    # contatore=0
    # for i in range(32):
    #     for x in range(32):
    #         for y in range(3):
    #             if(  truemask[i][x]==1):
    #                 contatore= contatore+1

    # print("se non esce  >210 errore ", contatore)
 
    return truemask



# Funzione per calcolare MSE e PSNR
def get_mse_psnr(a, b):
    max_ab = float(np.nanmax(np.concatenate((a, b))))
    mse = np.nanmean((np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)) ** 2)
    if mse == 0:
        return 0, 100
    return mse, 20 * np.log10(max_ab / (np.sqrt(mse)))

# Funzione per calcolare il peso di un pixel modificato
def calcola_peso(immagine_originale, immagine_modificata):
    mse, psnr = get_mse_psnr(immagine_originale, immagine_modificata)
    return psnr

# Funzione per calcolare il valore di una modifica ai pixel
def calcola_valore(previsione_iniziale, previsione_modificata):
    max_previsione_iniziale = np.max(previsione_iniziale)
    max_previsione_modificata = np.max(previsione_modificata)
    predict_diff = max_previsione_iniziale - max_previsione_modificata
    indice_maxprevisione_iniziale = np.argmax(previsione_iniziale)
    indice_maxrevisione_modificata = np.argmax(previsione_modificata)
    if(indice_maxprevisione_iniziale !=indice_maxrevisione_modificata):
        predict_diff=100  #può capitare che si trova già una previsione sbagliata ma non controllando l'indice non veniva acchiappato
    return predict_diff

############# NOTE PAZZE POTREI FARLO RICORSIVOOO
# Funzione principale per selezionare i pixel da modificare, fa una modifica alla volta 
#OB. trade-off, numero di pixel da modificare, numero di cicli,  valore della perturbazione, 
def seleziona_pixel(numeroscelto,modificabili, immagine, limite_visibilita, loaded_model, perturbazione,max_iter=1, tolleranza=0.01):
    best_pixel = None
    immagine_modificata = np.copy(immagine)
    #with lock:
    previsione_iniziale = loaded_model.predict(immagine[np.newaxis, ...])
    riuscito = False
    #pixel_pesi_valori = []
    perturbazioni_range = perturbazione * np.array([-1, 1])

    #print("Perturbazione attuale:", perturbazione , "/255")
    #prev = loaded_model.predict(immagine_modificata[np.newaxis, ...])
   # print("Previsione prima di qualsiasi modifica", np.max(prev))

    def process_pixel(pixel):
        riga, colonna = pixel
        best_perturbazione = None
        valori_imp = None
        model = keras.models.load_model("C:/Users/Dell/Min8.h5")
       # for delta_r in perturbazioni_range:   #da rimettere se voglio fare focus su perturbazioni positive e/o negative
        for delta_r in perturbazioni_range:   #da rimettere se voglio fare focus su perturbazioni positive e/o negative
            for delta_g in perturbazioni_range: #il problema è che le modifiche qua non tengono conto delle precedenti
                for delta_b in perturbazioni_range:
                    # delta_b=delta_r
                    # delta_g=delta_r
                    # perturbazione_corrente = np.array([delta_r, delta_g, delta_b])
                    # #canali_originali = immagine[riga, colonna]
                    # #canali_modificati = immagine_modificata[riga, colonna]
                    # canali_modificati_temp = canali_modificati +  perturbazione_corrente #applico la perturbazione di questo ciclo 
                    # immagine_modificata_temp = np.copy(immagine_modificata)          
                    # immagine_modificata_temp[riga, colonna] = canali_modificati_temp#assegno la perturbazione all'immagine temporanea (non all'img originale)
                    # immagine_modificata_temp[riga, colonna] = tf.clip_by_value(immagine_modificata_temp[riga, colonna], 0, 1)
                    # peso = calcola_peso(immagine, immagine_modificata_temp) #peso è il grado di modifica più è alto meglio è (quanto è poco visibile la modifica)
                    # previsione_modificata = loaded_model.predict(immagine_modificata_temp[np.newaxis, ...])
                    # valore = calcola_valore(previsione_iniziale, previsione_modificata)
                    # peso = 100-peso# necessario perché a me il peso ha un'accezione positiva, più è alto e meglio è , nel problema dello zaino no, per cui , per concentrare il focus su sta cosa faccio il val max del psnr = 100- il psnr trovato , in  modo tale che più è alto il psnr dell amodifica attuale e meno impatta sul valore, se invece la modifica è troppo evidente èè giusto che impatti maggior mente sul valore 
                    # combined_metric = (valore * 100) / peso
                    
                    perturbazione_corrente = np.array([delta_r, delta_g, delta_b])
                    canali_modificati_temp = immagine_modificata[riga, colonna] + perturbazione_corrente
                    immagine_modificata_temp = np.copy(immagine_modificata)
                    immagine_modificata_temp[riga, colonna] = tf.clip_by_value(canali_modificati_temp, 0, 1)
                    peso = calcola_peso(immagine, immagine_modificata_temp)
                    #with lock
                    previsione_modificata = model.predict(immagine_modificata_temp[np.newaxis, ...])
                    valore = calcola_valore(previsione_iniziale, previsione_modificata)
                    peso = 100 - peso
                    combined_metric = (valore * 100) / peso


                    if best_perturbazione is None or combined_metric > best_perturbazione:
                        best_perturbazione = combined_metric
                        valori_imp = (peso, valore, perturbazione_corrente)

        if valori_imp is not None:
            peso, valore, perturbazione_corrente = valori_imp
            if valore > 0 and peso >= limite_visibilita:
                return pixel, peso, valore, perturbazione_corrente
        return None
    
    num_threads=9
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        #max_workers = executor._max_workers
  #      print(f"Numero di thread creati: {max_workers}") 
        results = list(executor.map(process_pixel, modificabili))
    #insomma in sto modo creo una lista , passo all'executor dei trhead il processo e modificabili , foss il suo input, ftto ciò tutti i risultati di sta funzione vengono passati in una lista
  

    results = [result for result in results if result is not None]

    if not results:
        print("Nessun pixel modificabile con peso sufficiente")    
        return False,modificabili

    scala_valore = 100
    combined_metrics = [(p, peso, valore * scala_valore, perturbazione_corrente) for p, peso, valore, perturbazione_corrente in results]
    combined_metrics.sort(key=lambda x: x[2] / x[1], reverse=True)
        
    pixel_modificabili_veloci= []

    if len(combined_metrics) < 9:
        for i in range (len(combined_metrics)):
         pixel_modificabili_veloci.append(combined_metrics[i][0])
    else:
        for i in range (9):
            pixel_modificabili_veloci.append(combined_metrics[i][0])
        
    best_pixel, peso, valore, perturbazione_corrente = combined_metrics[0]
    riga, colonna = best_pixel
    immagine_modificata[riga, colonna] += perturbazione_corrente
    immagine_modificata[riga, colonna] = tf.clip_by_value(immagine_modificata[riga, colonna], 0, 1)
    
#    print(f"Miglior pixel: {best_pixel}, Peso: {peso}, Valore: {valore}")

    if valore < tolleranza:
        print("Nessun pixel modificabile con peso sufficiente")    
        return False,modificabili
   # print("Perturbazione attuale:", perturbazione , "/255")
  #  with lock:
    prev = loaded_model.predict(immagine_modificata[np.newaxis, ...])
  #  print(np.max(prev))
    if np.argmax(y_test_normalized[numeroscelto]) != np.argmax(prev):
        riuscito = True
 #       print("Tappost")

    a = calcola_peso(immagine, immagine_modificata)
 #   print("Valore di view nel ciclo della funzione:", a)
    # if not isinstance(riuscito, bool) or pixel_modificabili_veloci is None:
    #     #raise ValueError("La funzione seleziona_pixel deve restituire un booleano e un vettore")
    #     for i in range (9):
    #         pixel_modificabili_veloci.append([0,0])
    


    #print("print fantasiosa per capire cosa sta ritornando che causa l'errore",riuscito,pixel_modificabili_veloci)
    return riuscito,pixel_modificabili_veloci

# def trova_minima_perturbazione(immagine,modificabili, limite_visibilita,loaded_model,massimo_iniziale, tolleranza=1): #1e-5
#     left = 0
#     right = massimo_iniziale
#     best_perturbazione = massimo_iniziale
    
#     while right - left > tolleranza:
#         # prev=loaded_model.predict(immagine[np.newaxis,...])
#         # print("per sicurezza , sto valore deve essere sempre 9 .... ->",np.max(prev))
#         #
#         mid = (left + right) // 2
#         #mid = (left + right) /2
#         mid_normalizzato =mid/255
#         if seleziona_pixel(modificabili, immagine, limite_visibilita, loaded_model, mid_normalizzato ):#qua ci metto midnormalizzto / mid
#             best_perturbazione = mid
#             right = mid  # dimezza la perturbazione
#         else:
#             left = mid  # prendi il valore intermedio
        
#     return best_perturbazione

def trova_minima_perturbazione(numeroscelto,immagine,modificabili, limite_visibilita,loaded_model,massimo_iniziale, tolleranza=1): #1e-5
    left = 0
    right = massimo_iniziale
    best_perturbazione = massimo_iniziale
    vector_pixel_mod=modificabili

    while right - left > tolleranza:
        # prev=loaded_model.predict(immagine[np.newaxis,...])
        # print("per sicurezza , sto valore deve essere sempre 9 .... ->",np.max(prev))
        #
        mid = (left + right) // 2
        #mid = (left + right) /2
        mid_normalizzato =mid/255
        valBool, vector_pixel_mod = seleziona_pixel(numeroscelto,vector_pixel_mod, immagine, limite_visibilita, loaded_model, mid_normalizzato )
        if valBool:#qua ci metto midnormalizzto / mid
            best_perturbazione = mid
            right = mid  # dimezza la perturbazione
        else:
            left = mid  # prendi il valore intermedio
        
    return best_perturbazione


def process_bucket(bucket_index, bucket_element, x_test_normalized):
        tasso_successo = 0
        save_results = []
        loaded_model=keras.models.load_model("C:/Users/Dell/Min8.h5")

        for element_index, element in enumerate(bucket_element):
            numeroscelto = element[0]
            print(numeroscelto)
            input_test = x_test_normalized[numeroscelto]

            mask = codformask.back_from_most_important(loaded_model, numeroscelto, x_test_normalized)
            data = codformask.give_foglio(loaded_model, numeroscelto, x_test_normalized)
            vector_importante = codformask.clustering(data, mask)

            pixel_vector = vector_importante
            pixel_indices = pixel_vector.reshape(-1, 2)

            indici_ordinati = np.lexsort((pixel_indices[:, 1], pixel_indices[:, 0]))
            elementi_ordinati = pixel_indices[indici_ordinati]
            rivettore = elementi_ordinati.reshape(-1, len(pixel_vector))
            mask = create_mask2(rivettore)
            pixel_modificabili = np.argwhere(mask == 1)

            immagine = input_test
            modificabili = pixel_modificabili
            limite_visibilita = 30
            massimo_iniziale = 255
            previsione = 0

            minima_perturbazione = trova_minima_perturbazione(numeroscelto, immagine, modificabili, limite_visibilita, loaded_model, massimo_iniziale)
            if minima_perturbazione != 255:
                previsione = 1

            save_results.append((bucket_index, element_index, previsione, minima_perturbazione))
            tasso_successo += previsione

        if tasso_successo != 0:
            tasso_successo = tasso_successo / len(bucket_element)
        
        return bucket_index, tasso_successo, save_results

if __name__ == '__main__':
    # Loading CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    start_time = time.time()
    x_train_normalized = x_train.astype('float32')
    x_test_normalized = x_test.astype('float32')
    x_train_normalized /= 255
    x_test_normalized /= 255

    #Convert class vectors to binary class matrices
    y_train_normalized = keras.utils.to_categorical(y_train, 10)
    y_test_normalized = keras.utils.to_categorical(y_test, 10)
   
    # Set up data augmentation
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')

    loaded_model = keras.models.load_model("C:/Users/Dell/Min8.h5")
    loaded_model.summary()
    loaded_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    

    #----------------------------------input non modificiato---------------------
    #min_confidence,bucket_size,b= buckets.returnBuskets(loaded_model,x_test_normalized,y_test_normalized)
    nome='C:/Users/Dell/Desktop/pyfiles/baskets-backup.csv'
    intestazione = ["immagine usata:", "buskets size:" ,"busket :"]
    # with open(nome, 'w', newline='') as file_csv:
    #    writer = csv.writer(file_csv)
    #    writer.writerows(intestazione)
    
    # Scrivi le righe dei dati
    

    # Converte il numero in lettera della colonna corrispondente (es: 1 -> 'A', 2 -> 'B', ...)
    # workbook= Workbook()
    # worksheet= workbook.active
    # # Scrive il numero nella cella corrispondente sulla riga 1
    # worksheet.append(intestazione)
        
    # for riga in b:
    #     for n in range(len(riga)):
    #         #worksheet.append({'A':'valore fibra' , 'B' : x ,'C' : y})
    #         values = (riga[n] for n in range (len(riga)))
    #         for p in range(len(values)):
    #             worksheet.append(p)                 

    # workbook.save(nome)

    # with open(nome, 'w', newline='') as file_csv:
    #     writer = csv.writer(file_csv)

    # # Scrivi l'intestazione
    #     writer.writerow(["Bucket Range", "Selected Elements"])
    
    # # Scrivi le righe dei dati
    #     for i, selected_elements in enumerate(b):
    #         bucket_range = f"{min_confidence + i * bucket_size:.2f} - {min_confidence + (i + 1) * bucket_size:.2f}"
    #         selected_elements_str = ", ".join(str(element) for element in selected_elements)
    #         writer.writerow([bucket_range, selected_elements_str])
    bask_range = []
    bask_elements_list = []
    selected_elements_list = []
    with open(nome, newline='') as csvfile:
    # Legge il file CSV
        csvreader = csv.reader(csvfile)
        next(csvreader)
        # Itera sulle righe del file CSV
        for row in csvreader:
            # Assicurati che la riga abbia almeno due elementi
            if len(row) >= 2:
                # Estrae la parte tra le virgolette
                data_in_quotes = row[1].strip('"')
                
                # Stampa la parte tra le virgolette come una tupla
                if data_in_quotes:
                    # Converte la stringa in una tupla utilizzando eval()
                    data_tuple = eval(data_in_quotes)
                    print(data_tuple)
                
                # Aggiungi i risultati alla lista
                    selected_elements_list.append(data_tuple)

    # Stampa i risultati
    for elements in selected_elements_list:
        print(elements)

    #-------------------------------------------------TORNARE A QUESTO PUNTO NEL CASO-------------------------
    
    # Lista per salvare i risultati dei buckets
    save_buckets_results = []
    save_results=[]
    all_save_results=[]
    #anche questo è parallelizabile
    #for bucket_index, bucket in enumerate(b[9:], start=9):
    # def process_bucket(bucket_index,bucket):
    #     tasso_successo=0
    


    num_threads = 4  # Numero massimo di processi attivi contemporaneamente

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_bucket, bucket_index, bucket_element, x_test_normalized)for bucket_index, bucket_element in enumerate(selected_elements_list)]

            for future in as_completed(futures):
                bucket_index, tasso_successo, save_results = future.result()
                save_buckets_results.append((bucket_index, tasso_successo))
                all_save_results.extend(save_results)

# Esegui la funzione principale
  
    # num_processes = 2
    # with ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     futures = [executor.submit(process_bucket, bucket_index, bucket)for bucket_index, bucket in enumerate(selected_elements_list)]
    
    #     for future in as_completed(futures):
    #         result = future.result()
    #         print(result)
    # Ora 'risultati' contiene l'output di ogni thread
 

    for valori in save_results:
        print(f"Bucket Index: {valori[0]}, element index: {valori[1]}, la previsione e' {valori[2]}, con minima perturbazione: {valori[3]}")

    for result in save_buckets_results:
        print(f"Bucket Index: {result[0]}, Tasso di Successo: {result[1]}")
   
    for valor in all_save_results:#serve solo questo
        print(valor)

    end_time = time.time()

        # # Calcola il tempo trascorso
    execution_time = end_time - start_time

    # print("Tempo di esecuzione:", execution_time, "secondi")
    print( " in minuti sarebbero : ,",execution_time/60)
   
    nome_file1 = 'C:/Users/Dell/Desktop/pyfiles/busketTUTTI.csv'
    intestazione = ["Bucket Index:", "Tasso di Successo:"]
    with open(nome_file1, 'w', newline='') as file_csv:
       writer = csv.writer(file_csv)
       writer.writerows(intestazione)
    
    # Scrivi le righe dei dati
       for riga in save_buckets_results:
        writer.writerow(riga)

  


    nome_file = 'C:/Users/Dell/Desktop/pyfiles/listebusketTUTTI.csv'
    intestazione = ["Bucket Index", "Element index", "Previsione", "Minima perturbazione"]
    with open(nome_file, 'w', newline='') as file_csv:
       writer = csv.writer(file_csv)
       writer.writerows(intestazione)
    
    # Scrivi le righe dei dati
       for riga in all_save_results:
        writer.writerow(riga)

  