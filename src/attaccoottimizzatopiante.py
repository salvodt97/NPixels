import random
import csv
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import regularizers
import matplotlib.pyplot as plt
from keras import backend  as K
import copy
import openpyxl
from openpyxl import Workbook
import pandas as pd
from sklearn.cluster import KMeans
#from load_dataset import load_test_dataset
from load_dataset import load_test_dataset
import LayerforMobilNet as LM
optimizers = {"ResNet50DvsC": 'SGD', "MobilNet": tf.keras.optimizers.Adam(), "ResNet24": tf.keras.optimizers.Adam()}
losses = {"ResNet50DvsC": 'sparse_categorical_crossentropy', "MobilNet": tf.keras.losses.categorical_crossentropy, "ResNet24": tf.keras.losses.categorical_crossentropy}
        #queste liste per ogni rete dicono quali ottimizzatori e quali loss utilizzare 
#------------------------------------------------------SEZIONE ATTAXCCO
def trova_minima_perturbazione(immagine, modificabili, limite_visibilita, loaded_model, massimo_iniziale, tolleranza=1): #1e-5
    left = 0
    find= False
    right = massimo_iniziale
    best_perturbazione = massimo_iniziale
    vector_pixel_mod=modificabili
    contatore_sicurezza=0
    salva_risultato=0
    psnr=psnr_salvato=0
    conta_iterazioni=0
    while right - left > tolleranza:
        contatore_sicurezza+=contatore_sicurezza +1
        # prev=loaded_model.predict(immagine[np.newaxis,...])
        # print("per sicurezza , sto valore deve essere sempre 9 .... ->",np.max(prev))
        #

        mid = (left + right) // 2
        # if contatore_sicurezza==2: mid=255
        mid_normalizzato =mid/255
        #riuscito, salva_immagine, salva_psnr, conta_iterazioni,New_modificabili_short
        valBool,pixel_modificati,psnr,conta_iterazioni,vector_pixel_mod = seleziona_pixel(vector_pixel_mod, immagine, limite_visibilita, loaded_model, mid_normalizzato,psnr,conta_iterazioni )
        if valBool:#qua ci metto midnormalizzto / mid
            right=mid
            if psnr > psnr_salvato:
                best_perturbazione = mid
                salva_risultato=pixel_modificati
                find = True
                right = mid  # dimezza la perturbazione
                psnr_salvato=psnr
        else:
            left = mid 
            if mid >= 191:
                break
             # prendi il valore intermedio
            
     
    return best_perturbazione,vector_pixel_mod,salva_risultato,find,psnr_salvato,conta_iterazioni

def get_mse_psnr(a, b):
    max_ab = float(np.nanmax(np.concatenate((a, b))))
    mse = np.nanmean((np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)) ** 2)
    if mse == 0:
        return 0, 100
    return mse, 20 * np.log10(max_ab / (np.sqrt(mse)))

def calcola_valore(previsione_iniziale, previsione_modificata):
    max_previsione_iniziale = np.max(previsione_iniziale)
    max_previsione_modificata = np.max(previsione_modificata)
    predict_diff = max_previsione_iniziale - max_previsione_modificata #gatto 0,999 ---#gatto 0,993
    indice_maxprevisione_iniziale = np.argmax(previsione_iniziale)
    indice_maxrevisione_modificata = np.argmax(previsione_modificata)
    if(indice_maxprevisione_iniziale !=indice_maxrevisione_modificata):
        predict_diff=100  #può capitare che si trova già una previsione sbagliata ma non controllando l'indice non veniva acchiappato
    return predict_diff
def calcola_peso(immagine_originale, immagine_modificata):
    mse, psnr = get_mse_psnr(immagine_originale, immagine_modificata)
    return psnr

def genera_clusters(coords, n_clusters=80, random_state=0):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(coords)
    return kmeans.labels_

def valuta_clusters(immagine, coords, labels, perturbazioni_range, previsione_iniziale, loaded_model,conta_iterazioni):
    pixel_pesi_valori = []
    for cluster in range(np.max(labels) + 1):
        cluster_coords = coords[labels == cluster] if len(labels) > 0 else []
        best_perturbazione = None
        for delta_r in perturbazioni_range:
            for delta_g in perturbazioni_range:
                for delta_b in perturbazioni_range:
                    conta_iterazioni += 1
                    perturbazione_corrente = np.array([delta_r, delta_g, delta_b])
                    immagine_modificata_temp = np.copy(immagine)

                    # Applica la perturbazione ai pixel nel cluster
                    for coord in cluster_coords:
                        immagine_modificata_temp[coord[0], coord[1]] += perturbazione_corrente
                    
                    # Clip dei valori dell'immagine modificata
                    immagine_modificata_temp = np.clip(immagine_modificata_temp, 0, 1)

                    peso = calcola_peso(immagine, immagine_modificata_temp)
                    previsione_modificata = loaded_model.predict(immagine_modificata_temp[np.newaxis, ...])
                    valore = calcola_valore(previsione_iniziale, previsione_modificata)
                    
                    peso = 100 - peso
                    combined_metric = (valore * 100) / peso
                    if best_perturbazione is None or combined_metric > best_perturbazione:
                        best_perturbazione = combined_metric
                        valori_imp = (peso, valore, perturbazione_corrente,cluster_coords)
        
        peso, valore, perturbazione_corrente,cluster_coords = valori_imp
        pixel_pesi_valori.append((peso, valore, perturbazione_corrente, cluster_coords))
    
    return pixel_pesi_valori,conta_iterazioni


def valuta_clusters2(immagine, coords, perturbazioni_range, previsione_iniziale, loaded_model,conta_iterazioni):
    pixel_pesi_valori = []
    for cluster in coords:
        best_perturbazione = None
        for delta_r in perturbazioni_range:
            for delta_g in perturbazioni_range:
                for delta_b in perturbazioni_range:
                    conta_iterazioni += 1
                    perturbazione_corrente = np.array([delta_r, delta_g, delta_b])
                    immagine_modificata_temp = np.copy(immagine)

                    # Applica la perturbazione ai pixel nel cluster
                    for coord in cluster[0]:
                        immagine_modificata_temp[coord[0], coord[1]] += perturbazione_corrente
                    
                    # Clip dei valori dell'immagine modificata
                    immagine_modificata_temp = np.clip(immagine_modificata_temp, 0, 1)

                    peso = calcola_peso(immagine, immagine_modificata_temp)
                    previsione_modificata = loaded_model.predict(immagine_modificata_temp[np.newaxis, ...])
                    valore = calcola_valore(previsione_iniziale, previsione_modificata)
                    
                    peso = 100 - peso
                   # pixel_pesi_valori.append((peso, valore, perturbazione_corrente, cluster))
                    combined_metric = (valore * 100) / peso
                    if best_perturbazione is None or combined_metric > best_perturbazione:
                        best_perturbazione = combined_metric
                        valori_imp = (peso, valore, perturbazione_corrente,cluster[0])
        
        peso, valore, perturbazione_corrente,cluster_coords = valori_imp
        pixel_pesi_valori.append((peso, valore, perturbazione_corrente, cluster_coords))
    
    return pixel_pesi_valori,conta_iterazioni

#def seleziona_pixel(immagine, modificabili, limite_visibilita, loaded_model, perturbazione, conta_iterazioni, psnr_salvato, max_iter=1, tolleranza=0.01):
  
def seleziona_pixel(modificabili, immagine, limite_visibilita, loaded_model, perturbazione,psnr_salvato,conta_iterazioni, max_iter=1, tolleranza=0.01):
    immagine_modificata = np.copy(immagine)
    previsione_iniziale = loaded_model.predict(immagine[np.newaxis, ...])
    
    perturbazioni_range = perturbazione * np.array([-1, 1])
    print("Perturbazione attuale:", perturbazione * 255, "/255")
    
    prev = loaded_model.predict(immagine_modificata[np.newaxis, ...])
    print("Previsione prima di qualsiasi modifica:", np.max(prev))
   #50176 modificabili=modificabili.reshape(2)
    # Trova le posizioni dei pixel modificabili
    if len(modificabili) > 9:
        coords = np.column_stack(np.where(modificabili == 1)) 
        # Genera i cluster originali una volta sola
        labels_original = genera_clusters(coords) if len(coords) > 0 else np.array([])
        pixel_pesi_valori_original,conta_iterazioni = valuta_clusters(immagine, coords, labels_original, perturbazioni_range, previsione_iniziale, loaded_model,conta_iterazioni) #64 iterazioni = n_cluster * combinazioni check deep rgb (n_cluter x 8) 
    else:
        coords=modificabili     
        pixel_pesi_valori_original,conta_iterazioni = valuta_clusters2(immagine, coords, perturbazioni_range, previsione_iniziale, loaded_model,conta_iterazioni) #64 iterazioni = n_cluster * combinazioni check deep rgb (n_cluter x 8) 
    # Valutazione dei cluster originali
    
    combined_metrics_original = [(peso, valore * 100, perturbazione_corrente, cluster_coords) for peso, valore, perturbazione_corrente, cluster_coords in pixel_pesi_valori_original]
    combined_metrics_original.sort(key=lambda x: x[1] / x[0], reverse=True)

    
    
    #creo sottìinsieme di elementi buoni
    New_modificabili_short = []
    for i in range (9):
      # Variabili per tenere traccia del miglioramento
        migliori_cluster_coords = combined_metrics_original[i][3]
        migliori_perturbazione_corrente = combined_metrics_original[i][2]
        migliori_peso = combined_metrics_original[i][0]
        migliori_valore = combined_metrics_original[i][1]
        New_modificabili_short.append([migliori_cluster_coords,migliori_perturbazione_corrente,migliori_peso,migliori_valore])
    

    #applico le perturbazioni
    immagine_modificata_np = np.copy(immagine_modificata)
    for i in range(len(New_modificabili_short)):
        migliori_cluster_coords,migliori_perturbazione_corrente,migliori_peso,migliori_valore=New_modificabili_short[i]
        for coord in migliori_cluster_coords:
            immagine_modificata_np[coord[0], coord[1]] += migliori_perturbazione_corrente
        immagine_modificata = tf.convert_to_tensor(immagine_modificata_np)
        immagine_modificata = tf.clip_by_value(immagine_modificata, 0, 1)
        peso = calcola_peso(immagine, immagine_modificata)
        temp = loaded_model.predict(immagine_modificata[np.newaxis, ...])
        if np.argmax(previsione_iniziale) != np.argmax(temp):
            riuscito = True
            #print(f"Miglior cluster: {migliori_cluster_coords}, Peso: {peso}, Valore: {migliori_valore}")
            break
        else:
            riuscito = False
        if peso < limite_visibilita:
            break
    
    if riuscito == True:
        a = calcola_peso(immagine, immagine_modificata)
        print("PSNR:", a)
        salva_psnr=a
        salva_immagine=immagine_modificata
        #newarr = immagine_modificata.reshape(1, 224, 224, 3)
        print("Modifica completata")
        return  riuscito, salva_immagine, salva_psnr, conta_iterazioni,New_modificabili_short
    else:
        riuscito=False
        print("Modifica non riuscita")
        return False, [], 0, conta_iterazioni,New_modificabili_short



if __name__ == '__main__':
   
    
    #carico il modello
    model_name = "MobilNet.h5"
   
    loaded_model = tf.keras.models.load_model("C:/Users/Dell/Desktop/partefinalwe/MobilNet.h5")
    loaded_model.summary()
    
    x_test, y_test = load_test_dataset(loaded_model.input_shape, model_name)
    print(x_test.shape[0])#(501, 224, 224, 3)
    print(y_test.shape)#(501, 2)
    casi_di_studio=[]
    contatore_successo=0
    contatore_fallimento=0
    dim_campioni = x_test.shape[0]
    # for i in range (0, dim_campioni,1):
    #     print(i)
    #     campione=x_test[i]
    #     # plt.imshow(campione)
    #     # plt.show()
    #     # plt.close()


    array_list = []
    risultati=[]
    dim_campioni = x_test.shape[0]
    for i in range(15,30,1):#dim_campioni
        numeroscelto=i    
        input_test = x_test[numeroscelto]
        dX=input_test.shape[0]
        dY=input_test.shape[1]
        dZ=input_test.shape[2]
        newarr = input_test.reshape(1,dX,dY,dZ)    
        #carico il dataset cifar10 

        #IN QUESTO CASO FORSE DEVO RIVEDERE IL CLUSTERING , A ME PARE LO FACCIA AL LIVELLO SBAGLITO, HO PROVATO A CORREGGERE PERò POI MI ESCE SFASATO SULL'IMMAGINE ORIGINALE

        #----------------------------------input non modificiato---------------------
        inputunchanged= copy.deepcopy(input_test)

        previsione1 = loaded_model.predict(newarr) 
     #   plt.figure()

        print ("print input: \n",previsione1)
        if (np.argmax(y_test[numeroscelto]) != np.argmax(previsione1)):
            print("previsione sbagliata")
            continue  
        else:
            print("previsione corretta")
        # plt.imshow(input_test)
        # plt.show()    

            
        image_label = np.argmax(previsione1)
        test_acc =np.max(previsione1)
        image_label_arr = []
        image_label_arr.append(image_label)
        image_label_arr = np.array(image_label_arr)
        classes = ["Healthy","multiple_diseases", "rust", "scab"]
        image_label_arr = tf.keras.utils.to_categorical(image_label_arr, len(classes))
        inputunchanged= copy.deepcopy(input_test)
        predicted_label=image_label

        plt.imshow(input_test)  # To change [-1, 1] to [0,1]
        image_class = classes[image_label] 
        plt.title('{} : {:.2f}% Confidence'.format(image_class, test_acc*100))
        #plt.show()
        
        print(classes[image_label] + " : " + str(image_label))
        print(image_label_arr)
        plt.savefig("C:/Users/Dell/Desktop/Piante/new/Input{}.png".format(numeroscelto))
        plt.close()

    
        dim_X = input_test.shape[0]
        dim_y = input_test.shape[1]
        last_layer=8
        #ottenimento della maschera, e del vettore importante
        mask= LM.back_from_most_important(loaded_model,numeroscelto,x_test,last_layer)
        data = LM.give_foglio ( loaded_model,numeroscelto,x_test,last_layer) #ricordo che sta funzione mi permette di ottenere i risultati del primo layer di convoluzione
        vector_importante = LM.clustering(data,mask,numeroscelto)
        pixel_vector=vector_importante
        pixel_indices = pixel_vector.reshape(-1, 2)

    # Ordina gli elementi in base alla coordinata x (prima colonna) e poi y (seconda colonna)
        indici_ordinati = np.lexsort((pixel_indices[:, 1], pixel_indices[:, 0]))
    # Riordina gli elementi in base agli indici ottenuti
        elementi_ordinati = pixel_indices[indici_ordinati]
        if elementi_ordinati.size == 0:
            continue
        rivettore = elementi_ordinati.reshape(-1,len(pixel_vector))
     
        mask = LM.create_mask2(x_test,loaded_model,numeroscelto,pixel_indices,last_layer)  # Valori casuali 0 o 1
        #--------------------------------------------------------------------------------------------------------------#----------------------------------------------------------------------------------------------------------------------------------
        matrix = mask
        # Configura la figura e gli assi
        fig, ax = plt.subplots()
        # Usa imshow per visualizzare la matrice
        # cmap='gray' usa una scala di grigi (0=bianco, 1=nero)
        cax = ax.imshow(matrix, cmap='gray_r', interpolation='none')
        # Rimuove gli assi
        ax.axis('off')
        center_row = (matrix.shape[0] - 224) // 2
        center_col = (matrix.shape[1] - 224) // 2
        matrix_224x224 = matrix[center_row:center_row+224, center_col:center_col+224]
        matrix_expanded = np.stack((matrix, matrix, matrix), axis=-1) 
       # matrix_expanded = np.stack((matrix, matrix, matrix), axis=-1) 
        new_im= matrix_expanded +input_test
        # Mostra l'immagine
        plt.imshow(new_im)  
        plt.savefig("C:/Users/Dell/Desktop/Piante/new/inp{}.png".format(numeroscelto))
        #plt.show()
        plt.close()  
  
    #     #---------------------------------------------------------------------------------------------------------------------------------#-----------------------------------------------------------------------------------------------------------------------------------
        pixel_modificabili = np.argwhere(mask == 1)
        contatore_successo=0
        contatore_fallimento=0


    # Definisci loaded_model secondo il tuo modello preaddestrato
        immagine = input_test # Esempio di immagine 32x32x3
        modificabili = pixel_modificabili  # Lista dei pixel modificabili  -> ci vorrà dalla mashcera ottenere sto vettore di tuple
    #    print (modificabili)    si blocca con l'ultimo pixel di merda
        limite_visibilita = 20
        massimo_iniziale = 255
        previsione = 0
        incremento_visibilita = 5
        
        find=False
        best_minima_perturbazione = best_vettore = best_immagine_mod = immagine_mod = psnr = best_psnr= 0
        find_salvato=False
        #potrei metterci anche il valore del psnr, salvo i risultati solamente se il psnr è migliore, e riutilizzo sempre il miglior vettore, se il miglior vettore a perturbazioni basse è vuoto, allora ellimino l'attacco
         #best_perturbazione,vector_pixel_mod,salva_risultato,find,psnr_salvato,conta_iterazioni
       # while True:
        minima_perturbazione, vettore, immagine_mod, find, psnr,conta_iterazioni = trova_minima_perturbazione(immagine, mask, limite_visibilita, loaded_model, massimo_iniziale)
                
        if find:# if best_psnr(-1) > limite_visibilità, incremento il limite_visibilità
            if psnr>=best_psnr:
                best_minima_perturbazione = minima_perturbazione
                best_vettore = vettore
                best_immagine_mod = immagine_mod
                found = True
                limite_visibilita += incremento_visibilita  #potrei metterci anche il valore del psnr, salvo i risultati solamente se il psnr è migliore, e riutilizzo sempre il miglior vettore, se il miglior vettore a perturbazioni basse è vuoto, allora ellimino l'attacco
                modificabili = vettore
                best_psnr=psnr
                find_salvato=True
                # if best_psnr > limite_visibilita: #in pratica se ottengo un risultato ottimo , del tipo limite_visibilità = 40, ma ottengo un attacco con psnr=50, chiaramente è inutile che cerco di ripetere l'attacco per i valori <50 , ma lo alzo 
                #     limite_visibilita = best_psnr+5
        #         else:
            #             break
            # else:
            #     break




    # # Stampa del miglior pixel selezionato 

        print("Miglior pertrurbazione selezionata:", best_minima_perturbazione)

        
        
        if find_salvato==True:
        
            pr = loaded_model.predict(best_immagine_mod[np.newaxis,...]) 
            if (np.argmax(y_test[numeroscelto]) != np.argmax(pr)):          
                print("la label e'", np.argmax(pr))
            else:
                print("fallito")      
                
            psnr=calcola_peso(input_test,best_immagine_mod)
            print("siet ca",psnr)
            print("il psnr e' ", psnr)
            #attacco(modificabili, immagine, limite_visibilita, model, 125 )
        # Stampa del miglior pixel selezionato
            print("Miglior pertrurbazione selezionata:", best_minima_perturbazione)
            psnr = calcola_peso(input_test, best_immagine_mod)
            print("PSNR:", psnr)
            #newarr = immagine_modificata.reshape(1, 224, 224, 3)
            print("Modifica completata, salvo l'immagine modificata")

                #all'inizio for i in tutti i campioni che esistono  
            numero_elementi = numeroscelto#pari al numero di predizinoi corrette
            nome_array =  numero_elementi  #identificativo immagine
            elementi_modificati = best_immagine_mod # elementi che vengono modificati 
            salva_label=np.argmax(pr)
            
            
            
            array_list.append([nome_array, best_minima_perturbazione,psnr,salva_label,conta_iterazioni,predicted_label])#best_psnr
    #         #

            image_label_atk = np.argmax(pr)
            test_acc =np.max(pr)
            image_label_arr = []
            image_label_arr.append(image_label_atk)
            image_label_arr = np.array(image_label_arr)
            image_label_arr = tf.keras.utils.to_categorical(image_label_arr, len(classes))
            inputunchanged= copy.deepcopy(input_test)
            predicted_label=image_label

            plt.imshow(best_immagine_mod)  # To change [-1, 1] to [0,1]
            image_class = classes[image_label_atk] 
            plt.title('{} : {:.2f}% Confidence, psnr{}'.format(image_class, test_acc*100,psnr))
            print(classes[image_label] + " : " + str(image_label_atk))
            print(image_label_arr)
            #plt.show()
            plt.savefig("C:/Users/Dell/Desktop/Piante/new/immaginemodificata{}.png".format(numeroscelto))
            
            print(classes[image_label] + " : " + str(image_label))
            print(image_label_arr)
            plt.close()
        else:
            contatore_fallimento=contatore_fallimento+1
            riuscito=False
            print("Modifica non riuscita")


    data_dict = {'ID': [], 'Minima perturbazionee': [], 'PSNR': [], 'Indice Predict' : [],'iterazioni richieste': [],'Predicted Label':[]}
    
    #for i, bucket in enumerate(buckets):
    # Riempimento del dizionario con i dati dai buckets   for tutti i risultati
    #for array in enumerate(risultati):
    for index, element in enumerate(array_list):
        
        data_dict['ID'].append(element[0])
        data_dict['Minima perturbazionee'].append(element[1])
        data_dict['PSNR'].append(element[2])
        data_dict['Indice Predict'].append(element[3])
        data_dict['iterazioni richieste'].append(element[4])
        data_dict['Predicted Label'].append(element[5])

    # Creazione di un DataFrame da dictionary
    df = pd.DataFrame(data_dict)

    # Definizione del nome del file Excel di output

    nome_file_excel='C:/Users/Dell/Desktop/Piante/new/Campioni0-30.xlsx'
    # Salvataggio del DataFrame in un file Excel
    df.to_excel(nome_file_excel, index=False)

    print(f"Il file Excel '{nome_file_excel}' è stato creato con successo.")
    print("il numero di fallimenti e' pari a :",contatore_fallimento)
