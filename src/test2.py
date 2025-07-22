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

def adv_sample(test):
    p =  np.random.randint(low =0 , high = 250, size= (32,32,3) )
    new_x_test = test + p
    new_x_test %= 255
    newarr2 = new_x_test.reshape(1,32,32,3)
    return newarr2

def adv_sample2(patate):
    test = patate
    print("modifico la matrice 3x3 in poiszione [27,29][27,29]")
    for n in range (26,29):  #(start,stop,step)--> quandi parto a 26 e finisco a 28, il terminatore non viene mai conteggiato 
        for m in range(26,29) :
            test[m][n][0]=test[m][n][1]=test[m][n][2]= 0
    
    #ho notato che se invece del limite dex ci metto 30 invece che 29, no nsbaglia più lla predizione   -26,29 è il minimo però se già metto 26-28 cambia la previsione, se metto 25-28 non sbaglia
    test_adv = test.reshape(1,32,32,3)
    return test_adv
#quindi il range migliore che porta SEMPRE ad un risultato diverso va da [26,28] (26,29), non sempre a quanto pare, per alcune immagini non funziona
#se cambio in (27,30) cambia l'ooutput del singolo layer ma non modifica la previsione
#da 28 in poi fa schifo
#in realtà, con un modello allenato bene perde sempre sto alg.


def adv_sample3(cozze,riga,colonna):
    test = cozze
   
    test[riga][colonna][0]=test[riga][colonna][1]=test[riga][colonna][2]= 0
    
    #ho notato che se invece del limite dex ci metto 30 invece che 29, no nsbaglia più lla predizione   -26,29 è il minimo però se già metto 26-28 cambia la previsione, se metto 25-28 non sbaglia
    
    return test



if __name__ == '__main__':
    # Loading CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
   # print("printo i valori che si trovano nel punto di interesse :",x_test[10][26][26][0],x_test[10][26][26][1],x_test[10][26][26][2])
    #single_img_reshaped = np.transpose(np.reshape(x_test[3],(3, 32,32)), (1,2,0))
    # plt.imshow(x_test[3])#stampo l'immagine buona
    # plt.show()

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
    
#     minnet = keras.Sequential()
# #    minnet.add(layers.InputLayer(input_shape=(32,32,3), name = 'inp'))
#     minnet.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3), name="FullyConnected1" ))
# #https://stackoverflow.com/questions/43164728/visualization-of-keras-convolution-layer-outputs
   
# # #  trovare il metodo per stampare gli output da questo layer

#     minnet.add(layers.MaxPooling2D(pool_size=(2, 2)))
#     minnet.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
#     minnet.add(layers.MaxPooling2D(pool_size=(2, 2)))
#     minnet.add(layers.Flatten())
#     minnet.add(layers.Dense(64, activation='relu'))
#     minnet.add(layers.Dense(10, activation='softmax'))
    
#     #Compile the model
#     minnet.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
#     minnet.summary()
#      #la fi fa il training del modello
#     history = minnet.fit(datagen.flow(x_train_normalized, y_train_normalized, batch_size=64), steps_per_epoch=x_train_normalized.shape[0] // 64, epochs=200, validation_data=(x_test_normalized, y_test_normalized))
#     test_loss, test_acc = minnet.evaluate(x_test_normalized, y_test_normalized, verbose=2)
#     minnet.save("C:/Users/Dell/Min8.h5")



    # with tf.Session() as sess:
    #     FC1 = tf.get_default_graph().get_tensor_by_name('fullyconnect1Lv:0')
    #     FC1_values = sess.run(FC1, feed_dict={x: x_test})



                    #----------------------caricare il modello --------------------

    loaded_model = keras.models.load_model("C:/Users/Dell/Min8.h5")
    loaded_model.summary()
    loaded_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    

#     K.result_type

    #--------------------------SHOW OUTPUT FIRST LAYER------------------------------
   
   #https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer?rq=3
   # func = K.function([loaded_model.get_layer('input').input], loaded_model.get_layer('conv').output) such layer: input. Existing layers are: ['FullyConnected1', 'max_pooling2d', 'conv2d', 'max_pooling2d_1', 'flatten', 'dense', 'dense_1']. 
    # get_1st_layer_output =K.function([minnet.layers[0].input],[minnet.layers[1].output])
    # layer_output = get_1st_layer_output(newarr)[0]
    # print(layer_output)
             

    # for layer in loaded_model.layers: #mi stampo tutti gli "shape" del modello caricato
    #     print(layer.output_shape)
        #metodo per ottenere l'output del primo layer

    # print("stampo l'input al primo layer")
    # print(loaded_model.layers[0].input)
    # print("stampo l'outotpit del primo layer")
    # print(loaded_model.layers[0].output)
    #sta print mi da la descrizione del layer, in corrispettiva del valore nelle[], per il layer[0] in input ho il layer di input , in output mi darà il convolutionale
    #layer[0]= è il layer di input
    #layer[1]= è il fully connected 1 ...eccecc


    #----------------------------------input non modificiato---------------------
    #x_test[7] è immune
    #FARE UN CICLO E TESTARE TUTTI I CASI , VED
    #input_test = copy.deepcopy(x_test[700])#in qualche modo dovrebbe funzionare così ma non funziona
    contatore = 0
    for cicli in range (10000):
        input_test = x_test[cicli]
        newarr = input_test.reshape(1,32,32,3)                
        print("\n\n\n\n")
        #sta backend.function si prende il layer di input e output e calcola quelle che sono le elaborazioni nel nodo specificato, se provo a farlo di un layer di "mezzo", devo usare un dato elaborato già dai layer precedenti, posso usare come barbatrucco di usare come valore di input quello del primo layer [0] e chiedere l'output di qualsiasi altro layer , se metto[-1]che srebbe l'ultimo mi fa praticamente la predizione
        get_1st_layer_output =K.function([loaded_model.layers[0].input],[loaded_model.layers[0].output])
        layer_output = get_1st_layer_output(newarr)[0]# non so perché ci vuole sto zero
    #    print("printo l'output del K.function \n" , layer_output) 
        
        print("printo la dim dell'output del k-function : ", layer_output.shape)#mo voglio capire da sto tensore (1,15,15,32) quali elementi sono i pià grandi
        # for ( i = 0 ; i< 7 ; i ++) for *elemento che incrementa* in *condizione*
        #il mio tensore sarà fatto da : 1= non so cosa , 15 matrici , 15 righe , 32 colonne
        #la print mi stampa solo le prime due e le ultime due matrici, forse se lo stampo in un file me le fa vede tutte
        
        # plt.imshow(input_test)
        # plt.show()
        previsione1 = loaded_model.predict(newarr) 
        print ("print input: \n",previsione1)
    #ad una certa si bugga
    #----------------METODO PER PRINTARE L'OUTPUT DAL PRIMO LAYER CON VALORE MAGGIORE--------
        # BELLO E FUNZIONNANTE MA MI TROVA SOLO UN ELEMENTO 
        # max = 0
        # n = x = y = 0
        # print( "print fantasionsa perché non s capisc", layer_output[0,7,29,4]) # NOTA: i primi risusltati sono falsati poiché sbagliavo la stampa dei punti .-. inoltre il "nodo" di max non è sempre lo stesso, ma cambia in base all'immagine
        # for n in range(30):                                     
        #      for x in range(30):
        #           for y in range(32):
        #             if (max < layer_output[0,n,x,y]):
        #                 max=layer_output[0,n,x,y]  
        #                 saveN= n  
        #                 saveX= x
        #                 savey= y

        # print("il valore più altro trovato e' ",max,"in posizione ",saveN , saveX, savey)
        # print("da qua il sec tentativ ------ \n")
        
        
        #METODO PER TROVARE PIù ELEMENTI DI MAX
        
        max_attuale = 0 
        max_massimo = 0
        a = b = c = d = e = pos_min=  0

    
        vett_max = np.zeros(9, dtype=float)
        vet_pos_max= np.zeros(27, dtype=int)
        min = 0 
        
        vett_max[0] = layer_output[0,0,0,0]
        for a in range (30):
            for b in range (30):
                for c in range (32):
                    for d in range (9):
                        for e in range(9):
                            if(vett_max[d] >= vett_max[e] and d!=e ):
                                min = vett_max[e]
                                pos_min = e
                                break  
                    if( layer_output[0,a,b,c] > min ):
                        vett_max[pos_min] = layer_output[0,a,b,c]
                        patate = pos_min * 3
                        vet_pos_max[patate +0]=a
                        vet_pos_max[patate +1]=b
                        vet_pos_max[patate +2]=c
        # for k in range (9):
        #     print("print degli elmenti massimi trovati:", vett_max[k])
        
        # print("print delle posizioni degli elmenti massimi trovati:", end=' ')
        # for f in range (27):
        #     print( vet_pos_max[f], end=' ')
        # NOTA: molti output (almeno pe ril campione 250 )  sono del filtro 4 ---_> (mi interessa sta cosa?????????????????)

    # vett_max [50 40 50 20 0 0 0]     min = 40 pos_min = 1 
    #                          60         pos(c)=50 c = 0 d++  forse sto metodo è migliore del primo

    # 1 2 3 , 2 4 6 , 3 6 9     
    #idea, salvare l'elemento chiamto min e cambiare quello, non il massimo.
    #devo modificare ora tutti sti elementi e schiattarli o a 0 o al massimo ( i pixel dell'immagine in ing),in pratica passo le posizioni e modificio gli elementi , senza considerare però la posizine "c" che è dettata dal "filtro"
        #--------------------------------------capiamo se riusciamo a modificare l'output-.--------------

        max = 0
        n = x = y = 0
        for f in range(0,27,3):
            newarr2 = adv_sample3(input_test,vet_pos_max[f],vet_pos_max[f+1]) #in realtà l'elmento in posizione +2 non mi serve perché è il filtro
        test_adv = newarr2.reshape(1,32,32,3)
        get_1st_layer_output =K.function([loaded_model.layers[0].input],[loaded_model.layers[0].output])
        layer_output2 = get_1st_layer_output(test_adv)[0]# non so perché ci vuole sto zero
        #print(layer_output2) 
        # for n in range(30):
        #      for x in range(30):
        #           for y in range(32):
        #             if (max < layer_output2[0,n,x,y]):
        #                 max=layer_output2[0,n,x,y]  
        #                 saveN= n  
        #                 saveX= x
        #                 savey= y

        # print("il valore più altro trovato e' ",max,"in posizione ",n , x, y)
        # max_attuale = 0 
        # max_massimo = 0
        # a = b = c = d = e = pos_min=  0

    
        # vett_max = np.zeros(9, dtype=float)
        # vet_pos_max= np.zeros(27, dtype=int)
        # min = 0 
        
        # vett_max[0] = layer_output[0,0,0,0]
        # for a in range (30):
        #     for b in range (30):
        #         for c in range (32):
        #             for d in range (9):
        #                 for e in range(9):
        #                     if(vett_max[d] >= vett_max[e] and d!=e ):
        #                         min = vett_max[e]
        #                         pos_min = e
        #                         break  
        #             if( layer_output2[0,a,b,c] > min ):
        #                 vett_max[pos_min] = layer_output[0,a,b,c]
        #                 patate = pos_min * 3
        #                 vet_pos_max[patate +0]=a
        #                 vet_pos_max[patate +1]=b
        #                 vet_pos_max[patate +2]=c
        # for k in range (9):
        #     print("print degli elmenti massimi trovati:", vett_max[k])
        # for f in range (27):
        #     print("print delle posizioni degli elmenti massimi trovati:", vet_pos_max[f])
        
        
        #------------------------------metodo per stamparmi + immagini vicine----------------------------
        
        newarr2_img= test_adv.reshape(32,32,3)
        

        # fig,(ax1, ax2) = plt.subplots(nrows=1,ncols=2)
        # ax1.imshow(input_test)   #per oscuri motivi l'immagine di partenza , del set si modifica
        # ax2.imshow(newarr2_img)
        # ax1.set_title("immagine normale")
        # ax1.set_xlabel('boh') 
        # ax2.set_title("immgine cambiata")
        # ax2.set_xlabel('boh')
        # plt.imshow(newarr2_img)
        # plt.show()
        # # plt.imshow(x_test[10])
        # # plt.show()   non capisco perché si cambia propriox_test[10]  e non la variabile d'appoggio solamente
        inp_adv_mod = newarr2.reshape(1,32,32,3) 
        yprevisione2 = loaded_model.predict(inp_adv_mod)  
        print(yprevisione2)
       

        for n in range (10):
            if(previsione1[0][n] != yprevisione2[0][n]):
                contatore = contatore +1
                break

          

    tasso_successo = 10000 / contatore  
    print("il tasso di successo e' :",tasso_successo,"%")
    # # intermediate_layer_model = minnet(inputs=minnet.input, outputs=minnet.get_layer('FullyConnected1').output)
    # # intermediate_output = intermediate_layer_model.predict(newarr)
    # # print(intermediate_output)

    # #------------------------------------TEST PER CAMBIARE L'INPUT------------------------------- fase1
    # print("-------------------------cambio input check output-----------")
  
    
    # y_adv = loaded_model.predict(newarr2) 
    # print ("print input adv: \n",y_adv)

    # newarr22 = newarr2.reshape(32,32,3)
    # plt.imshow(newarr22)#stampo l'immagine buona
    # plt.show()

    # while np.array_equal(y,y2):
        
    #     newarr2=adv_sample(x_test[3])
    #     y2 = loaded_model.predict(newarr2) 
       
    # newarr3 = newarr2.reshape(32,32,3)
    # print (y2)
    # plt.imshow(newarr3)#stampo l'immagine buona
    # plt.show()