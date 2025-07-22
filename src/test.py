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
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#come tolgo il commento non carica più il modello .-.
#se metto il commento lo carica , non capisco il problema 
from keras import backend  as K

def adv_sample(test):
    p =  np.random.randint(low =0 , high = 250, size= (32,32,3) )
    new_x_test = test + p
    new_x_test %= 255
    newarr2 = new_x_test.reshape(1,32,32,3)
    return newarr2

def adv_sample2(test):
    
    test[31][31][0]= test[31][31][1]=test[31][31][2] = 0 
    test[30][30][0]= test[30][30][1]=test[30][30][2] = 0 
    test[29][29][0]= test[29][29][1]=test[29][29][2] = 0 
    newarr2 = test.reshape(1,32,32,3)
    return newarr2

if __name__ == '__main__':
    # Loading CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    
    #single_img_reshaped = np.transpose(np.reshape(x_test[3],(3, 32,32)), (1,2,0))
    # plt.imshow(x_test[3])#stampo l'immagine buona
    # plt.show()

    #print(x_test.shape)#stampa (10000,32,32,3) ovvero 10000 elementi di (32,32,3)
    print(x_test[0])
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
#     history = minnet.fit(datagen.flow(x_train_normalized, y_train_normalized, batch_size=64), steps_per_epoch=x_train_normalized.shape[0] // 64, epochs=10, validation_data=(x_test_normalized, y_test_normalized))
#     test_loss, test_acc = minnet.evaluate(x_test_normalized, y_test_normalized, verbose=2)
#     minnet.save("C:/Users/Dell/Min7.h5")



    # with tf.Session() as sess:
    #     FC1 = tf.get_default_graph().get_tensor_by_name('fullyconnect1Lv:0')
    #     FC1_values = sess.run(FC1, feed_dict={x: x_test})



                    #----------------------caricare il modello --------------------

    loaded_model = keras.models.load_model("C:/Users/Dell/Min7.h5")
    loaded_model.summary()
    loaded_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    
#     K.result_type

    #--------------------------SHOW OUTPUT FIRST LAYER------------------------------
    # x = tf.keras.Input(tf.float32, shape=[1, 32, 32, 3], name='x')
    # with tf.Session() as sess:
    #     FC1 = tf.get_default_graph().get_tensor_by_name('FullyConnected1:0')
    #     FC1_values = sess.run(FC1, feed_dict={x: x_test[3]}) 

    # def get_layer_outputs(image):
    #     '''This function extracts the numerical output of each layer.'''
    #     outputs    = [layer.output for layer in loaded_model.layers]
    #     comp_graph = [K.function([loaded_model.input] + [K.learning_phase()], [output]) for output in outputs]
   
   #https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer?rq=3
   # func = K.function([loaded_model.get_layer('input').input], loaded_model.get_layer('conv').output) such layer: input. Existing layers are: ['FullyConnected1', 'max_pooling2d', 'conv2d', 'max_pooling2d_1', 'flatten', 'dense', 'dense_1'].
    newarr = x_test[3].reshape(1,32,32,3)                 
    # get_1st_layer_output =K.function([minnet.layers[0].input],[minnet.layers[1].output])
    # layer_output = get_1st_layer_output(newarr)[0]
    # print(layer_output)

    for layer in loaded_model.layers: #mi stampo tutti gli "shape" del modello caricato
        print(layer.output_shape)
        #metodo per ottenere l'output del primo layer

    print("stampo l'input al primo layer")
    print(loaded_model.layers[0].output)
    #sta print mi da la descrizione del layer, in corrispettiva del valore nelle[], per il layer[0] in input ho il layer di input , in output mi darà il convolutionale
    #layer[0]= è il layer di input
    #layer[1]= è il fully connected 1 ...eccecc

    print("\n\n\n\n")
    #sta backend.function si prende il layer di input e output e calcola quelle che sono le elaborazioni nel nodo specificato, se provo a farlo di un layer di "mezzo", devo usare un dato elaborato già dai layer precedenti, posso usare come barbatrucco di usare come valore di input quello del primo layer [0] e chiedere l'output di qualsiasi altro layer , se metto[-1]che srebbe l'ultimo mi fa praticamente la predizione
    get_1st_layer_output =K.function([loaded_model.layers[0].input],[loaded_model.layers[0].output])
    layer_output = get_1st_layer_output(newarr)[0]# non so perché ci vuole sto zero
    print(layer_output) 
    
    print(layer_output.shape)#mo voglio capire da sto tensore (1,15,15,32) quali elementi sono i pià grandi
    # for ( i = 0 ; i< 7 ; i ++) for *elemento che incrementa* in *condizione*
    #il mio tensore sarà fatto da : 1= non so cosa , 15 matrici , 15 righe , 32 colonne
    #la print mi stampa solo le prime due e le ultime due matrici, forse se lo stampo in un file me le fa vede tutte
    
    
    

   
   #----------------METODO PER PRINTARE L'OUTPUT DAL PRIMO LAYER CON VALORE MAGGIORE--------
   
    max = 0
    n = x = y = 0

    for n in range(30):
         for x in range(30):
              for y in range(32):
                if (max < layer_output[0,n,x,y]):
                    max=layer_output[0,n,x,y]  
                    saveN= n  
                    saveX= x
                    savey= y

    print("il valore più altro trovato e' ",max,"in posizione ",n , x, y)


    max = 0
    n = x = y = 0
    #--------------------------------------capiamo se riusciamo a modificare l'output-.--------------
    newarr2 = adv_sample2(x_test[3])
    get_1st_layer_output =K.function([loaded_model.layers[0].input],[loaded_model.layers[0].output])
    layer_output2 = get_1st_layer_output(newarr2)[0]# non so perché ci vuole sto zero
    #print(layer_output2) 
    for n in range(30):
         for x in range(30):
              for y in range(32):
                if (max < layer_output2[0,n,x,y]):
                    max=layer_output2[0,n,x,y]  
                    saveN= n  
                    saveX= x
                    savey= y

    print("il valore più altro trovato e' ",max,"in posizione ",n , x, y)
    # y= loaded_model.layers[0].predict(newarr) non funziona
    # print(y)

    # intermediate_layer_model = minnet(inputs=minnet.input, outputs=minnet.get_layer('FullyConnected1').output)
    # intermediate_output = intermediate_layer_model.predict(newarr)
    # print(intermediate_output)

    #------------------------------------TEST PER CAMBIARE L'INPUT------------------------------- fase1
    print("-------------------------cambio input check output-----------")
    newarr = x_test[3].reshape(1,32,32,3)
    y = loaded_model.predict(newarr) 
    print (y)
    newarr2=adv_sample2(x_test[3])
    y2 = loaded_model.predict(newarr2) 
    print (y2)

    # while np.array_equal(y,y2):
        
    #     newarr2=adv_sample(x_test[3])
    #     y2 = loaded_model.predict(newarr2) 
       
    # newarr3 = newarr2.reshape(32,32,3)
    # print (y2)
    # plt.imshow(newarr3)#stampo l'immagine buona
    # plt.show()