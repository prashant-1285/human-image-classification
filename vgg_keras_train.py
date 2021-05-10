import cv2
import numpy as np
import os
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten,Conv2D,Activation,Dropout
from keras import backend as K
import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import MaxPool2D
import config as cfg
import argparse
import image
K.tensorflow_backend._get_available_gpus()




def VGG16():
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name='vgg16'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(256, activation='relu', name='fc1'))
    model.add(Dense(128, activation='relu', name='fc2'))
    model.add(Dense(int(cfg.num_classes), activation='softmax', name='output'))
    return model

def main(args):
    #Parsing of train and test path
    train_path = os.path.join(args.input_dir,"train")
    validation_path=os.path.join(args.input_dir,"val")
    output_weight=args.output_weight
    vgg_weight=args.vgg_weight

    #Provide training path
    class_names_train=os.listdir(train_path)
    class_names_val=os.listdir(validation_path)
    print('train data class names',class_names_train)
    print('validation data class names',class_names_val)
    train_datagen = ImageDataGenerator(zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15)
    validation_datagen = ImageDataGenerator()

    #Train and test Generator
    train_generator = train_datagen.flow_from_directory(train_path,target_size=(cfg.image_height, cfg.image_height),batch_size=cfg.train_batch_size,shuffle=True,class_mode='categorical')
    test_generator = validation_datagen.flow_from_directory(validation_path,target_size=(cfg.image_height,cfg.image_height),batch_size=cfg.val_batch_size,shuffle=False,class_mode='categorical')


    #create a model
    model=VGG16()
    Vgg16 = Model(inputs=model.input, outputs=model.get_layer('vgg16').output)


    if not os.path.isfile(os.path.join(output_weight,'best_model.h5')):
        print('Starting training from scratch')
        #load weights will load model in another model
        Vgg16.load_weights(os.path.join(vgg_weight,"vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"))

    else:
        # If this is a continued training, load the trained model from before
        print('Continue training based on previous trained model')
        print('Loading weights from {}'.format(os.path.join(output_weight,'best_model.h5')))
        model=load_model(os.path.join(output_weight,'best_model.h5'))

    #Make all the layers except final layer non trainable
    for layer in Vgg16.layers:
        layer.trainable = False

    #Get the summary of the trainable and non trainable layers
    for layer in model.layers:
        print(layer, layer.trainable)


    opt = SGD(lr=cfg.lr, momentum=cfg.momentum)
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

    # This saves the model
    mc = ModelCheckpoint(os.path.join(output_weight,'best_model.h5'), monitor='val_accuracy', mode='max', save_best_only=True)

    #train the model
    model.fit_generator(train_generator,validation_data=test_generator,epochs=cfg.num_iters,verbose=1,callbacks=[mc])
 
    

if __name__ == '__main__':
    # if GPU available then it trains on GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='path to the directory where the train and validation images will be read from')
    parser.add_argument('--output_weight', type=str,
                        help='path to the directory where the weights will be saved to')
    parser.add_argument('--vgg_weight', type=str,
                        help='path to the directory where initial vgg16 weights is stored')
    args = parser.parse_args()
    main(args)
