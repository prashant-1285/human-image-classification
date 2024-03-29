import cv2
import numpy as np
import os
import pickle
from keras.preprocessing.image import ImageDataGenerator
#from keras.layers import Dense,Flatten,Conv2D,Activation,Dropout
from keras import backend as K
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D,Dropout
import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import MaxPool2D
from keras.initializers import glorot_uniform
import config as cfg
import argparse
#import image
from keras.applications.resnet50 import ResNet50
K.tensorflow_backend._get_available_gpus()
from IPython.display import clear_output
from matplotlib import pyplot as plt


class TrainingPlot(keras.callbacks.Callback):
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            print(N)
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            print("accyracies",self.losses,self.val_losses)
            plt.plot(N, self.losses, label = "train_loss")
            
            plt.plot(N, self.val_losses, label = "val_loss")
            
            plt.title("Training Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig("/content/drive/MyDrive/Resnet/loss.jpg")
            
            plt.figure()
           
            plt.plot(N, self.acc, label = "train_acc")
            print("accyracies",self.acc,self.val_acc)
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig("/content/drive/MyDrive/Resnet/accuracy.jpg")

plot = TrainingPlot()


def main(args):
    #Parsing of train and test path
    train_path = os.path.join(args.input_dir,"train")
    validation_path=os.path.join(args.input_dir,"val")
    output_weight=args.output_weight
    

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
    model=ResNet50(weights='imagenet',input_shape=(224,224,3),include_top=False)


    
    #Vgg16 = Model(inputs=model.input, outputs=model.get_layer('vgg16').output)

    #Make all the layers except final layer non trainable
    for layer in model.layers:
        layer.trainable = False
        
    x = model.output
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(2, activation= 'softmax')(x)
    model = Model(inputs = model.input, outputs = predictions)
    
    #Get the summary of the trainable and non trainable layers
    for layer in model.layers:
        print(layer, layer.trainable)
       

    opt = SGD(lr=cfg.lr, momentum=cfg.momentum)
    model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

    # This saves the model
    mc = ModelCheckpoint(os.path.join(output_weight,'best_model.h5'), monitor='val_accuracy', mode='max', save_best_only=True)

    #train the model
    h=model.fit_generator(train_generator,validation_data=test_generator,epochs=cfg.num_iters,verbose=1,callbacks=[mc,plot])


if __name__ == '__main__':
    # if GPU available then it trains on GPU 0
    #os.environ["CUDA_VISIBLE_DEVICES"]="1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='path to the directory where the train and validation images will be read from')
    parser.add_argument('--output_weight', type=str,
                        help='path to the directory where the weights will be saved to')

    args = parser.parse_args()
    main(args)
