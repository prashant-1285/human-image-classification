
import cv2
import numpy as np
import os

from keras.layers import Dense,Flatten,Conv2D,Activation,Dropout
from keras import backend as K
import keras
from keras.models import Sequential, Model
from sklearn import metrics
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import MaxPool2D
import config as cfg
import argparse
#import image
K.tensorflow_backend._get_available_gpus()
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
def main(args):
    #Provide path
    test_dir=args.test_dir
    output_weight=args.output_weight
    print("outputweight is",output_weight)
    image_dir=args.image_dir
    #Test data generator 
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(test_dir,target_size=(cfg.image_height,cfg.image_height),batch_size=cfg.val_batch_size,shuffle=False,class_mode='categorical')
    model_test=load_model(os.path.join(output_weight,"best_model.h5"))
    label_map=test_generator.class_indices
    print("Label mapping",label_map)
    #evaluate model on test data
    score=model_test.evaluate_generator(test_generator)
    print("loss: %.3f - acc: %.3f" % (score[0], score[1]))
    print("List of image directory",image_dir)
    
    predictions = model_test.predict_generator(test_generator, steps=13)
    print("predictions",predictions)
    
    
    val_preds = np.argmax(predictions, axis=-1)
    print("predictions",val_preds)
    print("predictions type",type(val_preds))
    val_trues = test_generator.classes
    print("true values",val_trues)
    print("true values type",type(val_trues))
    #print("true values shape",val_trues)
    
    labels = test_generator.class_indices.keys()
    print("classification report on validation data: \n",metrics.classification_report(val_trues.tolist(), val_preds.tolist()))
    
    print("The validation accuracy  is : ",metrics.accuracy_score(val_trues.tolist(), val_preds.tolist()))
    if image_dir is not None:
        for filename in os.listdir(image_dir):
            image = cv2.imread(os.path.join(image_dir,filename))
            image = cv2.resize(image, (299,299))
            image_exp=np.expand_dims(image,axis=0)
            #classes=model_test.predict_classes(image_exp)
            predict_prob=model_test.predict(image_exp)
            classes=np.argmax(predict_prob,axis=1)
            for key in label_map:
                if label_map[key]==classes[0]:
                    print("The image {} is of :{}".format(filename,key))
            



if __name__ == '__main__':
    # if GPU available then it trains on GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str,
                        help='path to the directory where the test images will be read from')
    parser.add_argument('--output_weight', type=str,
                        help='path to the directory where the trained weights will be saved to')
    parser.add_argument('--image_dir', type=str,
                        help='path of the  image directory', default=None)
    args = parser.parse_args()
    main(args)
