
# Split images into train, validation and test.
The script `split_train_test.py` will split image folders into train, test and validation set. Python version 3.6 and above is required to run this script.The format of image folder should be:  
```
    ├── /home/documents/data
      ├── class1
      │  ├──01.jpg  
      │  ├──02.jpg
      │
      │     
      |       
      ├── class2 
      │  ├──01.jpg  
      │  ├──02.jpg  
      │      
      
          
```

Demo:    
``` 
python3 split_train_test.py --input_dir /home/documents/data --output_dir /home/documents/output
```

# Augment the data in the training folder to increase training data:
This should be run after splitting the image data into ,train , test and validation set. The script `data_update.py` will apply data augmentation techniques:affine transformation and gaussian noise to the training data. After augmentation the new images will be added in the same corresponding folders  
Demo:    
``` 
python3 data_update.py --train_dir /home/documents/output/train
```

# Training the VGG16 model:
The pretrained imagenet weights of vgg16 in .h5 format should be downloaded and saved.  

Demo:    
``` 
python3 vgg_keras_train.py --input_dir /home/documents/output --output_weight /home/documents/output_weight --vgg_weight /home/documents/vgg_weight
```

# Manage configuration:
`config.py` has the necessary configuration parameters that needs to be changed accordingly. Example:
```
num_classes = 2
train_batch_size = 64
val_batch_size = 64
image_height = 224
image_width = 224
image_channels = 3
num_iters = 100
lr = 0.00001
momentum = 0.9
```
# Test accuracy on test data:
To find out the accuracy on the test data after training.  
Demo:
```
python3 vgg_keras_test.py --test_dir /home/documents/output/test --output_weight /home/documents/output_weight --image_dir /home/documents/demo_images
```
Here, image_dir is is path of some images that can be used to verfy the model i.e if the model correctly predicts the images present.