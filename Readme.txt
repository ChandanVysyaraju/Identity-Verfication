README
install the following
cv2, numpy, tensorflow, keras, matplotlib, imutils

Download LFW dataset and place it in dataset folder
Train the dataset with train.txt in the dataset folder to label the images and it will create a txt file with image labels
Give the configuration paths inside the config.py
Train the model using train_triplet_network.py
Test the model using test_triplet_network.py 
Command is python test_triplet_network.py - i "path of testimages folder" 
