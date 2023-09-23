# import the necessary packages
import os

# define the path to the base output directory
BASE_OUTPUT = "output"

# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 100
FULL_DATA = True
SAVED_DATA = True
DATASET_DIR_PATH = "C:\\Users\\chand\\Desktop\\CV_Project\\dataset\\lfw"
DATASET_LABELS_PATH = "C:\\Users\\chand\\Desktop\\CV_Project\\dataset\\train.txt"
SAVED_DATASET_PATH = "C:\\Users\\chand\\Desktop\\CV_Project\\dataset\\lfw_images_labels_array_new.npz"
SAVED_DATASET_NAME = "C:\\Users\\chand\\Desktop\\CV_Project\\dataset\\lfw_images_labels_array_new"
TL_EPOCHS = 100
TL_BATCH_SIZE = 1024
TL_IMG_SHAPE = (64, 64, 3)
TL_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "triplet_network_model.h5"])
TL_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "triplet_plot.png"])

TEST_DATASET_PATH = "C:\\Users\\chand\\Downloads\\CV_Project\\test_set.npz"
TEST_DATALABEL_PATH = "C:\\Users\\chand\\Downloads\\CV_Project\\test_labels.npy"

