# %%
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# %%
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
 
# For a single device (GPU 5) 
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


from tensorflow.python.client import device_lib
AVAILABLE_GPUS = len([gpu for gpu in device_lib.list_local_devices() if gpu.device_type=="GPU"])
if(AVAILABLE_GPUS == 0):
    AVAILABLE_GPUS = 1
print("AVAILABLE_GPUS=", AVAILABLE_GPUS)

# %%
# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


# %%
EXPERIMENT_NO = 17
LOWER_BOUND = 0   #AIRSPEED UPPER AND LOWER BOUND
UPPER_BOUND = 119
SIM_LABEL_COLUMN_NAME = "Airspeed(Ind)"
TEST_LABEL_COLUMN_NAME = "True Airspeed(kn)"
IMAGE_PATH = "image_path"
train_csv_path = "train_2022-01-05 15-08-53.mp4.csv"
test_root_folder_path = "C:/Users/Mahmu/OneDrive - Rowan University/Desktop/tachometer"
test_csv_path = "{}/combined_csv.csv".format(test_root_folder_path)
TESTING_DIR = "{}/".format(test_root_folder_path)

View = "C1_Broomcloset_view"
View_root_folder_path = "./{}".format(View)
result_folder_path = "{}/Results".format(View_root_folder_path)
if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)
    print(result_folder_path + " directory created.")
else:
    print(result_folder_path + " already exists.")

c_experiment_data_path = "{0}/Experiment_{1}".format(result_folder_path, EXPERIMENT_NO)
if not os.path.exists(c_experiment_data_path):
    os.makedirs(c_experiment_data_path)
    print(c_experiment_data_path + " directory created.")
else:
    print(c_experiment_data_path + " already exists.")

# HEIM_CONFIG = pd.read_csv("./{}/")

SPLIT_RATIO = 0.2
SEED = 101

if(os.path.isfile("{}/train_csv.csv".format(c_experiment_data_path))):
    train_csv = pd.read_csv("{}/train_csv.csv".format(c_experiment_data_path))
    test_csv = pd.read_csv("{}/test_csv.csv".format(c_experiment_data_path))
    print("Experiment train and test CSV already exist")

else:
    print('Reading training csv file...')
    train_csv = pd.read_csv('{}'.format(train_csv_path), index_col = False)
    print("Dropping {} values are lower than {} and upper than {} values...".format(SIM_LABEL_COLUMN_NAME, LOWER_BOUND, UPPER_BOUND))
    train_csv = train_csv.drop(train_csv[(train_csv[SIM_LABEL_COLUMN_NAME] < LOWER_BOUND ) | (train_csv[SIM_LABEL_COLUMN_NAME] > UPPER_BOUND)].index)
    print("TRAIN CSV : {}".format(len(train_csv)))
    train_csv[SIM_LABEL_COLUMN_NAME] = train_csv[SIM_LABEL_COLUMN_NAME].astype(int)
    OUTPUT_CLASSES = len(train_csv[SIM_LABEL_COLUMN_NAME].unique())
    print("Number of Training classes : {}".format(OUTPUT_CLASSES))

    print("Reading testing csv file ...")
    test_csv = pd.read_csv("{}".format(test_csv_path), index_col = False)
    print("Dropping {} values are lower than {} and upper than {} values...".format(TEST_LABEL_COLUMN_NAME, LOWER_BOUND, UPPER_BOUND))
    test_csv = test_csv.drop(test_csv[(test_csv[TEST_LABEL_COLUMN_NAME] < LOWER_BOUND ) | (test_csv[TEST_LABEL_COLUMN_NAME] > UPPER_BOUND)].index)
    print("Number of Testing Classes : " + str(len(train_csv[SIM_LABEL_COLUMN_NAME].unique())))
    print("TEST CSV : {}".format(len(test_csv)))

# %%
# ## ============ ONCELIKLE LABEL COLUMNLARINI BIN SIZE A AYIRMAM GEREKIYOR. 
# # MESELA TRAIN_CSV DEKI AIRSPEED VALUELARINI 1 ER 1 ER DEGIL DE 3 ER 3 ER BINLERE AYIRIP,
# # ONLARIN DEGERLERINI YENI BIR COLUMNDA TUTMAM LAZIM. (isimleri Airspeed(Ind)_D_3 3 burada 3 er ucer arttirdigimi temsil ediyor.)

# DEGREES = [1,3,5]#,4,5,6]
# distributions = {}
# for degree in DEGREES:
    
#     distributions["Train_Degree_{0}".format(degree)] = {}
#     distributions["Test_Degree_{0}".format(degree)] = {}
#     for label in range(0, OUTPUT_CLASSES):
#         distributions["Train_Degree_{0}".format(degree)]["{0}".format(label)] = len(train_csv["{}_D_{}".format(SIM_LABEL_COLUMN_NAME,degree)]== label)
#         distributions["Test_Degree_{0}".format(degree)]["{0}".format(label)] = len(test_csv[test_csv["{}_D_{}".format(TEST_LABEL_COLUMN_NAME ,degree)]== label])

#     plt.clf()
#     plt.hist(train_csv["{0}_D_{1}".format(SIM_LABEL_COLUMN_NAME, degree)], bins = OUTPUT_CLASSES, align="mid")
#     plt.xlabel("Classes")
#     plt.ylabel("Number of Examples")
#     plt.title(r'{1} for $\alpha$={0}'.format(degree, "Joint train distribution D {0}".format(degree)))
#     plt.tight_layout()
#     plt.savefig("{0}/Joint_Train_Distrbtn_D_{1}.png".format(c_experiment_data_path, 
#                                 degree), dpi=300)

    
#     plt.clf()
#     plt.hist(test_csv["{0}".format(TEST_LABEL_COLUMN_NAME)], bins = OUTPUT_CLASSES, align="mid")
#     plt.xlabel("Classes")
#     plt.ylabel("Number of Examples")
#     plt.title(r'{1} for $\alpha$={0}'.format(degree, "Joint test distribution D {0}".format(degree)))
#     plt.tight_layout()
#     plt.savefig("{0}/Joint_Test_Distrbtn_D_{1}.png".format(c_experiment_data_path, 
#                                 degree), dpi=300)
    
    
        
# with open("{0}/Train_Test_Distributions_Info.txt".format(c_experiment_data_path), 'w') as file:  
#     for key in distributions.keys():
#         file.write(str(key) + ":" + str(distributions[key]) + ",")
#         file.write("\n\n")
    
# (pd.DataFrame.from_dict(data=distributions, orient='index')
#    .to_csv("{0}/Train_Test_Distributions.csv".format(c_experiment_data_path), header=True))


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import model_from_json
import os
import datetime
import numpy
import gc
from time import sleep
from keras import backend as K
# %matplotlib inline
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    print("tick_marks=",tick_marks)
    plt.xticks(tick_marks, classes)#, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_history(history, model_name, c_model_data_path):
    plt.figure()
        # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('{0} accuracy'.format("{0}".format(model_name).split(".")[1]))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("{0}/ACC_Plot_MName_{1}.jpeg".format(c_model_data_path, model_name), dpi=300)
    plt.show()
   

    plt.figure()
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('{0} loss'.format("{0}".format(model_name).split(".")[1]))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("{0}/LOSS_Plot_MName_{1}.jpeg".format(c_model_data_path, model_name), dpi=300)
    plt.show()


# %%
BATCH_SIZE_PER_REPLICA = 64
LR = 0.01
IMAGE_SIZE = 67 
INPUT_SIZE = (IMAGE_SIZE, IMAGE_SIZE, 3)
EPOCHS = 250
MONITOR = "val_loss"
EARLY_STOPPING_PATIENCE = 5
LOSS = "categorical_crossentropy"
PREDICTION_ACTIVATION = "softmax"
DROPOUT_RATE = 0.5
INCLUDE_TOP = False
POOLING = None#"avg"
FINAL_LAYER_CLASSIFICATION_FUCNTION = "softmax"
INITIALIZATION_WIEGHTS = "imagenet"
Is_To_Add_Dropout = True
# Dropout_Rate = 0.5
ML_Model = None

TRAINING_DIR = "gauge_images/airspeed/2022-01-05 15-08-53.mp4/"
xcol = IMAGE_PATH
ycol = SIM_LABEL_COLUMN_NAME

# %%

# EXPERIMENT_NO = 15
# TRAIN_CSV = "training.csv"
# label = 'Airspeed(Ind)'
# image_path = "image_path"
# TRAINING_DIR = "gauge_images/airspeed/combined/"
# IMAGE_SIZE = 67
# EPOCHS = 111
# BATCH_SIZE_PER_REPLICA = 32
# LR = 0.001
# CHECKPOINT_PATH = "./Weights/{}_Resnet50_xplane_airspeed_weights.hdf5".format(EXPERIMENT_NO)
# MONITOR = 'val_loss'
# xcol, ycol = image_path, label
# INPUT_SIZE = (IMAGE_SIZE, IMAGE_SIZE, 3)
# EARLY_STOPPING_PATIENCE = 25
# LOSS ="categorical_crossentropy" #"binary_crossentropy"
# PREDICTION_ACTIVATION ="softmax"
# DROPOUT_RATE = 0.5
# INCLUDE_TOP = False
# POOLING = None#"avg"
# FINAL_LAYER_CLASSIFICATION_FUCNTION = "softmax"
# INITIALIZATION_WIEGHTS = "imagenet"
# Is_To_Add_Dropout = True
# # Dropout_Rate = 0.5
# ML_Model = None

# %%
# def load_model_from_path(c_model_data_path):
#     # load YAML and create model
#     with open(c_model_data_path) as yaml_file:
#         loaded_model_yaml = yaml_file.read()

#     loaded_model = model_from_yaml(loaded_model_yaml)
#     return loaded_model

# def load_weights_from_path(model, weights_path, available_gpu):
#     if(available_gpu > 1):
#         model = multi_gpu_model(model, available_gpu)
#         print("Multi-GPU model weights loaded. AVAILABLE_GPUS: {0}".format(AVAILABLE_GPUS))
#     model.load_weights(weights_path)
#     return model

# %%

from sklearn.model_selection import train_test_split 

X_train, X_valid, y_train, y_valid = train_test_split(train_csv[[IMAGE_PATH]], train_csv[SIM_LABEL_COLUMN_NAME], test_size = 0.1, random_state= 42) #, stratify= df[label]) 

# %%

import os
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D,Dropout, GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from tensorflow.keras.applications import VGG16, VGG19, InceptionV3
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from tensorflow.keras import optimizers , layers, applications
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D,Dropout, GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from tensorflow.keras.applications import VGG16, VGG19, InceptionV3
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, save_model
import efficientnet.tfkeras as efn

# %%
from enum import Enum
from keras import applications as K_Models

# MODEL SELECTION HELP OR AVAILABLE MODELS
class Available_Models(Enum):
    VGG16 = "VGG16"
    VGG19 = "VGG19"
    InceptionV3 = "InceptionV3"
    Xception = "Xception" # TensorFlow ONLY
    ResNet50 = "ResNet50"
    DenseNet121 = "DenseNet121"
    DenseNet169 = "DenseNet169"
    DenseNet201 = "DenseNet201"
    MobileNet = "MobileNet"
    InceptionResNetV2 = "InceptionResNetV2"
    NASNetLarge = "NASNetLarge"
    NASNetMobile = "NASNetMobile"
    EfficeintNetB0 = "efficientnetb0"
    EfficeintNetB1 = "efficientnetb1"
    EfficeintNetB2 = "efficientnetb2"
    EfficeintNetB3 = "efficientnetb3"
    EfficeintNetB4 = "efficientnetb4"
    EfficeintNetB5 = "efficientnetb5"
    EfficeintNetB6 = "efficientnetb6"
    EfficeintNetB7 = "efficientnetb7"
    
##############################################################################################   
def Load_Model(SELECTED_MODEL):
#     global ML_Model
    if(SELECTED_MODEL == Available_Models.VGG16):
        ML_Model = K_Models.VGG16(include_top=INCLUDE_TOP, weights=INITIALIZATION_WIEGHTS, 
#                                                               input_tensor=INPUT_TENSOR,
                                  input_shape=INPUT_SIZE,
                                                              pooling=POOLING, classes=OUTPUT_CLASSES)
    elif(SELECTED_MODEL == Available_Models.VGG19):
        ML_Model = K_Models.VGG19(include_top=INCLUDE_TOP, weights=INITIALIZATION_WIEGHTS, 
#                                                               input_tensor=INPUT_TENSOR, 
                                  input_shape=INPUT_SIZE,
                                                              pooling=POOLING, classes=OUTPUT_CLASSES)
    elif(SELECTED_MODEL == Available_Models.InceptionV3):
        ML_Model = K_Models.InceptionV3(include_top=INCLUDE_TOP, weights=INITIALIZATION_WIEGHTS, 
#                                                               input_tensor=INPUT_TENSOR, 
                                        input_shape=INPUT_SIZE,
                                                              pooling=POOLING, classes=OUTPUT_CLASSES)
    elif(SELECTED_MODEL == Available_Models.ResNet50):

        ML_Model = K_Models.ResNet50(include_top=INCLUDE_TOP, weights=INITIALIZATION_WIEGHTS, 
#                                                               input_tensor=INPUT_TENSOR,
                                     input_shape=INPUT_SIZE,
                                                              pooling=POOLING, classes=OUTPUT_CLASSES)
    elif(SELECTED_MODEL == Available_Models.DenseNet121):
        ML_Model = K_Models.DenseNet121(include_top=INCLUDE_TOP, weights=INITIALIZATION_WIEGHTS, 
#                                                               input_tensor=INPUT_TENSOR,
                                        input_shape=INPUT_SIZE,
                                                              pooling=POOLING, classes=OUTPUT_CLASSES)
    elif(SELECTED_MODEL == Available_Models.DenseNet169):
        ML_Model = K_Models.DenseNet169(include_top=INCLUDE_TOP, weights=INITIALIZATION_WIEGHTS, 
#                                                               input_tensor=INPUT_TENSOR,
                                        input_shape=INPUT_SIZE,
                                                              pooling=POOLING, classes=OUTPUT_CLASSES)
    elif(SELECTED_MODEL == Available_Models.DenseNet201):
        ML_Model = K_Models.DenseNet201(include_top=INCLUDE_TOP, weights=INITIALIZATION_WIEGHTS, 
#                                                               input_tensor=INPUT_TENSOR, 
                                        input_shape=INPUT_SIZE,
                                                              pooling=POOLING, classes=OUTPUT_CLASSES)
    elif(SELECTED_MODEL == Available_Models.MobileNet):
        ML_Model = K_Models.MobileNet(include_top=INCLUDE_TOP, weights=INITIALIZATION_WIEGHTS, 
#                                                               input_tensor=INPUT_TENSOR,
                                      input_shape=INPUT_SIZE,
                                                              pooling=POOLING, classes=OUTPUT_CLASSES)

    elif(SELECTED_MODEL == Available_Models.InceptionResNetV2):
        ML_Model = K_Models.InceptionResNetV2(include_top=INCLUDE_TOP, weights=INITIALIZATION_WIEGHTS, 
#                                                               input_tensor=INPUT_TENSOR,
                                              input_shape=INPUT_SIZE,
                                                              pooling=POOLING, classes=OUTPUT_CLASSES)
    elif(SELECTED_MODEL == Available_Models.NASNetLarge):
        ML_Model = K_Models.NASNetLarge(include_top=INCLUDE_TOP, weights=INITIALIZATION_WIEGHTS, 
#                                                               input_tensor=INPUT_TENSOR,
                                        input_shape=INPUT_SIZE,
                                                            pooling=POOLING, classes=OUTPUT_CLASSES)
    elif(SELECTED_MODEL == Available_Models.Xception):
        ML_Model = K_Models.Xception(include_top=INCLUDE_TOP, weights=INITIALIZATION_WIEGHTS, 
#                                                               input_tensor=INPUT_TENSOR, 
                                     input_shape=INPUT_SIZE,
                                                              pooling=POOLING, classes=OUTPUT_CLASSES)
    elif(SELECTED_MODEL == Available_Models.NASNetMobile):
        ML_Model = K_Models.NASNetMobile(include_top=INCLUDE_TOP, weights=INITIALIZATION_WIEGHTS, 
#                                                               input_tensor=INPUT_TENSOR,
                                         input_shape=INPUT_SIZE,
                                                              pooling=POOLING, classes=OUTPUT_CLASSES)
    elif(SELECTED_MODEL == Available_Models.EfficeintNetB0):
        ML_Model = efn.EfficientNetB0(weights=INITIALIZATION_WIEGHTS, include_top=INCLUDE_TOP, 
                           input_shape=INPUT_SIZE, pooling=POOLING)
    elif(SELECTED_MODEL == Available_Models.EfficeintNetB1):
        ML_Model = efn.EfficientNetB1(weights=INITIALIZATION_WIEGHTS, include_top=INCLUDE_TOP, 
                           input_shape=INPUT_SIZE, pooling=POOLING)
    elif(SELECTED_MODEL == Available_Models.EfficeintNetB2):
        ML_Model = efn.EfficientNetB2(weights=INITIALIZATION_WIEGHTS, include_top=INCLUDE_TOP, 
                           input_shape=INPUT_SIZE, pooling=POOLING)
    elif(SELECTED_MODEL == Available_Models.EfficeintNetB3):
        ML_Model = efn.EfficientNetB3(weights=INITIALIZATION_WIEGHTS, include_top=INCLUDE_TOP, 
                           input_shape=INPUT_SIZE, pooling=POOLING)
    elif(SELECTED_MODEL == Available_Models.EfficeintNetB4):
        ML_Model = efn.EfficientNetB4(weights=INITIALIZATION_WIEGHTS, include_top=INCLUDE_TOP, 
                           input_shape=INPUT_SIZE, pooling=POOLING)
    elif(SELECTED_MODEL == Available_Models.EfficeintNetB5):
        ML_Model = efn.EfficientNetB5(weights=INITIALIZATION_WIEGHTS, include_top=INCLUDE_TOP, 
                           input_shape=INPUT_SIZE, pooling=POOLING)
    elif(SELECTED_MODEL == Available_Models.EfficeintNetB6):
        ML_Model = efn.EfficientNetB6(weights=INITIALIZATION_WIEGHTS, include_top=INCLUDE_TOP, 
                           input_shape=INPUT_SIZE, pooling=POOLING)
    elif(SELECTED_MODEL == Available_Models.EfficeintNetB7):
        ML_Model = efn.EfficientNetB7(weights=INITIALIZATION_WIEGHTS, include_top=INCLUDE_TOP, 
                           input_shape=INPUT_SIZE, pooling=POOLING)

    else:
        print("NO MODEL IDENTIFIER MATCHED")


    if(POOLING is None):
        last_layer = Flatten()(ML_Model.output)
    else:
        last_layer = ML_Model.output
        
    if(Is_To_Add_Dropout):
#         for neurons in Dense_Layers:
#             last_layer = Dense(neurons, activation='relu')(last_layer)
        last_layer =  Dropout(DROPOUT_RATE)(last_layer)
        
    last_layer = Dense(OUTPUT_CLASSES, activation=FINAL_LAYER_CLASSIFICATION_FUCNTION,
                       name='softmax')(last_layer)
    ML_Model = Model(ML_Model.input, last_layer)
    return ML_Model

# %%
# from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

# %%
train_csv = pd.concat([X_train,y_train],axis=1)
train_csv[SIM_LABEL_COLUMN_NAME] = train_csv[SIM_LABEL_COLUMN_NAME].astype(str) 
val_csv = pd.concat([X_valid,y_valid],axis=1)
val_csv[SIM_LABEL_COLUMN_NAME] = val_csv[SIM_LABEL_COLUMN_NAME].astype(str) 
SEED = 101

# %%
tf.debugging.set_log_device_placement(True)
GPUS = ["GPU:2", "GPU:3"]#,"GPU:2", "GPU:3","GPU:4","GPU:5", "GPU:6"]
strategy = tf.distribute.MirroredStrategy(GPUS)
strategy.num_replicas_in_sync
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * AVAILABLE_GPUS


# %%
# %% 
def train_model(train_csv, val_csv, 
                y_col, x_col, class_weights,
                model_name="efficientnet"):
    print("SELECTED MODEL=", model_name)
    c_model_data_path = "{0}/{1}".format(c_experiment_data_path, model_name)
    if not os.path.exists(c_model_data_path):
        os.makedirs(c_model_data_path)
        print(c_model_data_path + " directory created.")
    
    c_experiment_cam_data_path = "{0}/CAM".format(c_model_data_path)
    if not os.path.exists(c_experiment_cam_data_path):
        os.makedirs(c_experiment_cam_data_path)
    train_data = ImageDataGenerator(
            rescale=1./255., 
#             preprocessing_function=INPUT_PRE_PROCESSING
                                   )
    val_data = ImageDataGenerator(
        rescale=1./255., 
#         preprocessing_function=INPUT_PRE_PROCESSING
                                 )

#         if(train_datagen is None):
#             print("Error train_data is none")

    train_generator = train_data.flow_from_dataframe(
                        dataframe=train_csv,
                        directory=TRAINING_DIR,
                        x_col=x_col,
                        y_col=y_col,
#                             subset="training",
                        batch_size=BATCH_SIZE,
                        seed=SEED,
                        shuffle=True,
                        class_mode="categorical",
                        target_size=(INPUT_SIZE[0], INPUT_SIZE[0]))

    val_generator  = val_data.flow_from_dataframe(
                        dataframe=val_csv,
                        directory = TRAINING_DIR,
                        x_col=x_col,
                        y_col=y_col,
#                             subset="training",
                        batch_size=BATCH_SIZE,
                        seed=SEED,
#                         shuffle=False,
                        class_mode="categorical",
                        target_size=(INPUT_SIZE[0], INPUT_SIZE[0]))
                                                               

    # if(AVAILABLE_GPUS):
    #     with tf.device('/cpu:0'):
    #         model = Load_Model(SELECTED_MODEL=model_name)
    # else:
    #     model = Load_Model(SELECTED_MODEL=model_name)
    # p_model = model
    # if(AVAILABLE_GPUS > 1):
    #     p_model = multi_gpu_model(model, AVAILABLE_GPUS)
    #     print("Multi-GPU model training. AVAILABLE_GPUS: {0}".format(AVAILABLE_GPUS))
    
        
        
#     #optimization details
#     adam = optimizers.Adam(learning_rate=LR)#SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
#     # categorical_crossentropy
#     p_model.compile(loss=LOSS, optimizer=adam, metrics=['accuracy'])
# #     model.summary()
    

    # {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    best_weights_path = "{0}/Checkpoint_Weights_MName_{1}_LR_{2}_CW_{3}.hdf5".format(c_model_data_path , model_name, LR, WEIGHT_MULTIPLIER)
    checkpointer = ModelCheckpoint(filepath=best_weights_path, verbose=1, 
                               save_best_only=True, monitor=MONITOR)
    early_stopping = EarlyStopping(monitor=MONITOR, min_delta=0, patience=EARLY_STOPPING_PATIENCE, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=7, 
                                                    verbose=1, 
                                                    factor=0.95,
                                                    min_lr=0.00009)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="{}/logs/".format(c_model_data_path), histogram_freq=1)
#     return
#     Fit the Model
#     try:
#         model = load_weights_from_path(model=model, weights_path=best_weights_path, available_gpu=AVAILABLE_GPUS)
#     except Exception as ex:
#         print("Weights Not FOUND", str(ex))
    tf.get_logger().setLevel('ERROR')

    import time
    start = time.time()
    with strategy.scope():
        model = Load_Model(SELECTED_MODEL=model_name)
        #optimization details
        adam = optimizers.Adam(learning_rate=LR)#SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        # categorical_crossentropy
        model.compile(loss=LOSS, optimizer=adam, metrics=['accuracy'])
    #     model.summary()

        history = model.fit(train_generator,
                    epochs=EPOCHS,
                    validation_data=val_generator,
                    verbose=1, 
                    class_weight= class_weights,
                    callbacks=[checkpointer, early_stopping, reduce_lr, tb_callback],
                    workers = 4, 
                    use_multiprocessing=False,
                    max_queue_size=4
                    #https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/
                   )



# ==============================================================================================================================================

#     model.save("{0}/{1}".format(c_model_data_path , model_name))
    #saving the model to a YAML file
    # writing the yaml model to the yaml file

    with open("{0}/architecture.yaml".format(c_model_data_path), 'w') as yaml_file:
        yaml_file.write(model.to_yaml())
#     return
        # print(history.history)
    plot_history(history=history, model_name=model_name, c_model_data_path=c_model_data_path)
    hist_df = pd.DataFrame(history.history) 
    with open("{0}/history_MName_{1}_LR_{2}_CW_{3}.csv".format(c_model_data_path, model_name, LR, WEIGHT_MULTIPLIER), mode='w') as his:
        hist_df.to_csv(his)
 


# ===============================================TESTING================================================   
#     predictions = []
#     GTs = []
# #     val_generator.reset()

#     test_data = ImageDataGenerator(
#         rescale=1./255., 
# #         preprocessing_function=INPUT_PRE_PROCESSING
#                                  )

# #         if(train_datagen is None):
# #             print("Error train_data is none")

#     test_generator = test_data.flow_from_dataframe(
#                         dataframe=test_csv,
#                         directory=TESTING_DIR,
#                         x_col=x_col,
#                         y_col=None,
# #                             subset="training",
#                         batch_size=1,
#                         seed=SEED,
#                         shuffle=False,
#                         class_mode=None,
#                         target_size=(INPUT_SIZE[0], INPUT_SIZE[0]))

#     try:
#         preds = np.argmax(model.predict(test_generator,batch_size = 1))
#         predictions.append(preds)
#         GTs.append(test_csv[TEST_LABEL_COLUMN_NAME])
#     except Exception as expn:
#         print("Exception occurred during prediction")
        
# #     predictions =np.argmax(p_model.predict_generator(generator=val_generator), axis=1).tolist()
# #     GTs = val_generator.classes#.tolist()#np.argmax(validation_generator.classes, axis=1).tolist()
    
#     print("GTS:", len(GTs))
#     print("predictions:", len(predictions))
#     print(GTs[0])
#     print(predictions[0])
#     print(type(predictions))
#     print(type(GTs))
#     report = pd.DataFrame([],
#                columns=['GT', 'Prediction'])
#     report["GT"]= GTs
#     report["Prediction"] = predictions
#     report.to_csv("{0}/Report_MName_{1}_LR_{2}_CW_{3}.csv".format(c_model_data_path, model_name, LR, WEIGHT_MULTIPLIER))
    
#     cnf_matrix = confusion_matrix(GTs, predictions)
#     numpy.set_printoptions(precision=2)
#     plt.figure()

#     plot_confusion_matrix(cnf_matrix, classes = numpy.arange(OUTPUT_CLASSES),
#             normalize=True,
#                       title='{0} Confusion Matrix'.format("{0}".format(model_name).split(".")[1]))
    
#     plt.savefig("{0}/CM_MName_{1}_LR_{2}_CW_{3}.png".format(c_model_data_path, model_name, LR, WEIGHT_MULTIPLIER), 
#                 dpi = 300)
#     plt.show()
#     plt.clf()


    # print("Loading weights: ", best_weights_path)
    # model = load_weights_from_path(model=model, weights_path=best_weights_path, available_gpu=AVAILABLE_GPUS)
# ===============================================TESTING================================================   



# ========================= EXECUTE THE CODE ===========================

LRs = [0.0001, 0.00001]#, 0.00001] #, 0.005, 0.009]
CL_Weights = [1, 2]
WEIGHT_MULTIPLIER = 1
INPUT_PRE_PROCESSING = None

for lr in LRs:
    LR = lr
    print("LR=", LR)
    GT_Column = "Airspeed(Ind)"
    Image_Path = "image_path"
    train_csv[GT_Column] = train_csv[GT_Column].astype('str')
    val_csv[GT_Column] = val_csv[GT_Column].astype('str')


 
    from sklearn.utils import class_weight
    class_weights = dict(zip(np.unique(y_train),
                        class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))) 
    x = class_weights.keys()
    y = class_weights.values()
    plt.bar(x,y, 0.5)
    plt.xlabel('Categories')
    plt.ylabel("Values")
    plt.title('Class Weightage')
    plt.legend('')
    plt.savefig("{}/class_weights.png".format(c_experiment_data_path))

    # class_weights = {k: v for k, v in enumerate([x * weights_multiplr for x in list(class_weights.values())])}
    # print("class_weights={0} for CL_Multiplier {1}".format(class_weights, weights_multiplr))  

    INPUT_SIZE = (224, 224, 3)

    INPUT_PRE_PROCESSING = efn.preprocess_input
    train_model(train_csv=train_csv, val_csv=val_csv, y_col=GT_Column, 
                x_col=Image_Path, model_name=Available_Models.VGG16, class_weights=class_weights)

# %%
