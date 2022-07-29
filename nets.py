'''
Similar to plotter, this is a utility script to store functions to make U-Nets
'''
from keras.utils import np_utils
import numpy as np
from keras.models import Model
from keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, UpSampling3D, Concatenate
from keras.layers import Input, BatchNormalization, Flatten, Dense, Dropout, Activation, Convolution3D
from keras import optimizers as opt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, MeanIoU, SparseCategoricalAccuracy, BinaryAccuracy
from keras import backend as K
import volumes

#---------------------------------------------------------
#  Vanilla dice coefficient
#---------------------------------------------------------
def dice_coef(y_true, y_pred):
    print('>>>>>> ', y_true.shape)
    smooth   = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # print(type(y_true_f))
    # print(type(y_pred_f))

    # print(y_true_f.dtype)
    # print(y_pred_f.dtype)

    y_true_f = K.cast(y_true_f, dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#---------------------------------------------------------
#
#---------------------------------------------------------
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#---------------------------------------------------------
# List of metrics: history.history will store these quantities for each model.
# CLASSIFICATION ONLY
# accuracy in most cases will default to binary_acccuracy but not necessarily.
#---------------------------------------------------------
cla_metrics_list = [dice_coef,BinaryAccuracy(),SparseCategoricalAccuracy(),MeanIoU(num_classes=2)]
reg_metrics_list = [MeanSquaredError(),MeanAbsoluteError(),MeanAbsolutePercentageError()]

#---------------------------------------------------------
# Summary statistics for a Numpy array
#---------------------------------------------------------
def summary(array):
  print('### Summary Statistics ###')
  print('Shape: ',str(array.shape))
  print('Mean: ',np.mean(array))
  print('Median: ',np.median(array))
  print('Maximum: ',np.max(array))
  print('Minimum: ',np.min(array))
  print('Std deviation: ',np.std(array))
#---------------------------------------------------------
#
#---------------------------------------------------------
def minmax(a):
    return (a-np.min(a))/(np.max(a)-np.min(a))
#---------------------------------------------------------
# 7/21 NOTE this is the one to use
#---------------------------------------------------------
def assemble_cube2(Y_pred,GRID,SUBGRID, OFF):
    cube  = np.zeros(shape=(GRID,GRID,GRID))
    nbins = GRID // SUBGRID + 1 + 1 + 1
    cont  = 0
    
    SUBGRID_4 = SUBGRID//4
    SUBGRID_2 = SUBGRID//2
    
    for i in range(nbins):
        if i==0:
            di_0 = SUBGRID*i - OFF*i
            di_1 = SUBGRID*i - OFF*i + SUBGRID_4+SUBGRID_2
            si_0 =  0
            si_1 = -SUBGRID_4
        else:
            di_0 = SUBGRID*i - OFF*i + SUBGRID_4
            di_1 = SUBGRID*i - OFF*i + SUBGRID_4+SUBGRID_2
            si_0 =  SUBGRID_4
            si_1 = -SUBGRID_4            
            if i==nbins-1:
                di_0 = SUBGRID*i - OFF*i + SUBGRID_4
                di_1 = SUBGRID*i - OFF*i + SUBGRID
                si_0 =  SUBGRID_4
                si_1 =  SUBGRID

        for j in range(nbins):
            if j==0:
                dj_0 = SUBGRID*j - OFF*j
                dj_1 = SUBGRID*j - OFF*j + SUBGRID_4+SUBGRID_2
                sj_0 =  0
                sj_1 = -SUBGRID_4
            else:
                dj_0 = SUBGRID*j - OFF*j + SUBGRID_4
                dj_1 = SUBGRID*j - OFF*j + SUBGRID_4+SUBGRID_2
                sj_0 = SUBGRID_4
                sj_1 = -SUBGRID_4
                if j==nbins-1:
                    dj_0 = SUBGRID*j - OFF*j + SUBGRID_4
                    dj_1 = SUBGRID*j - OFF*j + SUBGRID
                    sj_0 = SUBGRID_4
                    sj_1 = SUBGRID
                                                                    
                                            
            for k in range(nbins):
                if k==0:
                    dk_0 = SUBGRID*k - OFF*k
                    dk_1 = SUBGRID*k - OFF*k + SUBGRID_4+SUBGRID_2
                    sk_0 =  0
                    sk_1 = -SUBGRID_4
                else:
                    dk_0 = SUBGRID*k - OFF*k + SUBGRID_4
                    dk_1 = SUBGRID*k - OFF*k + SUBGRID_4+SUBGRID_2
                    sk_0 =  SUBGRID_4
                    sk_1 = -SUBGRID_4
                    if k==nbins-1:
                        dk_0 = SUBGRID*k - OFF*k + SUBGRID_4
                        dk_1 = SUBGRID*k - OFF*k + SUBGRID
                        sk_0 = SUBGRID_4
                        sk_1 = SUBGRID                                                                                                        
                    
                cube[di_0:di_1, dj_0:dj_1, dk_0:dk_1] = Y_pred[cont, si_0:si_1, sj_0:sj_1, sk_0:sk_1,0]
                cont = cont+1
                
    return cube
#---------------------------------------------------------
# Don't use this function...reconstruction issues with Indra
#---------------------------------------------------------
def assemble_cube(Y_pred,GRID,SUBGRID,OFF):
    cube  = np.zeros(shape=(GRID,GRID,GRID))
    nbins = GRID // SUBGRID + 1 + 1 + 1
    cont  = 0

    SUBGRID_4 = SUBGRID//4
    SUBGRID_2 = SUBGRID//2
    
    for i in range(nbins):
        off_i = SUBGRID*i - OFF*i + SUBGRID_4
        for j in range(nbins):
            off_j =SUBGRID*j - OFF*j + SUBGRID_4
            for k in range(nbins):
                off_k =SUBGRID*k - OFF*k + SUBGRID_4

                print(i,j,k, '|', off_i,off_i+SUBGRID_2,off_j,off_j+SUBGRID_2,off_k,off_k+SUBGRID_2,'|',SUBGRID_2)
                cube[off_i:off_i+SUBGRID_2, off_j:off_j+SUBGRID_2, off_k:off_k+SUBGRID_2] = Y_pred[cont,SUBGRID_4:-SUBGRID_4,SUBGRID_4:-SUBGRID_4,SUBGRID_4:-SUBGRID_4,0]
                cont = cont+1
                
    return cube
#---------------------------------------------------------
# For loading training and testing data for training
# if loading data for regression, ensure classification=False!!
#---------------------------------------------------------
def load_dataset_all(FILE_DEN, FILE_MASK, SUBGRID, DILATION=0, preproc=False, classification=True):
  print(f'Reading volume ')
  den = volumes.read_fvolume(FILE_DEN)
  print(f'Reading spine ')
  msk = volumes.read_fvolume(FILE_MASK)
  den_shp = den.shape
  msk_shp = msk.shape
  summary(den); summary(msk)
  
  if preproc == True:
    #den = minmax(np.log10(den))
    den = minmax(den)
    msk = minmax(msk)
    print('Ran preprocessing to scale both density and mask to [0,1]')
    print('New summary statistics: ')
    summary(den); summary(msk)

  # Make wall mask
  #msk = np.zeros(den_shp,dtype=np.uint8)
  n_bins = den_shp[0] // SUBGRID

  cont = 0 
  X_all = np.zeros(shape=((n_bins**3)*4, SUBGRID,SUBGRID,SUBGRID,1))
  if classification == False:
    Y_all = np.ndarray(((n_bins**3)*4, SUBGRID,SUBGRID,SUBGRID,1),dtype=np.float32)
  else:
    Y_all = np.ndarray(((n_bins**3)*4, SUBGRID,SUBGRID,SUBGRID,1),dtype=np.uint8)

  for i in range(n_bins):
    for j in range(n_bins):
      for k in range(n_bins):
        sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        X_all[cont,:,:,:,0] = sub_den
        sub_msk = msk[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        Y_all[cont,:,:,:,0] = sub_msk
        cont = cont+1

        sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_den = volumes.rotate_cube(sub_den,2)
        X_all[cont,:,:,:,0] = sub_den
        sub_msk = msk[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_msk = volumes.rotate_cube(sub_msk,2)
        Y_all[cont,:,:,:,0] = sub_msk
        cont = cont+1

        sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_den = volumes.rotate_cube(sub_den,1)
        X_all[cont,:,:,:,0] = sub_den
        sub_msk = msk[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_msk = volumes.rotate_cube(sub_msk,1)
        Y_all[cont,:,:,:,0] = sub_msk
        cont = cont+1

        sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_den = volumes.rotate_cube(sub_den,0)
        X_all[cont,:,:,:,0] = sub_den
        sub_msk = msk[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_msk = volumes.rotate_cube(sub_msk,0)
        Y_all[cont,:,:,:,0] = sub_msk
        cont = cont+1

    print(i,j,k)

  X_all = X_all.astype('float32')
  # Y_all = Y_all - 1
  return X_all, Y_all
#---------------------------------------------------------
#
#---------------------------------------------------------
def load_dataset(file_in, SUBGRID, OFF):
  #--- Read density field
  #den   = np.load(file_in)
  den = volumes.read_fvolume(file_in)
  
  nbins = (den.shape[0] // SUBGRID) + 1 + 1 + 1
  X_all = np.zeros(shape=(nbins**3, SUBGRID,SUBGRID,SUBGRID,1))
  
  cont  = 0
  for i in range(nbins):
    off_i = SUBGRID*i - OFF*i
    for j in range(nbins):
      off_j = SUBGRID*j - OFF*j
      for k in range(nbins):
        off_k = SUBGRID*k - OFF*k
        #print(i,j,k,'|', off_i,':',off_i+SUBGRID,',',off_j,':',off_j+SUBGRID,',',off_k,':',off_k+SUBGRID)
        sub_den = den[off_i:off_i+SUBGRID,off_j:off_j+SUBGRID,off_k:off_k+SUBGRID]
        X_all[cont,:,:,:,0] = sub_den
        cont = cont+1
      
  X_all = X_all.astype('float32')
  return X_all
#---------------------------------------------------------
#
#---------------------------------------------------------
def run_predict(model, FILE_WEIGHT, FILE_DEN, FILE_OUT, GRID, SUBGRID, OFF):
    print(FILE_WEIGHT, ' | ', FILE_DEN)
    model.load_weights(FILE_WEIGHT)
    X_train = load_dataset(FILE_DEN, SUBGRID, OFF)
    Y_pred = model.predict(X_train, batch_size=1, verbose = 2)
    PRED = assemble_cube2(Y_pred, GRID, SUBGRID, OFF)
    volumes.write_fvolume(PRED, FILE_OUT)
    return PRED
#---------------------------------------------------------
#
#---------------------------------------------------------
def subcube_run_pred(model,FILE_WEIGHT,FILE_DEN,FILE_OUT,GRID,SUBGRID,OFF):
    print(FILE_WEIGHT, ' | ', FILE_DEN)
    model.load_weights(FILE_WEIGHT)
    X_train = load_dataset(FILE_DEN,SUBGRID,OFF)
    
    Y_pred = model.predict(X_train,batch_size=1,verbose=2)

#---------------------------------------------------------
#  3D UNet, 64 filters may not be enough to encode all the gradients in the first layer. Increase as memory allows
#---------------------------------------------------------
def get_net(input_shape, kernel, model_name='U-Net'):
  inputs = Input(shape=input_shape, name="input_1")
  x = inputs

  strides_1 = (1,1,1)
  strides_2 = (2,2,2)

  # Encoding
  #--------------------------
  encode1a = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same', name='encode1a', strides=strides_1)(x)
  encode1b = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same', name='encode1b', strides=strides_1)(encode1a)
  pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool1')(encode1b)

  encode2a = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same', name='encode2a', strides=strides_1)(pool1)
  encode2b = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same', name='encode2b', strides=strides_1)(encode2a)
  pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool2')(encode2b)

  encode3a = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same', name='encode3a', strides=strides_1)(pool2)
  encode3b = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same', name='encode3b', strides=strides_1)(encode3a)
  pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool3')(encode3b)

  # Bottleneck
  #--------------------------
  bottom_a = Conv3D(filters=512, kernel_size=kernel, activation='relu', padding='same')(pool3)
  bottom_b = Conv3D(filters=512, kernel_size=kernel, activation='relu', padding='same')(bottom_a)

  # Decoding 
  #--------------------------
  up2   = Concatenate(axis=4)([Conv3DTranspose(filters=256, kernel_size=(2,2,2), strides=strides_2, padding='same')(bottom_b), encode3b])
  decode2a = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same')(up2)
  decode2b = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same')(decode2a)

  up3   = Concatenate(axis=4)([Conv3DTranspose(filters=128, kernel_size=(2,2,2), strides=strides_2, padding='same')(decode2b), encode2b])
  decode1a = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same')(up3)
  decode1b = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same')(decode1a)

  up4   = Concatenate(axis=4)([Conv3DTranspose(filters=64, kernel_size=(2,2,2), strides=strides_2, padding='same')(decode1b), encode1b])
  decode0a = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same')(up4)
  decode0b = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same')(decode0a)

  # Output
  flatten = Convolution3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid')(decode0b)

  model = Model(inputs=inputs, outputs=flatten,name=model_name)
  model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_coef_loss, metrics=cla_metrics_list)

  return model
#==================================================================================    

def get_net_onedeep(input_shape, kernel, model_name='U-Net_OneDeep'):
  inputs = Input(shape=input_shape, name="input_1")
  x = inputs

  strides_1 = (1,1,1)
  strides_2 = (2,2,2)

  # Encoding
  # --------------------------
  encode1a = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same', name='encode1a', strides=strides_1)(x)
  encode1b = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same', name='encode1b', strides=strides_1)(encode1a)
  pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool1')(encode1b)

  encode2a = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same', name='encode2a', strides=strides_1)(pool1)
  encode2b = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same', name='encode2b', strides=strides_1)(encode2a)
  pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool2')(encode2b)

  encode3a = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same', name='encode3a', strides=strides_1)(pool2)
  encode3b = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same', name='encode3b', strides=strides_1)(encode3a)
  pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool3')(encode3b)

  encode4a = Conv3D(filters=512, kernel_size=kernel, activation='relu', padding='same', name='encode4a', strides=strides_1)(pool3)
  encode4b = Conv3D(filters=512, kernel_size=kernel, activation='relu', padding='same', name='encode4b', strides=strides_1)(encode4a)
  pool4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool4')(encode4b)

  # Bottleneck 
  # --------------------------
  bottom_a = Conv3D(filters=1024, kernel_size=kernel, activation='relu', padding='same')(pool4)
  bottom_b = Conv3D(filters=1024, kernel_size=kernel, activation='relu', padding='same')(bottom_a)

  # Decoding
  #--------------------------
  up1   = Concatenate(axis=4)([Conv3DTranspose(filters=512, kernel_size=(2,2,2), strides=strides_2, padding='same')(bottom_b), encode4b])
  decode3a = Conv3D(filters=512, kernel_size=kernel, activation='relu', padding='same')(up1)
  decode3b = Conv3D(filters=512, kernel_size=kernel, activation='relu', padding='same')(decode3a)

  up2   = Concatenate(axis=4)([Conv3DTranspose(filters=256, kernel_size=(2,2,2), strides=strides_2, padding='same')(decode3b), encode3b])
  decode2a = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same')(up2)
  decode2b = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same')(decode2a)

  up3   = Concatenate(axis=4)([Conv3DTranspose(filters=128, kernel_size=(2,2,2), strides=strides_2, padding='same')(decode2b), encode2b])
  decode1a = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same')(up3)
  decode1b = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same')(decode1a)

  up4   = Concatenate(axis=4)([Conv3DTranspose(filters=64, kernel_size=(2,2,2), strides=strides_2, padding='same')(decode1b), encode1b])
  decode0a = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same')(up4)
  decode0b = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same')(decode0a)

  # Output
  #--------------------------
  flatten = Convolution3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid')(decode0b)

  model = Model(inputs=inputs, outputs=flatten,name=model_name)
  model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_coef_loss, metrics=cla_metrics_list)

  return model

#==================================================================================
def get_net_oneshallow(input_shape, kernel, model_name='U-Net'):
  inputs = Input(shape=input_shape, name="input_1")
  x = inputs

  strides_1 = (1,1,1)
  strides_2 = (2,2,2)

  # Encoding
  # --------------------------
  encode1a = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same', name='encode1a', strides=strides_1)(x)
  encode1b = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same', name='encode1b', strides=strides_1)(encode1a)
  pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool1')(encode1b)

  encode2a = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same', name='encode2a', strides=strides_1)(pool1)
  encode2b = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same', name='encode2b', strides=strides_1)(encode2a)
  pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool2')(encode2b)

  
  # Bottleneck 
  # --------------------------
  bottom_a = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same', name='encode3a', strides=strides_1)(pool2)
  bottom_b = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same', name='encode3b', strides=strides_1)(bottom_a)

  # Decoding
  #--------------------------
  up3   = Concatenate(axis=4)([Conv3DTranspose(filters=128, kernel_size=(2,2,2), strides=strides_2, padding='same')(bottom_b), encode2b])
  decode1a = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same')(up3)
  decode1b = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same')(decode1a)

  up4   = Concatenate(axis=4)([Conv3DTranspose(filters=64, kernel_size=(2,2,2), strides=strides_2, padding='same')(decode1b), encode1b])
  decode0a = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same')(up4)
  decode0b = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same')(decode0a)

  # Output
  #--------------------------
  flatten = Convolution3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid')(decode0b)

  model = Model(inputs=inputs, outputs=flatten,name=model_name)
  # model.name = model_name
  model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_coef_loss, metrics=cla_metrics_list)

  return model

#==================================================================================
def get_UNet(input_shape, kernel, model_name='U-Net', depth='3'):
  if depth == 2:
    model = get_net_oneshallow(input_shape,kernel,model_name)
  elif depth == 3:
    model = get_net(input_shape,kernel,model_name)
  elif depth == 4:
    model = get_net_onedeep(input_shape,kernel,model_name)
  else:
    print('Please give a value of 2 (one layer shallower from UNet), 3 (default UNet), or 4 (one layer deeper from UNet)')
  return model

#==================================================================================
#
#==================================================================================
def get_net_dist_transform(input_shape, kernel, model_name='DT-U-Net'):
    '''
    Same thing as get_net, but with a different loss function.
    '''
    inputs = Input(shape=input_shape, name="input_1")
    x = inputs

    strides_1 = (1,1,1)
    strides_2 = (2,2,2)

    # Encoding
    #--------------------------
    encode1a = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same', name='encode1a', strides=strides_1)(x)
    encode1b = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same', name='encode1b', strides=strides_1)(encode1a)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool1')(encode1b)

    encode2a = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same', name='encode2a', strides=strides_1)(pool1)
    encode2b = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same', name='encode2b', strides=strides_1)(encode2a)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool2')(encode2b)

    encode3a = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same', name='encode3a', strides=strides_1)(pool2)
    encode3b = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same', name='encode3b', strides=strides_1)(encode3a)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool3')(encode3b)

    # Bottleneck
    #--------------------------
    bottom_a = Conv3D(filters=512, kernel_size=kernel, activation='relu', padding='same')(pool3)
    bottom_b = Conv3D(filters=512, kernel_size=kernel, activation='relu', padding='same')(bottom_a)

    # Decoding 
    #--------------------------
    up2   = Concatenate(axis=4)([Conv3DTranspose(filters=256, kernel_size=(2,2,2), strides=strides_2, padding='same')(bottom_b), encode3b])
    decode2a = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same')(up2)
    decode2b = Conv3D(filters=256, kernel_size=kernel, activation='relu', padding='same')(decode2a)

    up3   = Concatenate(axis=4)([Conv3DTranspose(filters=128, kernel_size=(2,2,2), strides=strides_2, padding='same')(decode2b), encode2b])
    decode1a = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same')(up3)
    decode1b = Conv3D(filters=128, kernel_size=kernel, activation='relu', padding='same')(decode1a)

    up4   = Concatenate(axis=4)([Conv3DTranspose(filters=64, kernel_size=(2,2,2), strides=strides_2, padding='same')(decode1b), encode1b])
    decode0a = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same')(up4)
    decode0b = Conv3D(filters=64, kernel_size=kernel, activation='relu', padding='same')(decode0a)

    # Output
    #flatten = Convolution3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid')(decode0b)
    # trying different output activation functions
    flatten = Convolution3D(filters=1, kernel_size=(1, 1, 1), activation='linear')(decode0b)
    # Add dense neurons after flatten?
    #dense_1 = Dense(1024, activation='relu', name='dense_1')(flatten)
    #outputs = Dense(512, activation='relu', name ='output')(dense_1)

    model = Model(inputs=inputs, outputs=flatten, name=model_name)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='MeanSquaredError', 
                  metrics=reg_metrics_list)

    return model


# Might need to delete some of these imports but leave for now
import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

#from tqdm import tqdm_notebook

def build_model(input_layer, start_neurons):
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.1)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.25)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.25)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.25)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.25)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.25)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.25)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    output_layer = Conv2D(1, (1,1), padding="same", activation="linear")(uconv1)
    
    return output_layer