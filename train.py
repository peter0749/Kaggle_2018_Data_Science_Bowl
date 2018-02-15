
# coding: utf-8

# # Intro
# Hello! This rather quick and dirty kernel shows how to get started on segmenting nuclei using a neural network in Keras.
#
# The architecture used is the so-called [U-Net](https://arxiv.org/abs/1505.04597), which is very common for image segmentation problems such as this. I believe they also have a tendency to work quite well even on small datasets.
#
# Let's get started importing everything we need!

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import gc
import sys
import random
import warnings
import numpy as np

#seed = 42 # for reproduction
#random.seed(seed)
#np.random.seed(seed)

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

# Set some parameters
IMG_WIDTH = 448
IMG_HEIGHT = 448
IMG_CHANNELS = 3
TRAIN_PATH = '/hdd/dataset/nuclei_dataset/stage1_train/'
TEST_PATH = '/hdd/dataset/nuclei_dataset/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


# In[2]:


# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]


# # Get the data
# Let's first import all the images and associated masks. I downsample both the training and test images to keep things light and manageable, but we need to keep a record of the original sizes of the test images to upsample our predicted masks and create correct run-length encodings later on. There are definitely better ways to handle this, but it works fine for now!

# In[3]:


from skimage.measure import regionprops
import cv2, imutils
import math

# Get and resize train images and masks
X_train = []
Y_train = []
#lsum = 0
#msum = 0
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    X_train.append(img)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mark = np.zeros(img.shape[:2], dtype=np.uint8)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        cnts = cv2.findContours(mask_.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        M = cv2.moments(cnts[0])
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
        else:
            p = regionprops(mask_, cache=True)[0]
            cY, cX = p.centroid
        r = int(math.ceil( max( img.shape[0], img.shape[1] ) * .01 ))
        cv2.circle(mark, (int(cX), int(cY)), r, 255, -1)
        mask = np.maximum(mask, mask_)
    mark = cv2.GaussianBlur(mark, (3,3), 0)
    #lsum += np.sum(mask)
    #msum += np.sum(mark)
    Y_train.append([mask, mark])

# print(msum / (lsum+msum))

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append(img.shape[:2])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')


# Let's see if things look all right by drawing some random images and their associated masks.

# In[4]:


# Check if training data looks all right
ix = random.randint(0, len(train_ids))
plt.imshow(X_train[ix])
plt.savefig('01.png')
plt.imshow(np.squeeze(Y_train[ix][0]), cmap='gray')
plt.savefig('02.png')
x_copy = X_train[ix].copy()
x_copy[np.squeeze(Y_train[ix][1])>0, :] = 255, 0, 0
plt.imshow(x_copy)
plt.savefig('03.png')


# Seems good!
#
# # Create our Keras metric
#
# Now we try to define the *mean average precision at different intersection over union (IoU) thresholds* metric in Keras. TensorFlow has a mean IoU metric, but it doesn't have any native support for the mean over multiple thresholds, so I tried to implement this. **I'm by no means certain that this implementation is correct, though!** Any assistance in verifying this would be most welcome!
#
# *Update: This implementation is most definitely not correct due to the very large discrepancy between the results reported here and the LB results. It also seems to just increase over time no matter what when you train ... *

# In[5]:


# Define IoU metric
def mean_iou(y_true_, y_pred_):
    y_true = y_true_[...,0]
    y_pred = y_pred_[...,0]
    prec = []
    for t in np.arange(0.5, 1.0, 0.1):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def mean_iou_marker(y_true_, y_pred_):
    y_true = y_true_[...,1]
    y_pred = y_pred_[...,1]
    prec = []
    for t in np.arange(0.5, 1.0, 0.1):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + intersection + smooth)

from keras.losses import binary_crossentropy

def custom_loss(y_true_, y_pred_):
    alpha = 0.83 # control amount of loss of markers

    y_true_hm = y_true_[...,0]
    y_true_mk = y_true_[...,1]

    y_pred_hm = y_pred_[...,0]
    y_pred_mk = y_pred_[...,1]

    s_marker_loss = .5 * binary_crossentropy(y_true_mk, y_pred_mk) - dice_coef(y_true_mk, y_pred_mk)
    s_heatmap_loss= .5 * binary_crossentropy(y_true_hm, y_pred_hm) - dice_coef(y_true_hm, y_pred_hm)

    losses = alpha * s_marker_loss + (1.-alpha) * s_heatmap_loss
    return losses


# # Build and train our neural network
# Next we build our U-Net model, loosely based on [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf) and very similar to [this repo](https://github.com/jocicmarko/ultrasound-nerve-segmentation) from the Kaggle Ultrasound Nerve Segmentation competition.
#
# ![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

# In[6]:


from keras.layers import Lambda, Add, Activation, UpSampling2D, Dropout, BatchNormalization
from keras.optimizers import Adam

def build_stage(inputs, last=None, id_='st1'):
    def conv(f, k=3, act='elu'):
        return Conv2D(f, (k, k), activation=act, kernel_initializer='he_normal', padding='same')
    def _incept_conv(input_shape, f, dropout=0, chs=[0.15, 0.5, 0.25, 0.1]):
        inputs = Input(shape=input_shape)
        fs = [] # determine channel number
        for k in chs:
            t = max(int(k*f), 1) # at least 1 channel
            fs.append(t)

        fs[1] += f-np.sum(fs) # reminding channels allocate to 3x3 conv

        c1x1 = conv(fs[0], 1, act=None) (inputs)
        c3x3 = conv(max(1, fs[1]//2), 1, act='elu') (inputs)
        c5x5 = conv(max(1, fs[2]//2), 1, act='elu') (inputs)
        cpool= MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same') (inputs)

        c3x3 = conv(fs[1], 3, act=None) (c3x3)
        c5x5 = conv(fs[2], 5, act=None) (c5x5)
        cpool= conv(fs[3], 1, act=None) (cpool)

        output = concatenate([c1x1, c3x3, c5x5, cpool], axis=-1)
        if dropout>0:
            output = Dropout(dropout) (output)
        return Model(inputs, output)

    def _res_conv(f, k=3, dropout=0, bn=True): # very simple residual module
        def block(inputs):

            channels = int(inputs.shape[-1])
            cols = int(inputs.shape[-2])
            rows = int(inputs.shape[-3])

            # here: mpoly-2 in polynet
            f_inception_shared = _incept_conv((rows, cols, channels), f, dropout=dropout)
            g_inception_shared = _incept_conv((rows, cols, f), f, dropout=dropout)

            f0 = f_inception_shared (inputs) # f
            g1 = g_inception_shared (f0) # f*g

            if f!=channels:
                t1 = conv(f, 1, None) (inputs) # identity mapping
            else:
                t1 = inputs

            out = Add()([t1, f0, g1]) # t1 + f + g
            if bn:
                out = BatchNormalization() (out)
            out = Activation('elu') (out)
            return out
        return block
    def pool():
        return MaxPooling2D((2, 2))
    def up(inputs, dropout=0):
        upsampled = Conv2DTranspose(int(inputs.shape[-1]), (2, 2), strides=(2, 2), padding='same') (inputs)
        if dropout>0:
            upsampled = Dropout(dropout) (upsampled)
        return upsampled

    if last is None:
        c1 = Lambda(lambda x: x / 255) (inputs) # 1st stage input, an image
    else:
        c1 = concatenate([inputs, last], axis=-1)

    c1 = _res_conv(32, 3) (c1)
    #c1 = _res_conv(32, 3) (c1)
    o1 = c1
    p1 = pool() (c1)

    c2 = _res_conv(64, 3) (p1)
    #c2 = _res_conv(64, 3) (c2)
    p2 = pool() (c2)

    c3 = _res_conv(128, 3) (p2)
    #c3 = _res_conv(128, 3) (c3)
    p3 = pool() (c3)

    c4 = _res_conv(256, 3) (p3)
    #c4 = _res_conv(256, 3) (c4)
    p4 = pool() (c4)

    c5 = _res_conv(512, 3) (p4)
    #c5 = _res_conv(512, 3) (c5)
    p5 = pool() (c5)

    c6 = _res_conv(1024, 3) (p5)
    #c6 = _res_conv(1024, 3) (c6)

    u7 = up (c6)
    c7 = concatenate([u7, c5])
    c7 = _res_conv(512, 3) (c7)
    #c7 = _res_conv(512, 3) (c7)

    u8 = up (c7)
    c8 = concatenate([u8, c4])
    c8 = _res_conv(256, 3) (c8)
    #c8 = _res_conv(256, 3) (c8)

    u9 = up (c8)
    c9 = concatenate([u9, c3])
    c9 = _res_conv(128, 3) (c9)
    #c9 = _res_conv(128, 3) (c9)

    u10 = up (c9)
    c10 = concatenate([u10, c2])
    c10 = _res_conv(64, 3) (c10)
    #c10 = _res_conv(64, 3) (c10)

    u11 = up (c10)
    c11 = concatenate([u11, c1])
    c11 = _res_conv(32, 3) (c11)
    #c11 = _res_conv(32, 3) (c11)

    outputs = Conv2D(2, (1, 1), activation=None) (c11)
    outputs = BatchNormalization() (outputs)
    outputs = Activation('sigmoid', name=id_+'_out') (outputs)
    return outputs, o1

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
out, _ = build_stage(inputs, None, 'st1')

with tf.device("/cpu:0"):
    cpu_model = Model(inputs=[inputs], outputs=[out])
    cpu_model.summary()


# *Update: Changed to ELU units, added dropout.*
#
# Next we fit the model on the training data, using a validation split of 0.1. We use a small batch size because we have so little data. I recommend using checkpointing and early stopping when training your model. I won't do it here to make things a bit more reproducible (although it's very likely that your results will be different anyway). I'll just train for 10 epochs, which takes around 10 minutes in the Kaggle kernel with the current parameters.
#
# *Update: Added early stopping and checkpointing and increased to 30 epochs.*

# In[7]:


from keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.utils import shuffle
from keras.utils import Sequence

class data_generator(Sequence):
    def __init__(self, data, label, batch_size=4, training=True):
        self.data = data
        self.label= label
        self.batch_size = batch_size
        self.training = training
    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)
    def __getitem__(self, index):
        base = index*self.batch_size
        limit= min(len(self.data), (index+1)*self.batch_size)
        batch_index = range(base, limit)
        dat_que = np.empty((limit-base, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        lab_que = np.empty((limit-base, IMG_HEIGHT, IMG_WIDTH, 2), dtype=np.bool)
        for i, index in enumerate(batch_index):
            img, lab, marker = self.fetch_data(index)
            dat_que[i,:,:,:] = img
            lab_que[i,:,:,0] = lab
            lab_que[i,:,:,1] = marker
        return dat_que, lab_que

    def fetch_data(self, index):
        img = self.data[index] # [0,255]
        lab = self.label[index][0]
        marker = self.label[index][1]
        if self.training:
            if np.random.rand() < .5: # flip vertical
                img = np.flip(img, 0)
                lab = np.flip(lab, 0)
                marker = np.flip(marker, 0)
            if np.random.rand() < .5: # flip horizontal
                img = np.flip(img, 1)
                lab = np.flip(lab, 1)
                marker = np.flip(marker, 1)

            # rotation, shearing
            if np.random.rand() < 0.5:
                y, x, _ = img.shape
                h, w = y, x
                img = cv2.copyMakeBorder(img, y//2, y//2, x//2, x//2, cv2.BORDER_REFLECT)
                lab = cv2.copyMakeBorder(lab, y//2, y//2, x//2, x//2, cv2.BORDER_REFLECT)
                marker = cv2.copyMakeBorder(marker, y//2, y//2, x//2, x//2, cv2.BORDER_REFLECT)
                y, x, _ = img.shape

                pts1 = np.float32([[5,5],[20,5],[5,20]])

                pt1 =    10*np.random.uniform()
                pt2 = 15+10*np.random.uniform()

                pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
                shear_M = cv2.getAffineTransform(pts1,pts2)

                rotT = np.random.uniform(-45,45)
                M = cv2.getRotationMatrix2D((x/2, y/2), rotT, 1)

                mA, mB = (M, shear_M) if np.random.rand()<.5 else (shear_M, M)

                img = cv2.warpAffine(img, mA, (x, y))
                lab = cv2.warpAffine(lab, mA, (x, y))
                marker = cv2.warpAffine(marker, mA, (x, y))

                img = cv2.warpAffine(img, mB, (x, y))
                lab = cv2.warpAffine(lab, mB, (x, y))
                marker = cv2.warpAffine(marker, mB, (x, y))

                img = img[h//2:h//2+h,w//2:w//2+w,:]
                lab = lab[h//2:h//2+h,w//2:w//2+w]
                marker = marker[h//2:h//2+h,w//2:w//2+w]

            img = img.astype(np.float32) / 255. # normalize

            # random amplify each channel
            a = .1 # amptitude
            t  = [np.random.uniform(-a,a)]
            t += [np.random.uniform(-a,a)]
            t += [np.random.uniform(-a,a)]
            t = np.array(t)

            img = img * (1. + t) # channel wise amplify
            up = np.random.uniform(0.95, 1.05) # change gamma
            img = img**up * 255. # apply gamma and convert back to range [0,255]
            img = img.astype(np.uint8) # convert back to uint8

            assert img.dtype == np.uint8 ## check type
            assert lab.dtype == np.uint8

            if np.random.rand() < .1: # randomly crop image
                s = np.random.uniform(0.8, 1.2) # random crop scale
                w, h = int(s*IMG_WIDTH), int(s*IMG_HEIGHT)
                paddx = max(0, w-img.shape[1])
                paddy = max(0, h-img.shape[0])
                img = cv2.copyMakeBorder(img, paddy//2, paddy-paddy//2, paddx//2, paddx-paddx//2, cv2.BORDER_REFLECT)
                lab = cv2.copyMakeBorder(lab, paddy//2, paddy-paddy//2, paddx//2, paddx-paddx//2, cv2.BORDER_REFLECT)
                marker = cv2.copyMakeBorder(marker, paddy//2, paddy-paddy//2, paddx//2, paddx-paddx//2, cv2.BORDER_REFLECT)
                cropx = img.shape[1] - w + 1
                cropy = img.shape[0] - h + 1
                offx = np.random.randint(0, cropx)
                offy = np.random.randint(0, cropy)
                img = img[offy:offy+h, offx:offx+w, :]
                lab = lab[offy:offy+h, offx:offx+w]
                marker = marker[offy:offy+h, offx:offx+w]
        ### end of data augmentation ###

        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        lab = resize(lab, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        marker = resize(marker, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

        lab[lab<130] = 0
        lab[lab>0] = 1

        marker[marker<130] = 0
        marker[marker>0] = 1

        return img, lab, marker

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, shuffle=True)

# Fit model
BS=10
EPOCHS=300

# earlystopper = EarlyStopping(patience=7, verbose=1)
# checkpointer = ModelCheckpoint('./models/model.{epoch:03d}.vl.{val_loss:.2f}.vi.{val_mean_iou:.2f}.vim.{val_mean_iou_marker:.2f}.h5', verbose=1, save_best_only=False)

from keras.utils.training_utils import multi_gpu_model
model = multi_gpu_model(cpu_model, gpus=2)
model.summary()
model.compile(loss=custom_loss, metrics=[mean_iou, mean_iou_marker], optimizer='adam')

from keras.callbacks import Callback
class CKPT(Callback):
    def __init__(self, path):
        self.path  = path
        self.best  = np.inf
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best:
            save_path = self.path
            sys.stderr.write('loss improved from %.4f to %.4f, saving model to %s\n'%(self.best, val_loss, save_path))
            self.model.save_weights(save_path)
            self.best = val_loss

best_weights_path = './top_weights.h5'

model.fit_generator(data_generator(X_train, Y_train, batch_size=BS, training=True),
                              steps_per_epoch=len(X_train) // BS , epochs=EPOCHS,
                              validation_data=data_generator(X_val, Y_val, batch_size=BS, training=False),
                              validation_steps=len(X_val) // BS,
                              callbacks=[CKPT(best_weights_path)], shuffle=True, workers=6)
del model
gc.collect()

with tf.device("/cpu:0"):
    cpu_model.compile(loss=custom_loss, metrics=[mean_iou, mean_iou_marker], optimizer='adam')
    cpu_model.load_weights(best_weights_path, by_name=True)
    cpu_model.save('model.h5')
del cpu_model
gc.collect()

from skimage.morphology import closing, square, remove_small_objects
from skimage.segmentation import clear_border

# Predict on train, val and test
model = load_model('./model.h5', custom_objects={'mean_iou': mean_iou, 'custom_loss': custom_loss, 'mean_iou_marker': mean_iou_marker, 'dice_coef': dice_coef})
preds_train = model.predict_generator(data_generator(X_train, Y_train, batch_size=1, training=False), steps=len(X_train), verbose=1)
preds_val = model.predict_generator(data_generator(X_val, Y_val, batch_size=1, training=False), steps=len(X_val), verbose=1)
preds_test = model.predict(X_test, verbose=1, batch_size=1)

preds_train, preds_train_marker = preds_train[...,0], preds_train[...,1]
preds_val, preds_val_marker = preds_val[...,0], preds_val[...,1]
preds_test, preds_test_marker = preds_test[...,0], preds_test[...,1]

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_train_marker_t = (preds_train_marker > 0.5).astype(np.uint8)
preds_val_marker_t = (preds_val_marker > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append((resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True),
                                 resize(np.squeeze(preds_test_marker[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True)
                                ))


# In[ ]:


from scipy import ndimage as ndi
from skimage.morphology import watershed

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def lb(image, marker):
    if np.sum(image) < np.sum(marker):
        image = marker
    else:
        marker = np.array((marker==1) & (image==1))
    distance = ndi.distance_transform_edt(image)
    markers = ndi.label(marker)[0]
    labels = watershed(-distance, markers, mask=image)
    if np.sum(labels) == 0:
        labels[0,0] = 1
    return labels

def prob_to_rles(x, marker, cutoff=0.5, cutoff_marker=0.5):
    lab_img = lb(x > cutoff, marker > cutoff_marker)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


# In[ ]:


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
print(ix)
shape = Y_train[ix][0].shape[:2]
plt.imshow(X_train[ix])
plt.savefig('10.png')
plt.imshow(np.squeeze(Y_train[ix][0]))
plt.savefig('11.png')
plt.imshow(resize(np.squeeze(preds_train_t[ix]), shape))
plt.savefig('12.png')
plt.imshow(np.squeeze(Y_train[ix][1]))
plt.savefig('13.png')
plt.imshow(resize(np.squeeze(preds_train_marker_t[ix]), shape))
plt.savefig('14.png')
lab = lb(preds_train_t[ix], preds_train_marker_t[ix])
plt.imshow(lab)
plt.savefig('15.png')


# The model is at least able to fit to the training data! Certainly a lot of room for improvement even here, but a decent start. How about the validation data?

# In[ ]:


# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
print(ix)
shape = Y_val[ix][0].shape[:2]
plt.imshow(X_val[ix])
plt.savefig('16.png')
plt.imshow(np.squeeze(Y_val[ix][0]))
plt.savefig('17.png')
plt.imshow(resize(np.squeeze(preds_val_t[ix]), shape))
plt.savefig('18.png')
plt.imshow(np.squeeze(Y_val[ix][1]))
plt.savefig('19.png')
plt.imshow(resize(np.squeeze(preds_val_marker_t[ix]), shape))
plt.savefig('20.png')
lab = lb(preds_val_t[ix], preds_val_marker_t[ix])
plt.imshow(lab)
plt.savefig('21.png')


# Not too shabby! Definitely needs some more training and tweaking.
#
# # Encode and submit our results
#
# Now it's time to submit our results. I've stolen [this](https://www.kaggle.com/rakhlin/fast-run-length-encoding-python) excellent implementation of run-length encoding.

# Let's iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage ...

# In[ ]:


new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n][0], preds_test_upsampled[n][1]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))


# ... and then finally create our submission!

# In[ ]:


# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)


# This scored 0.233 on the LB for me. That was with version 2 of this notebook; be aware that the results from the neural network are extremely erratic and vary greatly from run to run (version 3 is significantly worse, for example). Version 7 scores 0.277!
#
# You should easily be able to stabilize and improve the results just by changing a few parameters, tweaking the architecture a little bit and training longer with early stopping.
#
# **Have fun!**
#
# LB score history:
# - Version 7: 0.277 LB

# In[ ]:

