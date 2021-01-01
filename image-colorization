import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation,Concatenate, MaxPool2D,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, History,EarlyStopping
import os, tarfile, shutil, pickle
from tensorflow.keras import Input,losses
from matplotlib import pyplot as plt
import sys, os, urllib.request, tarfile, glob
import cv2

#データセットをダウンロード
class STL10:
  def __init__(self, download_dir):
    self.binary_dir = os.path.join(download_dir, "stl10_binary")
    if not os.path.exists(download_dir):
      os.mkdir(download_dir)
    if not os.path.exists(self.binary_dir):
      os.mkdir(self.binary_dir)

    def _progress(count, block_size, total_size):
      sys.stdout.write('\rDownloading %s %.2f%%' % (source_path,float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
      
    source_path = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    dest_path = os.path.join(download_dir, "stl10_binary.tar.gz")

    if not os.path.exists(dest_path):
      urllib.request.urlretrieve(source_path, filename=dest_path,reporthook=_progress)
      with tarfile.open(dest_path, "r:gz") as tar:
        tar.extractall(path=download_dir)

  def get_files(self, target):
    assert target in ["train", "test", "unlabeled"]
    if target in ["train", "test"]:
      images = self.load_images(os.path.join(self.binary_dir, target+"_X.bin"))
      labels = self.load_labels(os.path.join(self.binary_dir, target+"_y.bin"))
    else:
      images = self.load_images(os.path.join(self.binary_dir, target+"_X.bin"))
      labels = None

    return images, labels

    def load_images(self, image_binary):
      with open(image_binary, "rb") as fp:
        images = np.fromfile(fp, dtype=np.uint8)
        images = images.reshape(-1, 3, 96, 96)

      return np.transpose(images, (0, 3, 2, 1))

    def load_labels(self, label_binary):
      with open(label_binary) as fp:
        labels = np.fromfile(fp, dtype=np.uint8)
        
      return labels.reshape(-1, 1) - 1

    def get_one_label(self,target,label):
      X_images,y_labels = self.get_files(target)
      flag = 0
      for i in range(X_images.shape[0]):
        if flag == 0:
          if y_labels[i] == label:
            images = np.expand_dims(X_images[i],axis=0)
            flag = flag+1
        elif y_labels[i] == label:
          X_temp = np.expand_dims(X_images[i],axis=0)
          images = np.concatenate([images,X_temp])
      return images

#モデル
def model_conv2d(input,ch):
  x = input
  for i in range(2):
    x = Conv2D(ch,3,padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
  return x
  
def unet(output_layer):
    input = Input((96,96,1))
    block1_conv = model_conv2d(input,64)
    block1_pool = MaxPool2D(2)(block1_conv)
    block2_conv = model_conv2d(block1_pool,128)
    block2_pool = MaxPool2D(2)(block2_conv)
    block3_conv = model_conv2d(block2_pool,256)
    block3_pool = MaxPool2D(2)(block3_conv)
    block4_conv = model_conv2d(block3_pool,512)
    block4_pool = MaxPool2D(2)(block4_conv)

    block5_conv = model_conv2d(block4_pool,1024)
    block5_up = UpSampling2D(size=(2,2))(block5_conv)
    
    block5_up = Concatenate()([block4_conv,block5_up])
    block6_conv = model_conv2d(block5_up,512)
    block6_up = UpSampling2D(size=(2,2))(block6_conv)
    block6_up = Concatenate()([block3_conv,block6_up])
    block7_conv = model_conv2d(block6_up,256)
    block7_up = UpSampling2D(size=(2,2))(block7_conv)
    block7_up = Concatenate()([block2_conv,block7_up])
    block8_conv = model_conv2d(block7_up,128)
    block8_up = UpSampling2D(size=(2,2))(block8_conv)
    block8_up = Concatenate()([block1_conv,block8_up])
    block9_conv1 = model_conv2d(block8_up,64)
    block9_conv2 = Conv2D(output_layer,1)(block9_conv1)
    output = Activation("sigmoid")(block9_conv2)

    return Model(input,output)

#色空間の変換
def Lab(img,num):
  for i in range(num):
    img[i] = cv2.cvtColor(img[i],cv2.COLOR_RGB2Lab)
  return img

def RGB(img,num):
  for i in range(num):
    img[i] = cv2.cvtColor(img[i],cv2.COLOR_Lab2RGB)
  return img

def gray(img):
    return np.expand_dims(img[:,:,:,0]*0.299 + img[:,:,:,1]*0.587 + img[:,:,:,2]*0.114, axis=-1)

#Lab色空間での予測
def Lab_coloring():
  stl10 = STL10("./stl10")
  train_RGB = stl10.get_one_label("train",0)
  test_RGB = stl10.get_one_label("test",0)
  np.random.shuffle(train_RGB)
  np.random.shuffle(test_RGB)
  train_RGB = np.concatenate([train_RGB,test_RGB[:700,:,:,:]])
  test_RGB = test_RGB[700:,:,:,:]
  model = unet(2)
  model.compile(optimizer="Adam",loss=losses.MSE)
  for i in range(10):
    train_input = gray(train_RGB[100*i:100*(i+1)]/255.0).astype(np.float32)
    train_ans = (Lab((train_RGB[100*i:100*(i+1)]/255.0).astype(np.float32),100))[:,:,:,1:]
    train_ans = (train_ans+128.0)/255.0
    history = model.fit(train_input,train_ans,batch_size=32,epochs=100)

    test_input = gray(test_RGB[10*i:10*(i+1)]/255.0).astype(np.float32)
    test_output = model.predict(test_input)
    test_output = ((test_output*255.0)-128.0).astype(np.float32)
    test_ans = np.zeros(96*96*10*3).reshape(10,96,96,3).astype(np.float32)
    test_ans[:,:,:,0] = test_input[:,:,:,0]*100
    test_ans[:,:,:,1:] = test_output
    result = RGB(test_ans,10)
    result = (result*255).astype(np.uint8)
    for j in range(10):
      plt.imshow(result[j])
      plt.show()
      plt.imshow(test_RGB[10*i+j])
      plt.show()

#RGB色空間での予測
def RGB_coloring():
  stl10 = STL10("./stl10")
  train_RGB = stl10.get_one_label("train",0)
  test_RGB = stl10.get_one_label("test",0)
  np.random.shuffle(train_RGB)
  np.random.shuffle(test_RGB)
  train_RGB = np.concatenate([train_RGB,test_RGB[:700,:,:,:]])
  test_RGB = test_RGB[700:,:,:,:]
  model = unet(3)
  model.compile(optimizer="Adam",loss=losses.MSE)
  for i in range(10):
    train_input = gray(train_RGB[100*i:100*(i+1)]/255.0).astype(np.float32)
    train_ans = (train_RGB[100*i:100*(i+1)]/255.0).astype(np.float32)
    history = model.fit(train_input,train_ans,batch_size=32,epochs=100)
    test_input = gray(test_RGB[10*i:10*(i+1)]/255.0).astype(np.float32)
    test_output = model.predict(test_input)
    result = (test_output*255).astype(np.uint8)
    for j in range(10):
      plt.imshow(result[j])
      plt.show()
      plt.imshow(test_RGB[10*i+j])
      plt.show()

