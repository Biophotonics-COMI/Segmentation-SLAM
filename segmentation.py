from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math
import numpy as np
import os
import random
import time

import os.path
import tensorflow as tf
import skimage
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import rotate
from skimage.transform import rescale
from operator import itemgetter
from pathlib import Path

FLAGS = None

#ous=384;
#ins=384;
ous = 128*1 # network output size
ins = 128*1 # network input size (random cropping size during the training)
interv = 20 # difference of gray value in mask labeling
nclass = 9
batch_size = 32
iterModel=0
iterMax=100001-iterModel
print('iterMax:'+str(iterMax))
parameters = 'CellSegType9_train10_Batch32'
homedir = '/home/syou10/Segmentation/20191209_cell_seg'
#resultdir = homedir + '20180627 applied results'
imgdir = homedir

logdir = '/home/syou10/trainLog' + parameters
testdir = homedir + '/AllRat/'
validationdir = homedir + '/Composite/'
traindir = homedir + '/Gray_mask/'
resultModeldir = homedir + '/resultsModel' + parameters
resultImagedir = homedir + '/resultsImage' + parameters
resultApplydir = homedir + '/resultsApply' + parameters
imageType = 4 # channel
nc = 32

class AMSGradOptimizer(tf.train.Optimizer):
  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    
    self.t = 0
    self.m = {}
    self.v = {}
    self.v_hat = {}
    self.beta1_t = tf.Variable(1.0, trainable=False)
    self.beta2_t = tf.Variable(1.0, trainable=False)
    
    for var in tf.trainable_variables():
      self.m[var] = tf.Variable(tf.zeros(tf.shape(var.initial_value)), trainable=False)
      self.v[var] = tf.Variable(tf.zeros(tf.shape(var.initial_value)), trainable=False)
      self.v_hat[var] = tf.Variable(tf.zeros(tf.shape(var.initial_value)), trainable=False)
    
  def apply_gradients(self, gradient_variables):
    beta1_t = self.beta1_t.assign(self.beta1_t * self.beta1)
    beta2_t = self.beta2_t.assign(self.beta2_t * self.beta2)
    update_ops = []
    
    cur_learning_rate = self.learning_rate * tf.sqrt(1 - beta2_t) / (1 - beta1_t)
    tf.summary.scalar('True_lr',cur_learning_rate)
    for idx, var in enumerate(tf.trainable_variables()):
      g = gradient_variables[idx]
      m = self.m[var].assign(self.beta1 * self.m[var] + (1 - self.beta1) * g)
      v = self.v[var].assign(self.beta2 * self.v[var] + (1 - self.beta2) * g * g)
      v_hat = self.v_hat[var].assign(tf.maximum(self.v_hat[var], v))
      
      update = -cur_learning_rate * m / (self.epsilon + tf.sqrt(v_hat))
      update_ops.append(var.assign_add(update))
    
    return tf.group(*update_ops)

def load_data(file_dir):
  train_imgs = []
  label_imgs = []
  count = 0
  print('Total Number of gray mask');
  print(len([name for name in os.listdir(traindir) if os.path.isfile(os.path.join(traindir, name))]))
  for file in os.listdir(traindir):
#  for file in os.listdir(os.path.join(file_dir,'Gray_mask/')):
    if file.endswith(".png"):
      imgfn = file[:-4]
      imgfn1 = imgfn+'_2PF.png'
      imgfn2 = imgfn+'_3PF.png'
      imgfn3 = imgfn+'_SHG.png'
      imgfn4 = imgfn+'_THG.png'
      lI=imread(os.path.join(traindir,file))
      print(imgfn)
      print('label check: ', np.max(lI))
#      if np.max(lI)==150:
 #       exit()
        
#      lI=imread(os.path.join(file_dir,'Gray mask/',file))
      lI=skimage.img_as_float(lI)
      label_imgs.append(lI)
      tI=np.zeros((lI.shape[0],lI.shape[1],imageType))

      tI1 = skimage.img_as_float(imread(os.path.join(testdir, imgfn1)))
      tI2 = skimage.img_as_float(imread(os.path.join(testdir, imgfn2)))
      tI3 = skimage.img_as_float(imread(os.path.join(testdir, imgfn3)))
      tI4 = skimage.img_as_float(imread(os.path.join(testdir, imgfn4)))

#      tI1=skimage.img_as_float(imread(os.path.join(file_dir,'8 bit png/',imgfn1)))
#      tI2=skimage.img_as_float(imread(os.path.join(file_dir,'8 bit png/',imgfn2)))
#      tI3=skimage.img_as_float(imread(os.path.join(file_dir,'8 bit png/',imgfn3)))
#      tI4=skimage.img_as_float(imread(os.path.join(file_dir,'8 bit png/',imgfn4)))

      
      print(tI1.shape)
      print(tI2.shape)
      print(tI3.shape)
      print(tI4.shape)
      tI[:,:,0] = tI1
      tI[:,:,1] = tI2
      tI[:,:,2] = tI3
      tI[:,:,3] = tI4
      
      train_imgs.append(tI)
      
  return train_imgs, label_imgs

def load_test(file_dir):
  test_imgs = []
  for file in os.listdir(validationdir):
#  for file in os.listdir(os.path.join(file_dir,'8 bit png/')):
    if file.endswith(".png"):
      tI=imread(os.path.join(test_dir,file))
#      tI=imread(os.path.join(file_dir,'8 bit png/',file))
      tI=skimage.img_as_float(tI)
      #tI=rescale(tI,0.5,order=0)
      if (len(tI.shape)==2):
        tI=np.expand_dims(tI,axis=2)
      test_imgs.append(tI)
  return test_imgs

def get_image(train_imgs,label_imgs):
  i_idx = random.randint(0, len(train_imgs)-1)
  big_tI = train_imgs[i_idx]
  big_lI = label_imgs[i_idx]
  xs = big_tI.shape[0]
  ys = big_tI.shape[1]
  stx = random.randint(0,xs-ins)
  sty = random.randint(0,ys-ins)
  train_sample = np.zeros((ins,ins,imageType))
  label_sample = np.zeros((ins,ins))
  train_sample = big_tI[stx:stx+ins,sty:sty+ins,:]
  label_sample = big_lI[stx:stx+ins,sty:sty+ins]
  nrotate = random.randint(0, 3)
  train_sample = np.rot90(train_sample, nrotate)
  label_sample = np.round(np.rot90(label_sample, nrotate)*255).astype('uint8')
  nflip = random.randint(0, 1)
  if nflip:
    #print('flip')
    train_sample = np.fliplr(train_sample)
    label_sample = np.fliplr(label_sample)
  gstd = np.random.uniform(low=0.6,high=1.0)
  #print('before:',train_sample.max())
  train_sample = train_sample * gstd
  #print('after:',train_sample.max())
  return train_sample, label_sample

def get_test_image(test_imgs):
  i_idx = random.randint(0, len(test_imgs)-1)
  big_tI = test_imgs[i_idx]
  xs = big_tI.shape[0]
  ys = big_tI.shape[1]
  stx = random.randint(0,xs-ins)
  sty = random.randint(0,ys-ins)
  test_sample = np.zeros((ins,ins,imageType))
  label_sample = np.zeros((ins,ins))
  test_sample = big_tI[stx:stx+ins,sty:sty+ins,:]
  nrotate = random.randint(0, 3)
  test_sample = np.rot90(test_sample, nrotate)
  label_sample = np.round(np.rot90(label_sample, nrotate)*255).astype('uint8')
  nflip = random.randint(0, 1)
  if nflip:
    #print('flip')
    test_sample = np.fliplr(test_sample)
    label_sample = np.fliplr(label_sample)
  return test_sample, label_sample

def get_batch(train_imgs,label_imgs,batch_size):
  train_samples = np.zeros((batch_size,ins,ins,imageType))
  label_samples = np.zeros((batch_size,ins,ins))
  for i in range(batch_size):
    train_sample, label_sample = get_image(train_imgs,label_imgs)
    while (label_sample.max()==0):
      train_sample, label_sample = get_image(train_imgs,label_imgs)
    train_samples[i,:,:,:] = train_sample
    label_samples[i,:,:] = label_sample/interv
  return train_samples, label_samples.astype('int32')

def get_test_batch(test_imgs,batch_size):
  test_samples = np.zeros((batch_size,ins,ins,imageType))
  label_samples = np.zeros((batch_size,ins,ins))
  for i in range(batch_size):
    test_sample, label_sample = get_test_image(test_imgs)
    test_samples[i,:,:,:] = test_sample
    label_samples[i,:,:] = 2
  return test_samples, label_samples.astype('int32')

def weight_variable(shape,stdv):
  initial = tf.get_variable("weights", shape=shape, initializer=tf.random_normal_initializer(stddev=stdv))
  return initial

def bias_variable(shape):
  initial = tf.get_variable("bias", shape=shape, initializer=tf.constant_initializer(value=0.0))
  return initial

def conv_(x,k,nin,nout,phase,s=1,d=1):
  stdv = math.sqrt(2/(nin*k*k))
  return tf.layers.conv2d(x, nout, k, strides=[s,s], dilation_rate=[d,d], padding='same',
                                      kernel_initializer = tf.random_normal_initializer(stddev=stdv),
                                      use_bias=False)

def convnn_(x,k,s,nin,nout,phase):
  print('convnn')
  stdv = math.sqrt(1/(nin*k*k))
  return tf.layers.conv2d(x, nout, k, strides=[s,s], padding='same',
                                      kernel_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv,dtype=tf.float32),
                                      use_bias=False)

def bn_(x,phase):
  bn_result = tf.layers.batch_normalization(x, momentum=0.9, epsilon = 1e-5,
                                            gamma_initializer = tf.random_uniform_initializer(minval=0.0, maxval=1.0,dtype=tf.float32),
                                            training = phase)
  return bn_result

def relu_(x):
  return tf.nn.relu(x)

def cbr_(x,k,nin,nout,phase,s=1,d=1):
  x_conv = conv_(x,k,nin,nout,phase,s,d)
  x_bn = bn_(x_conv,phase)
  x_relu = relu_(x_bn)
  return x_relu

def bottleneck(x, nin, nout, phase, s=1, d=1):
  if (nin != nout) or (s != 1):
    print('conv_skip')
    with tf.variable_scope('skip'):
      skip_conv = conv_(x,1,nin,nout,phase,s=s)
      skip = bn_(skip_conv,phase)
  else:
    skip = x

  with tf.variable_scope('conv1'):
    c1_conv = conv_(x,3,nin,nout,phase,s=s,d=d)
    c1_bn = bn_(c1_conv,phase)
    c1_relu = relu_(c1_bn)
    
  with tf.variable_scope('conv2'):
    c2_conv = conv_(c1_relu,3,nout,nout,phase,d=d)
    c2_bn = bn_(c2_conv,phase)

  out = skip + c2_bn
  out_relu = relu_(out)
  return out_relu

def stack(x, nin, nout, nblock, phase, s=1, d=1, new_level=True):
  for i in range(nblock):
    with tf.variable_scope('block%d' % (i)):
      if i==0:
        x = bottleneck(x,nin,nout,phase,s=s,d=d)
      else:
        x = bottleneck(x,nout,nout,phase,d=d)
  return x

def max_pool_2x2(x):
  print('new pool')
  return tf.layers.max_pooling2d(x,[2,2],[2,2])

def avg_pool_2x2(x):
  print('avg')
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def show(x,k):
  x_sm = tf.nn.softmax(x)
  x_sl = tf.unstack(x_sm,num=nclass,axis=-1)[k]
  return tf.expand_dims(x_sl,-1)

def deconv_(x, nin, nout, phase):
  stdv = math.sqrt(2/(nin*2*2))
  conv_r = tf.layers.conv2d_transpose(x, nout, 4, strides=[2,2], padding='same',
                                      kernel_initializer = tf.random_normal_initializer(stddev=stdv),
                                      use_bias=False)
  bn_r = bn_(conv_r,phase)
  relu_r = relu_(bn_r)
  return relu_r
  '''shape = [4,4,nout,nin]
  stdv = math.sqrt(2/(nin*2*2))
  w = weight_variable(shape, stdv)
  input_shape = tf.shape(x)
  newshape = tf.stack([input_shape[0], 2*input_shape[1], 2*input_shape[2], nout])
  conv_result = tf.nn.conv2d_transpose(x, w, output_shape=newshape, strides=[1, 2, 2, 1], padding='SAME') + bias_variable([nout],stdv)
  bn_prepare = tf.reshape(conv_result, shape = newshape)
  bn_result = bn_(bn_prepare,phase)
  relu_result = relu_(bn_result)
  return relu_result'''

def deconv1_(x, nin, num_up, phase):
  cur_in = nin
  cur_out=(2**(num_up-1))

  for i in range(num_up):
    with tf.variable_scope('up%d' % (i)):
      x = deconv_(x,cur_in,cur_out,phase)
    cur_in = cur_out
    cur_out = math.floor(cur_out/2)

  return x

def deconvn_(x, nin, nout, num_up, phase):
  cur_in = nin
  cur_out=(2**(num_up-1)*nout)

  for i in range(num_up):
    with tf.variable_scope('up%d' % (i)):
      x = deconv_(x,cur_in,cur_out,phase)
    cur_in = cur_out
    cur_out = math.floor(cur_out/2)

  return x

def deconv2_(x, nin, num_up, phase):
  cur_in = nin
  cur_out=(2**(num_up))

  for i in range(num_up):
    with tf.variable_scope('up%d' % (i)):
      x = deconv_(x,cur_in,cur_out,phase)
    cur_in = cur_out
    cur_out = math.floor(cur_out/2)

  return x

def model_apply(sess,result,input_img,phase,tI,ins):
  wI=np.zeros([ins,ins])
  pmap=np.zeros([tI.shape[0],tI.shape[1],nclass-1])
  avI=np.zeros([tI.shape[0],tI.shape[1],nclass-1])
  for i in range(ins):
    for j in range(ins):
      dx=min(i,ins-1-i)
      dy=min(j,ins-1-j)
      d=min(dx,dy)+1
      wI[i,j]=d;
  wI = wI/wI.max()

  avk = 2
  nrotate = 1
  for i1 in range(math.ceil(float(avk)*(float(tI.shape[0])-float(ins))/float(ins))+1):
    for j1 in range(math.ceil(float(avk)*(float(tI.shape[1])-float(ins))/float(ins))+1):
      insti=math.floor(float(i1)*float(ins)/float(avk))
      instj=math.floor(float(j1)*float(ins)/float(avk))
      inedi=insti+ins
      inedj=instj+ins
      if inedi>tI.shape[0]:
        inedi=tI.shape[0]
        insti=inedi-ins
      if inedj>tI.shape[1]:
        inedj=tI.shape[1]
        instj=inedj-ins
      print(insti,inedi,instj,inedj)
      
      feed_image=np.zeros([nrotate,ins,ins,imageType])
      for i in range(nrotate):
        small_in = tI[insti:inedi,instj:inedj]
        feed_image[i,:,:,:] = np.rot90(small_in, i)
      small_out = sess.run(result,feed_dict={input_img:feed_image, phase: False})
      small_pmap = small_out[0,:,:,:]
      for i in range(1,nrotate):
        small_pmap = small_pmap + np.rot90(small_out[i,:,:,:],-i)
      small_pmap = small_pmap / nrotate

      for i in range(nclass-1):
        pmap[insti:inedi,instj:inedj,i] += np.multiply(small_pmap[:,:,i],wI)
        avI[insti:inedi,instj:inedj,i] += wI

  return np.divide(pmap,avI)

def spatial_dropout(x, phase):
  d = tf.shape(x)
  x_drop = tf.layers.dropout(x, noise_shape=[d[0],1,1,d[3]], training=phase)
  return x_drop

def branch(c1_2_conv, c2, c3, c4, nc, phase, scope):
  with tf.variable_scope(scope):
    with tf.variable_scope('up1'):
        up1_conv = convnn_(c1_2_conv,3,1,nc,2,phase)
        up1_bn = bn_(up1_conv, phase)
        up1 = relu_(up1_bn)
    
    with tf.variable_scope('up2'):
      up2 = deconv2_(c2,2*nc,1,phase)
  
    with tf.variable_scope('up3'):
      up3 = deconv2_(c3,4*nc,2,phase)

    with tf.variable_scope('up4'):
      up4 = deconv2_(c4,8*nc,3,phase)

    with tf.variable_scope('final'):
      f1 = tf.concat([up1, up2, up3, up4],3)
      with tf.variable_scope('final_conv1'):
        f1_conv = convnn_(f1,3,1,4*nclass,4*nclass,phase)
        f1_bn = bn_(f1_conv,phase)
        f1_relu = relu_(f1_bn)
      with tf.variable_scope('final_conv2'):
        print('1x1')
        output = convnn_(f1_relu,1,1,4*nclass,nclass,phase) + tf.concat([tf.constant([0],dtype=tf.float32),bias_variable([nclass-1])],axis = 0)
  return output

def model(input_img,phase,nc):
  with tf.variable_scope('place_holder'):
    with tf.variable_scope('model'):
      with tf.variable_scope('scale1'):
        with tf.variable_scope('conv1'):
          c1_1_conv = conv_(input_img,3,imageType,nc,phase)
          c1_1_bn = bn_(c1_1_conv,phase)
          c1_1_relu = relu_(c1_1_bn)
        with tf.variable_scope('conv2'):
          c1_2_conv = conv_(c1_1_relu,3,nc,nc,phase)
          c1_2_bn = bn_(c1_2_conv,phase)
          c1_2_relu = relu_(c1_2_bn)
    
      with tf.variable_scope('scale2'):
        pool1 = max_pool_2x2(c1_2_relu)
        c2 = stack(pool1,nc,2*nc,2,phase)
    
      with tf.variable_scope('scale3'):
        pool2 = max_pool_2x2(c2)
        c3 = stack(pool2,2*nc,4*nc,2,phase)
    
      with tf.variable_scope('scale4'):
        pool3 = max_pool_2x2(c3)
        c4 = stack(pool3,4*nc,8*nc,2,phase)
  
      scores = []
      results = []

      cur_score = branch(c1_2_relu, c2, c3, c4, nc, phase, 'branch%d' % (1))
      cur_result = tf.nn.softmax(cur_score)
      print('good score')
      scores.append(cur_score)
      results.append(cur_result)

  return scores, results

def main(_):
  if not os.path.exists(logdir):
    os.mkdir(logdir)
  for file in os.listdir(logdir):
    print('removing '+os.path.join(logdir,file))
    os.remove(os.path.join(logdir,file))

  model_loc = FLAGS.model
  testing_dir = FLAGS.test_dir

  if (len(testing_dir)==0):
    train_imgs, label_imgs = load_data(imgdir)
  #test_imgs = load_test(imgdir)
  #print('test:',len(test_imgs))
    count = [0.]*nclass
    for lI in label_imgs:
      for i in range(nclass):
        ok = np.equal(lI,float(i)*interv/255)
        count[i] += ok.sum()
    count = count/sum(count)
  else: 
    i=nclass-1
  

  input_img = tf.placeholder(tf.float32, [None,ins,ins,imageType])
  phase = tf.placeholder(tf.bool, name='phase')
  output_gt = tf.placeholder(tf.int32, [None, ins, ins])
  lr = tf.placeholder(tf.float32, name='lr')

  cost = tf.cast(tf.greater(output_gt,0),tf.float32)

  output_gt2 = output_gt - 1
  output_gt2 = relu_(output_gt2)

  tf.summary.image('input_img',input_img,batch_size)
  tf.summary.image('output_gt',tf.cast(tf.expand_dims(output_gt,axis=-1),tf.float32),batch_size)
  
  with tf.device('/gpu:%d' % (FLAGS.gpu)):
    scores, results = model(input_img,phase,nc)

    loss = tf.reduce_sum(tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = scores[0],labels = output_gt2 ,name="entropy"),cost))/tf.reduce_sum(cost)
    tf.summary.scalar('loss%d' %(i),loss)
  
    cross_entropy = loss

    #gradient = tf.gradients(cross_entropy, tf.trainable_variables())
    #optimizer = AMSGradOptimizer(learning_rate=lr)

    with tf.variable_scope('result'):
      result = results[0][:,:,:,1:nclass]

    tf.summary.scalar('error',cross_entropy)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      print('AmsGrad')
      #train_step = optimizer.apply_gradients(gradient)
      train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement=True
  sess = tf.Session(config=config)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(logdir,
                                       sess.graph)
  saver = tf.train.Saver(max_to_keep=100000000)

  for var in tf.trainable_variables():
    print(var)

  if (len(model_loc)==0):
    print('Init model from scratch')
    sess.run(tf.global_variables_initializer())
  else:
    print('Load model: ' + model_loc)
    saver.restore(sess, model_loc)
    sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name]))

  total_parameters = 0
  for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    total_parameters += variable_parametes
  print(total_parameters)
   
  if (len(testing_dir)>0):
    if not os.path.exists(resultApplydir + '/'):
      os.mkdir(resultApplydir + '/')
    print('Applying on: ' + testing_dir)
    files = []
    for file in os.listdir(testing_dir):
      if file.endswith(".png"):
        print(file)
        files.append(file[:-8])
    uniqf = list(set(files))

    for file in uniqf:
      if True:
        print(file)
      
        img_save_dir = resultApplydir + '/c' + str(2) + '_' + file +'.png'
        txt_save_dir = resultApplydir + '/mismatch_' + file + '.txt'

        if Path(img_save_dir).exists():
          print('exists')
          continue

        if (os.path.isfile(os.path.join(testing_dir,file+'_2PF.png'))!=1) or (os.path.isfile(os.path.join(testing_dir,file+'_3PF.png'))!=1) or (os.path.isfile(os.path.join(testing_dir,file+'_THG.png'))!=1) or (os.path.isfile(os.path.join(testing_dir,file+'_SHG.png'))!=1):
          print('incomplete files: ',os.path.join(testing_dir,file+'_3PF.png'));
          continue 

        testI1=skimage.img_as_float(imread(os.path.join(testing_dir,file+'_2PF.png')))
        testI2=skimage.img_as_float(imread(os.path.join(testing_dir,file+'_3PF.png')))
        testI3=skimage.img_as_float(imread(os.path.join(testing_dir,file+'_SHG.png')))
        testI4=skimage.img_as_float(imread(os.path.join(testing_dir,file+'_THG.png')))
        testI=np.zeros((testI1.shape[0],testI1.shape[1],imageType))

        if (testI2.shape[0]!=testI.shape[0]) or (testI3.shape[0]!=testI.shape[0]) or (testI4.shape[0]!=testI.shape[0]):
          print('Size mismatch');
          np.savetxt(txt_save_dir,[testI1.shape[0],testI2.shape[0],testI3.shape[0],testI4.shape[0]])
          continue

        testI[:,:,0] = testI1
        testI[:,:,1] = testI2
        testI[:,:,2] = testI3
        testI[:,:,3] = testI4

        pmap = model_apply(sess,result,input_img,phase,testI,ins)
        for k in range(nclass-1):
          pmap_img = np.zeros([testI.shape[0],testI.shape[1]],dtype='uint8')
          pmap_img[:,:] = pmap[:,:,k]*255
          print(pmap_img.max())
          print(pmap_img.min())
          img_save_dir = resultApplydir + '/c' + str(k+1) + '_' + file +'.png'
          imsave(img_save_dir,pmap_img)

  else:
    cur_lr = 5e-4

    for i in range(iterMax+1):
      print('iteration ' + str(i + iterModel))
      tic = time.time()
      train_batch, label_batch = get_batch(train_imgs,label_imgs,batch_size)
      toc = time.time() - tic
      print('batch time' + str(toc))
      print(train_batch.shape)
      print(label_batch.shape)
      tic = time.time()
      summary,error, dlr, _ = sess.run([merged,cross_entropy,lr,train_step], feed_dict={
            input_img:train_batch, output_gt: label_batch, phase: True, lr: cur_lr})
      toc = time.time() - tic
      print('train time' + str(toc))
      print(i,error,dlr)
      print('train batch check: ', np.max(train_batch))
      print('label batch check: ', np.max(label_batch))
      
      if math.isnan(error):
        print('error is nan')
        exit()
      
      train_writer.add_summary(summary, i)
      if ((i>0) and ((i+iterModel)%5000 == 0)) or (i==5):
        if not os.path.exists(resultModeldir + '/'):
          os.mkdir(resultModeldir + '/')
        if not os.path.exists(resultImagedir + '/'):
          os.mkdir(resultImagedir + '/')
        checkpoint_name = os.path.join(resultModeldir, str(i+iterModel) + '.ckpt')
        print('Saving model: ' + checkpoint_name)
        saver.save(sess,checkpoint_name)
        files = []
        for file in os.listdir(validationdir):
          if file.endswith(".png"):
            print(file)
            file = file[:-4]
            testI1=skimage.img_as_float(imread(os.path.join(testdir,file+'_2PF.png')))
            testI2=skimage.img_as_float(imread(os.path.join(testdir,file+'_3PF.png')))
            testI3=skimage.img_as_float(imread(os.path.join(testdir,file+'_SHG.png')))
            testI4=skimage.img_as_float(imread(os.path.join(testdir,file+'_THG.png')))
            if (testI2.shape[0]!=testI1.shape[0]) or (testI3.shape[0]!=testI1.shape[0]) or (testI4.shape[0]!=testI1.shape[0]):
              print('wrong size: ',file)
              continue

            testI=np.zeros((testI1.shape[0],testI1.shape[1],imageType))
            testI[:,:,0] = testI1
            testI[:,:,1] = testI2
            testI[:,:,2] = testI3
            testI[:,:,3] = testI4



            pmap = model_apply(sess,result,input_img,phase,testI,ins)
            for k in range(nclass-1):
              pmap_img = np.zeros([testI.shape[0],testI.shape[1]],dtype='uint8')
              pmap_img[:,:] = pmap[:,:,k]*255
              print(pmap_img.max())
              print(pmap_img.min())
              img_save_dir = resultImagedir + '/iter' +str(i+iterModel) +'_c' + str(k+1) + '_' + file +'.png'
              imsave(img_save_dir,pmap_img)
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-model', type=str, default='',
                      help='model location')
  parser.add_argument('-test_dir', type=str, default='',
                      help='model location')
  parser.add_argument('-gpu', type=int, default='1',
                      help='use gpu')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
