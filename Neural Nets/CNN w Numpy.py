import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import utilz
import backward

from tensorflow import keras
from tensorflow.keras import datasets

# load the data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)) # (samples, rows, cols, channels)
test_images = test_images.reshape((10000, 28, 28, 1)) # (samples, rows, cols, channels)

# simple normalization of the image pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

train_image = train_images[0].reshape(1,28,28,1) # one img x 28 pixels x 28 pixels x one channel
train_label = train_labels[0]

class my_CNN(object):
    
    def __init__(self):
        pass
    
    # dictionaries used later on for the backprop step
    weight_dict = {} 
    bias_dict = {}  
    forward_dict = {}
    unit_dict = {}
    
    # convolution function
    def convolve(self, image, kernel_dim, stride, num_filters, name):
        if name == 'conv1':
            self.forward_dict['image'] = image
        output = utilz.get_output(self, image, kernel_dim, stride, num_filters)
        numImages, heightImages, widthImages, depthImages = output.shape
        weights = []
        biases = []
        for i in range(numImages):
            for l in range(depthImages):
                weight, bias = utilz.get_parameters(self, image,kernel_dim) # weights change for each depth slice
                height = 0
                for j in range(heightImages): 
                    width = 0
                    for k in range(widthImages):
                        output[i,j,k,l] = np.sum(image[i,height:kernel_dim+height,width:kernel_dim+width,:]*weight) + bias
                        width += stride
                    height += stride
                weights.append(weight)
        
        # store weights, biases and neuron outputs
        self.weight_dict[name] = weights
        self.bias_dict[name] = bias
        self.forward_dict[name] = output
        return output
    
    # function for pooling operation
    def pool(self, conv_output, _image,pool_size, stride, name,fully_connect):
        output = utilz.get_output(self, conv_output, pool_size, stride, conv_output.shape[-1])
        numImages, heightImages, widthImages, _ = output.shape
        pooling_units = []
        for i in range(numImages): 
            height = 0
            for j in range(heightImages):
                width = 0
                for k in range(widthImages):
                    temp = conv_output[i,height:height+pool_size,width:width+pool_size,:]
                    output[i,j,k,:] = np.max(temp)
                    
                    (a,b,_) = temp.shape
                    temp = np.reshape(temp,(1,a,b,_))
                    
                    maxCoord = np.unravel_index(np.argmax(temp, axis = None), temp.shape) 
                    (i,a,b,_) = maxCoord
                    maxCoord = (i,a+width,b+height,_)    

                    pooling_units.append(maxCoord)                       
                    width+=stride
                height+=stride

        self.unit_dict[name] = pooling_units
        self.forward_dict[name] = output

        # if then statement, if True implement two fully connected layers (last one being prediction) then start backprop
        if fully_connect == False:
            return output
        else:
            # flatten
            p2_flat = np.reshape(output.flatten(),(1,1600))
            self.forward_dict['flatten_p2'] = p2_flat
            
            # fully_connected 1
            weight = np.random.randn(64,1600) * 0.01
            bias = np.random.randn(1) * 0.01
            f1 = self.relu(weight.dot(p2_flat.transpose())+bias)
            self.forward_dict['f1'] = f1  
            self.weight_dict['f1'] = weight
            self.bias_dict['f1'] = bias  
            
            # fully_connected 2
            weight = np.random.randn(10,64) * 0.01
            bias = np.random.randn(1) * 0.01
            f2 = self.softmax(weight.dot(f1)+bias)
            self.forward_dict['f2'] = f2
            self.weight_dict['f2'] = weight
            self.bias_dict['f2'] = bias
            
            ######################################
            ############# Backprop ###############
            ######################################
            """
            hn = conv, pn = pool, fn = fc
            cn = activ. function(hn)
            kn = conv. kernels, bn = bias, wn = fc weights
            """
            
            X,h1,p1,h2,p2= self.get_forward_dict()
            c1,c2 = self.relu(h1),self.relu(h2)
            f1,f2 = self.forward_dict['f1'],self.forward_dict['f2']
            k1,b1,k2,b2,w3,b3,w4,b4 = self.get_parameter_dict()
            
            # print keys for reference:
            print('forward keys:', self.forward_dict.keys())
            print('weight keys:', self.weight_dict.keys())
            print('bias keys:',self.bias_dict.keys())
            print('pooling keys:',self.unit_dict.keys())

            loss = backward.loss(self, f2, _image) 
            
            # fc second gradient
            gradfc2, gradw4, gradb4 = backward.fc_grad_second(self, _image, f2, f1,)
            
            # fc first gradient
            gradfc1, gradw3, gradb3 = backward.fc_grad_first(self, gradfc2, w4, p2_flat)
                        
            # flattened pool 2 gradient
            gradp2 = w3.T.dot(gradfc1)
            gradp2 = np.reshape(gradp2, p2.shape)
            
            # conv 2 gradient
            gradc2 = backward.maxpool_gradient(self, c2, gradp2, 3, 2)
            gradc2[conv2<=0] = 0
            
            # pool 1 gradient
            gradp1, gradk2, gradb2 = backward.convolution_gradient(self, gradc2, p1, k2, b2, 3, 1)
            
            # conv 1 gradient
            gradc1 = backward.maxpool_gradient(self,c1, gradp1, 2, 2)
            gradc1[conv1<=0] = 0
            
            # image gradient 
            dimage, gradk1, gradb1 = backward.convolution_gradient(self,gradc1, X, k1, b1, 3, 1)
            

    def relu(self,image):
        return np.maximum(0,image)
    
    def softmax(self, image):
        return np.exp(image)/np.sum(np.exp(image))
    
    def get_forward_dict(self):
        X,h1,h2 = self.forward_dict['image'],self.forward_dict['conv1'],self.forward_dict['conv2']
        p1,p2 = self.forward_dict['pool1'],self.forward_dict['pool2']
        return (X,h1,p1,h2,p2)
    
    def get_parameter_dict(self):
        k1,k2,w1,w2 = self.weight_dict['conv1'][0],self.weight_dict['conv2'][0],self.weight_dict['f1'],self.weight_dict['f2']
        b1,b2,b1_f,b2_f = self.weight_dict['conv1'],self.weight_dict['conv2'],self.weight_dict['f1'],self.weight_dict['f2']
        return (k1,b1,k2,b2,w1,b1_f,w2,b2_f)

conv1 = my_CNN().convolve(train_image,3,1,64,'conv1')
pool1 = my_CNN().pool(conv1,train_label,2,2,'pool1',False)
conv2 = my_CNN().convolve(pool1,3,1,64,'conv2')
pool2 = my_CNN().pool(conv2, train_label,3,2,'pool2',True)