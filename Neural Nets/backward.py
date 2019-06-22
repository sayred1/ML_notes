import numpy as np

def loss(self, output, label):
    
    _image_arr = np.zeros((10,1))
    _image_arr[label-1] = 1.
    label = _image_arr # array of 0s and a single 1 (true value)
    
    loss = -np.sum(label * np.log(output))
    return loss

def get_mask(conv):
    
    """
    args 
        - conv: convolution layer
    
    returns
        - mask of max values from convolution layer
    """

    mask = conv == np.max(conv)
    return mask

def maxpool_gradient(self, conv_prev, dpool, filSize, stride):
    
    """
    args
        - conv_prev: input into current pooling layer (normally convolution)
        - dpool: gradient of the loss w.r.t current pooling layer
        - filSize: filter size of pooling operation
        - stride: stride size of pooling operation
        
    returns
        - dconv_prev: gradient of loss w.r.t conv_prev
    """
    
    n, h_prev, w_prev, d_prev = conv_prev.shape     # shape of input into pool layer
    n, h, w, d = dpool.shape                        # shape of derivative of pool layer
    
    # initialize derivative
    dconv_prev = np.zeros(conv_prev.shape)       
    
    for i in range(n):
        height = 0
        for j in range(w):
            width = 0
            for k in range(h):
                
                conv_prev_slice = conv_prev[i, height:height+filSize, width:width+filSize, :]
                mask = get_mask(conv_prev_slice)
                dconv_prev[i, height:height+filSize, width:width+filSize, :] += np.multiply(mask, dpool[i, j, k, :])
                
                width += stride
            height += stride
    return dconv_prev
    
def convolution_gradient(self, dout, prev_input, W, b, filSize, stride):
    
    """
    args
        - dout: gradient of loss w.r.t convolution output 
        - prev_input: input into convolution layer (usually a pooling layer)
        - W: kernel of convolution layer
        - b: bias of convolution layer
        - filSize: filter size of pooling operation
        - stride: stride size of pooling operation
        
    returns
        - dprev_input: gradient of cost with respect to conv_prev
        - dW: gradient of loss w.r.t convolution kernel 
        - db: gradient of loss w.r.t convolution bias 
    """
    
    (n_prev, h_prev, w_prev, d_prev) = prev_input.shape       # input shape to conv layer
    (n_weight , f, f, d_weight) = W.shape                     # kernel shape of conv layer
    (n, h, w, d) = dout.shape                                 # gradient shape of conv layer derivative
    
    # initialize derivatives
    dprev_input = np.zeros((n_prev, h_prev, w_prev, d_prev))                      
    dW = np.zeros((n_weight , f, f, d_weight)) 
    db = np.zeros((n,1))

    for i in range(n):                       
        height = 0
        for j in range(h):                   
            width = 0
            for k in range(w):               
                for l in range(d):   
                    
                    prev_slice = prev_input[i, height:height+filSize, width:width+filSize, :] 
                    dprev_input[i, height:height+filSize, width:width+filSize, :] += W[i,:,:,:] * dout[i, j, k, l]
                    dW[i,:,:,:] += prev_slice * dout[i, j, k, l]
            
            db[i,:] += np.sum(dout)

    return dprev_input, dW, db

def fc_grad_second(self, label, output, prev_input):
    gradf2 = output - label
    gradw4 = gradf2.dot(prev_input.T)
    gradb4 = np.sum(gradf2, axis = 0)
    return(gradf2,gradw4,gradb4)

def fc_grad_first(self, gradfc2, weight, prev_input):
    f1 = self.forward_dict['f1']
    gradf1 = weight.T.dot(gradfc2)
    gradf1[f1<=0] = 0
    gradw3 = gradf1.dot(prev_input)
    gradb3 = np.sum(gradf1,axis = 0)
    return(gradf1, gradw3, gradb3)