import numpy as np

def get_output(self, image, kernel_dim, stride, num_filters):
    """
    initiate output
    """
    numImages, heightImages, widthImages, depthImages = image.shape
    output_volume = (widthImages-kernel_dim)/stride + 1
    if output_volume.is_integer() == False:
        raise NotImplementedError
    else:
        output_volume = int(output_volume)
        return np.zeros((numImages,output_volume,output_volume,num_filters))

def get_parameters(self, image, kernel_dim):
    """
    initiate weights and biases
    """
    depth = image.shape[-1]
    weight = np.random.randn(image.shape[0],kernel_dim, kernel_dim, depth)
    bias = np.random.randn(1)
    return (weight,bias)

