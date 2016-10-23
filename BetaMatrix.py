import tensorflow as tf
import numpy as np
import distanceTF as dt
import coordinate as cr
def tfBetaMatrix(DistanceMatrix, parameter, kernelFunction, GP=None):
    '''

    :param DistanceMatrix: Here should be a nxn matrix as tensor
    :param parameter: The parameter of the kernel function, for example,
           the exponential kernel function is beta0*exp(d/phi),parameter is (d,phi)
    :param kernelfunction: The kernel function, expoential is beta0*exp(d/phi)
    :param GP: Gaussian process fix factor
    :return: A nxn matrix as beta_{ij} for d_{ij}
    '''
    #the numpy version is directly using f to every entry of the distance matrix
    #So the first task is to found a functio in tensorflow to apply function to all entry of matrix in tensor
    BetaMatrix=tf.map_fn(lambda d:kernelFunction(d,parameter,GP),DistanceMatrix)
    return BetaMatrix
def zeroKernel(d,parameter,GP=None):
    return np.zeros(0)
def doubleKernel(d,parameter,GP=None):
    return d+d
def tfBetaRunning(lmModel):
    init=tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        lmModel = sess.run(lmModel)
        return lmModel
def exponentialKernel(d,parameter,GP=None):
    beta0=parameter[0]
    sigma=parameter[1]
    return beta0*tf.exp(-d*sigma)




if __name__ =="__main__":
    coordinate = cr.geodata(3)
    a = dt.Distance(coordinate)
    print (a)
    print ("after track")
    #print(tfBetaRunning(tfBetaMatrix(a,0,zeroKernel)))
    print ("after track")
    print(tfBetaRunning(tfBetaMatrix(a, [0.3,0.1],exponentialKernel)))