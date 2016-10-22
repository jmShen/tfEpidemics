import numpy as np
import tensorflow as tf
import coordinate as cr
def tfDistance(coordinate):
    '''

    :param coordinate: is the x,y coordinate for each individual
    :return: the distance matrix, the i-j entry of the matrix is the distance between individual i and j
    '''
    x=tf.constant(coordinate)
    #DistanceMatrix=-2 * np.dot(X, X.T) + np.sum(np.square( X), 1).reshape(num_people, 1) + np.sum(np.square( X), 1).reshape(1, num_people)
    #This is the numpy version distance matrix calculation
    #now we want transfer it into tensorflow version
    r=tf.reduce_sum(x*x,1)
    r=tf.reshape(r,[-1,1])
    D=-2*tf.matmul(x,tf.transpose(x))+r+tf.transpose(r)
    return tf.sqrt(D)

def tfDistanceLowMemory(coordinate):
    #from Chris. Jewell
    x = tf.constant(coordinate)
    D=tf.map_fn(lambda i: tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(i-x,2),1))),x)
    return D

def Distance(coordinate):
    lmModel = tfDistance(coordinate)
    # lmModel=tfDistanceLowMemory(coordinate)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        lmModel = sess.run(lmModel)
        return lmModel

if __name__ == "__main__":
    coordinate=cr.geodata(100)
    print Distance(coordinate)