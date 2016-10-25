import tensorflow as tf
import numpy as np
def tfTransformProbability(state,BetaMatrix,gamma):
    '''

    :param state: is the current state for each individuals, the structure is (1,0,0,1,0,2,1,0,....,)
    :param BetaMatrix: is the infected intensity from individual i get infected from individual j is the i,j entry of BetaMatrix
    :param gamma: the removal intensity for infected people
    :return: an array of probability for individuals transform to next state: S->I, I->R
    numpy version code:
    p_S_to_I = self.Lambda(state, BetaMatrix)
    # now it is a num_people array
    p_I_to_R = 1 - np.exp(-gamma)
    prob_transitions = np.zeros((self.num_people,))
    # if sum(p_S_to_I[p_S_to_I!=0])!=0:
    # prob_transitions[state==0] = p_S_to_I[p_S_to_I!=0]
    prob_transitions = p_S_to_I
    # prob_transitions[state==0] = p_S_to_I[p_S_to_I!=0]
    prob_transitions[state == 1] = p_I_to_R
    prob_transitions[state == 2] = 0
    '''

def tfTransformOneColumn(state,BetaMatrix,gamma):
    '''

    :param state: current state for each individuals, a list or a ndarray(a ndarray better?)
           state come from generator, which maybe just use the numpy version is enough?
           and come from a txt, so it is easy to load into a ndarray-> to get dimension easily.
           In previous code, it used the class member num_people, but this is function oriented so
           there is no self, but maybe use as a augument?   `
    :param BetaMatrix: Infection intensity of i infect j
    :return: an array of probability for susceptible individual to infected individual
    numpy version code
    probInfect = np.zeros((self.num_people))
    probInfect[state == 0] = BetaMatrix[:, state == 1].sum(1)[state == 0]
    probInfect = 1 - np.exp(-probInfect)
    '''
    state=np.array(state)
    num_people=np.size(state)
    stateTf=tf.constant(state)
    probInfectTF=tf.variable(tf.zeros([num_people,1]),name="Infprob")
    booleanIndexOfInfect = tf.equal(state, 1) #The state which equals to "1"=infected, one ops
    InfectIndex = tf.where(booleanIndexOfInfect)#The index about the infected, one ops
    init=tf.initialize_all_variables()
    with tf.Session as sess:
        sess.run(init)
        InfectTensor=sess.run(InfectIndex)
    Removal=np.exp(-gamma)*np.ones(InfectTensor.shape)
    update=tf.scatter_update(probInfectTF,InfectIndex,Removal)
    init = tf.initialize_all_variables()
    with tf.Session as sess:
        sess.run(init)
        InfectTensor=sess.run(InfectIndex)