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
    probInfectTF=tf.Variable(tf.zeros([num_people,1]),name="Infprob")
    update0=tf.to_double(probInfectTF)

    # probInfect[state == 0] = BetaMatrix[:, state == 1].sum(1)[state == 0]
    # There is no boolean index in tensorflow yet, issue #206 and #4639

    #So we first get the index about the element equals to "1" and "0" in the state
    booleanIndexOfInfect = tf.equal(state, 1) #The state which equals to "1"=infected, one ops
    InfectIndex = tf.where(booleanIndexOfInfect)#The index about the infected, one ops
    init=tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        InfectTensor=sess.run(InfectIndex)
    #Compute the state=1 index

    booleanIndexOfSuscep = tf.equal(state, 0) #The state which equals to "0"=Susceptible, one ops
    SuscepIndex = tf.where(booleanIndexOfSuscep)#The index about the infected, one ops
    init=tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        SuscepTensor=sess.run(SuscepIndex)
    #Compute the state=0 index

    #Now we got the index of susceptible individuals and infected individuals
    #Then we will slice the BetaMatrix to get cumulative infection intensity
    updateInfectIntensity=tf.reduce_sum(tf.convert_to_tensor(BetaMatrix[InfectTensor,SuscepTensor]),0)
    updateInfectIntensity=tf.to_float(updateInfectIntensity)
    Update1=tf.scatter_update(probInfectTF,SuscepIndex,updateInfectIntensity)

    Removal=np.exp(-gamma)*np.ones(InfectTensor.shape)
    Update2=tf.scatter_update(probInfectTF,InfectIndex,Removal)
    init = tf.initialize_all_variables()
    with tf.Session as sess:
        sess.run(init)
        sess.run(update0)
        sess.run(Update1)
        sess.run(Update2)
        print(probInfectTF)

if __name__=="__main__":
    import tensorflow as tf
    import numpy as np
    import distanceTF as dt
    import coordinate as cr
    import BetaMatrix as bt

    coordinate = cr.geodata(3)  # Generate the coordinate
    a = dt.Distance(coordinate)  # distance matrix
    b = bt.tfBetaRunning(bt.tfBetaMatrix(a, [0.3, 0.1], bt.exponentialKernel))  # BetaMatrix for testing\
    # 3 people in testing
    state = np.array([1, 0, 0])
    tfState = tf.constant(state)

    tfTransformOneColumn(np.array([1, 0, 0]), b, 0.3)