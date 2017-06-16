import sys
import vtlPythonAPI as vtl
import tensorflow as tf
import numpy as np
import TextScraper as sc
import TrainingDataGen as tdg
import matplotlib.pyplot as plt
import time




#********************************* DEFINITIONS************************************#
#*********************************************************************************#

#Will iterate through the list of csv training files, and create batches of training data
#the csv columns are organized as the output first (about 144 columns) then the audio input
#In our synthesizer model the params are input and then generate an audio
def readFilesnBatch(files,batchSize,inSize,outSize):
    filename_queue = tf.train.string_input_producer(files)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[0.0] for x in range(outSize + inSize )]

    #Features are the last inSize columns(the audio)
    row = tf.decode_csv(value, record_defaults=record_defaults)
    audio = tf.stack(row[outSize:outSize+inSize])
    params = tf.stack(row[0:outSize])

    min_after_dequeue = 1000
    capacity = min_after_dequeue+3 * batchSize
    audioBatch,paramBatch = tf.train.shuffle_batch([audio,params],batch_size=batchSize,capacity=capacity,min_after_dequeue=min_after_dequeue)
    return audio,params,audioBatch,paramBatch

#Given a audiofile name, and the number of samples to split the audio up into, will
#Pad the audio with trailing zeros, then put the splits into a tensorflow batch 
#So it can be handed to session.run in a single call.
def openSplit(fileName, inSize):
    audioList = vtl.wav_to_list(fileName)
    padSize = inSize - (len(audioList)%inSize)
    print(padSize)
    audioList.extend([0]*padSize)#trailing zeros if the audio doesnt divide nicely
    a = [audioList[a:a+inSize] for a in range(0,len(audioList),inSize)]

    return a

#Will generate weight variables of the given shape, [inputsize * outputSize]
#These weights are then used for the Network connections
def weight_variable(shape):
  initial = tf.random_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#I hope this is correct
#Review how to properly build this
#Contains the activation functions for each layer, including the input,hidden,output layers
#X is the input Data, W[i] are the weights of the i'th layer
#yHat is the prediction/regression result of the network
def forwardprop(X, W):  
    h = tf.nn.sigmoid(tf.matmul(X, W[0]),name = "InLayer")  
    count = 0;
    for w in W[1:len(W)-1]:
        name = "H-Layer" + str(count)
        h = tf.nn.sigmoid(tf.matmul(h,w),name = name)


    yhat = tf.matmul(h,W[len(W)-1],name = "OutputLayer") 
    return yhat

#The scales of the elements in the output are quite varied
#This is an attempt to normalize them for the networks sake 
#will be used on the true Y value, so the network will try to learn the normalized values as outputs
def normalizeOut(batch,batchSz,tubeCount,VTPCount,glotcount):
    #Order is GltStrt-End, VTPstrt-End, tubeStrt-End
    [_,gMin,gMax,_] = vtl.getGlottisParams(glotcount)
    [_,vMin,vMax,_] = vtl.getTractParams(VTPCount)
    #These Following 2 come from the newTrainingData() and describe tube lenths
    maxLen = 0.005 #if they all come out at max length it will be 20cm
    minLen = 0.0025

    for b in range(batchSz):
        i =0 
        for g in range(glotcount*2):
            batch[b,i] = (batch[b,i] - gMin[g%glotcount]) / (gMax[g%glotcount] - gMin[g%glotcount])
            i = i +1
        for v in range(VTPCount*2):
            batch[b,i] = (batch[b,i] - vMin[v%VTPCount]) / (vMax[v%VTPCount] - vMin[v%VTPCount])
            i = i+1

        for t in range(tubeCount*2):
            batch[b,i] = (batch[b,i] - minLen) / (maxLen - minLen)
            i = i+1

#Undo the previous normalization so that the values can be handed to the synthesizer with unabashed values
def denormalizeOut(batch,batchSz,tubeCount,VTPCount,glotcount):
    [_,gMin,gMax,_] = vtl.getGlottisParams(glotcount)
    [_,vMin,vMax,_] = vtl.getTractParams(VTPCount)
    #These Following 2 come from the newTrainingData() and describe tube lenths
    maxLen = 0.005 #if they all come out at max length it will be 20cm
    minLen = 0.0025

    for b in range(batchSz):
        i = 0
        for g in range(glotcount*2):
            batch[b,i] = (batch[b,i] * ( gMax[g%glotcount] - gMin[g%glotcount]) + gMin[g%glotcount])
            i = i+1

        for v in range(VTPCount*2):
            batch[b,i] = (batch[b,i] *  (vMax[v%VTPCount] - vMin[v%VTPCount]) + vMin[v%VTPCount])
            i = i+1
        for t in range(tubeCount*2):
            batch[b,i] = (batch[b,i] * (maxLen - minLen) + minLen)
            i = i+1
    

#******************************END DEFINITIONS************************************#
#*********************************************************************************#


#the outputs from the learner are:

#The Glottis (list of  length (numGlottisParams * numFrames))
#The Vocal Tract Parameters we start from (list of length(numVocalTractParams * numFrames))
#The Length of tubes in the Synth model (list of length (tubeSections * num Frames))
#The distances from glottis to incisors (list of length(numFrames))
#The areas of the velum (list of of length(numFrames))

#The input to the learner would be the desired sound
#The feedback/error is what is generated by the VTL model from the networks given parameters.


spkr ="test1.speaker"
framerate = 4

trainFiles = ["Train.csv","Train1.csv","Train2.csv","Train3.csv"]
#USe this to generate new training data
for f in trainFiles:
    tdg.newTrainingData(10,framerate,spkr,f)


#lets see if we can get it to match a single sound. with only a single frame transition of audio.
vtl.initSpeaker(spkr,False)
[srate,tubecount,vtcount,glotcount] = vtl.getSpeakerConstants()
inSize = int(srate/(framerate)) #I know this works with frameRate of 4

#Glottis + vtp + tubeLengths + incisor + velum + aspStrength
outSize = (glotcount + vtcount + tubecount + 3) * 2#Times two for a start and end

#Here We give TF a list of the files it can read to get our values from

batchSz = 3
audio,params,audioBatch,paramBatch= readFilesnBatch(trainFiles,batchSz,inSize,outSize)  

t_start = time.process_time()
with tf.Session() as sess:
    print("Entered Session")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    inputAudio = tf.placeholder(tf.float32,shape=[None,inSize],name = "inputAudio")
    outActual = tf.placeholder(tf.float32,shape=[None,outSize],name = "outActual")

    #create Weights (this is basically the network)
    h_size = inSize
    W = list()
    W.append(weight_variable((inSize,h_size)))
    for s in range(5):
        W.append(weight_variable((h_size,h_size)))
    W.append(weight_variable((h_size,outSize)))

    #Forward propagation(How to calculate from input to output)
    outEstimate = forwardprop(inputAudio, W)

    #Back Progoation(How to correct the Error from the estimate)
    cost =tf.reduce_mean(tf.squared_difference(outActual,outEstimate))
    train_step = tf.train.AdamOptimizer(0.9).minimize(cost)


    init = tf.global_variables_initializer()
    sess.run(init)
    print("Session Init complete")
    AC = 0
    ACL = list()
    costList = list()
    epochs = 100000
    x,y = sess.run([audioBatch,paramBatch])
    for i in range(epochs):  #It seems this will stop on the min(number of lines or range)
        print(i," of ",epochs)
           
        
        #normalizeOut(y,batchSz,tubecount,vtcount,glotcount)
        _, Cst = sess.run([train_step,cost], feed_dict={inputAudio: x, outActual: y})

        AC = (Cst + (i)*AC)/(i+1)
        ACL.append(AC)
        print("Average Cost: ",AC)
        costList.append(Cst)

        if (i>epochs-2):
            oE = sess.run(outEstimate,feed_dict={inputAudio: x})
            #denormalizeOut(oE,batchSz,tubecount,vtcount,glotcount)
            #denormalizeOut(y,batchSz,tubecount,vtcount,glotcount)

            est = oE[0,:]
            act = y[0,:]
            diff = list()
            for e in range(len(est)):
                diff.append(act[e]-est[e])
            plt.bar(range(len(diff)),diff)
            plt.show()
                             
    t_end = time.process_time()
    print("Total Training Network time = ",t_end-t_start)

#Open and split the test audio into even size chunks
    #print("Writing and synthesizing audio")
    #testAudio = openSplit('Mrgan_Free.wav', inSize)
    #outAudio = list()
    #for a in testAudio:
    #    testParams = sess.run(outEstimate,feed_dict={inputAudio: np.array(a).reshape(1,inSize)})
    #    outAudio += genAudio(testParams,inSize)

    #vtl.list_to_wave('Mrgan_Free_out.wav',outAudio,2,srate)

    coord.request_stop()
    coord.join(threads)

plt.plot(costList,'b')
plt.plot(ACL,'r')
plt.show()








        

