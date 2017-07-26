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
    record_defaults = [[0.0] for x in range(outSize + inSize)]

    #Features are the last inSize columns(the audio)
    row = tf.decode_csv(value, record_defaults=record_defaults)
    params = tf.stack(row[:outSize])
    audio = tf.stack(row[outSize:])


    min_after_dequeue = 1000
    capacity = min_after_dequeue+3 * batchSize
    audioBatch,paramBatch = tf.train.shuffle_batch([audio,params],batch_size=batchSize,capacity=capacity,min_after_dequeue=min_after_dequeue,allow_smaller_final_batch=True)
    return audio,params,audioBatch,paramBatch

#Given a audiofile name, and the number of samples to split the audio up into, will
#Pad the audio with trailing zeros and split into sections of size inSize
#So it can be handed to session.run
def openSplit(fileName, inSize):
    audioList = vtl.wav_to_list(fileName)

    #TODO: REmove when done debugging this
    plt.plot(audioList)
    plt.show()
    ###

    padSize = inSize - (len(audioList)%inSize)
    print("Testing audio needed a padding of: ",padSize, " zeros")
    audioList.extend([0]*padSize)#trailing zeros if the audio doesnt divide nicely
    a = [audioList[a:a+inSize] for a in range(0,len(audioList),inSize)]

    return a

#Will generate weight variables of the given shape, [inputsize * outputSize]
#These weights are then used for the Network connections
def weight_variable(shape):
  initial = tf.random_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def weight_summaries(var):
  #Attach a lot of summaries to a Tensor (for TensorBoard visualization).#
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('var_stddev'):
      variance = tf.reduce_mean(tf.square(var-mean))
      stddev = tf.sqrt(variance)
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('variance',variance)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('Weight_Values', var)

def variable_summary(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('var_stddev'):
            variance = tf.reduce_mean(tf.square(var-mean))
            stddev = tf.sqrt(variance)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('variance',variance)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

#I hope this is correct
#Review how to properly build this
#Contains the activation functions for each layer, including the input,hidden,output layers
#X is the input Data, W[i] are the weights of the i'th layer
#yHat is the prediction/regression result of the network
def forwardprop(X, W):
    with tf.name_scope("InLayer"):
        h = tf.nn.sigmoid(tf.matmul(X, W[0]),name = "InLayer")
        inputLayer = tf.matmul(X,W[0])
        inputLayerAct = tf.nn.sigmoid(inputLayer)
        weight_summaries(W[0])
    tf.summary.histogram("InputLayer",inputLayer)
    tf.summary.histogram("InputLayer_Activation",inputLayerAct)

    count = 0;
    for w in W[1:len(W)-1]:       
        name = "H-Layer-Weights_" + str(count)
        count +=1
        with tf.name_scope(name):
            layer = tf.matmul(h,w)
            h = tf.nn.sigmoid(layer, name = name)
            weight_summaries(w)
        tf.summary.histogram(name, layer)
        tf.summary.histogram(name + "_activation",h)
    
    with tf.name_scope("OutputLayer"):
        yhat = tf.matmul(h,W[len(W)-1],name = "OutputLayer")

    weight_summaries(W[len(W)-1])
    tf.summary.histogram("OutputLayer",yhat)

    return yhat

#The scales of the elements in the output are quite varied
#This is an attempt to normalize them for the networks sake 
#will be used on the true Y value, so the network will try to learn the normalized values as outputs
def normalizeOut(batch,batchSz,tubeCount,VTPCount,glotcount):
    #Order is GltStrt-End, VTPstrt-End, tubeStrt-End
    [_,gMin,gMax,_] = vtl.getGlottisParams(glotcount)
    [_,vMin,vMax,_] = vtl.getTractParams(VTPCount)
    #These Following 2 come from the newTrainingData() and describe tube lenths
    maxLen = 0.5 #if they all come out at max length it will be 20cm
    minLen = 2.5
    incisorMax = 20.5
    incisorMin = 9.5
    velumMax = 10
    velumMin = 0.5
    aspMin = -40 #The max is 0
    count = 0
    for b in range(batchSz):
        i =0 
        for g in range(glotcount*2):
            if(batch[b,i] > gMax[i%glotcount] or batch[b,i] < gMin[i%glotcount] ):
                count += 1
            batch[b,i] = (batch[b,i] - gMin[g%glotcount]) / (gMax[g%glotcount] - gMin[g%glotcount])
            i = i +1
        for v in range(VTPCount*2):
            batch[b,i] = (batch[b,i] - vMin[v%VTPCount]) / (vMax[v%VTPCount] - vMin[v%VTPCount])
            i = i+1
        for t in range(tubeCount*2):
            batch[b,i] = (batch[b,i] - minLen) / (maxLen - minLen)
            i = i+1
        for t in range(2):
            batch[b,i] = (batch[b,i] - incisorMin) / (incisorMax - incisorMin)
            i = i+1
        for v in range(2):
            batch[b,i] = (batch[b,i] - velumMin) / (velumMax - velumMin)
            i = i+1
        for a in range(2):
            batch[b,i] = (batch[b,i] - aspMin) / (0 - aspMin)
            i = i+1

#Undo the previous normalization so that the values can be handed to the synthesizer with unabashed values
def denormalizeOut(batch,batchSz,tubeCount,VTPCount,glotcount):
    [_,gMin,gMax,_] = vtl.getGlottisParams(glotcount)
    [_,vMin,vMax,_] = vtl.getTractParams(VTPCount)
    #These Following 2 come from the newTrainingData() and describe tube lenths
    maxLen = 0.5 #if they all come out at max length it will be 20cm
    minLen = 2.5
    incisorMax = 20.5
    incisorMin = 9.5
    velumMax = 10
    velumMin = 0.5
    aspMin = -40 #The max is 0

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
        for t in range(2):
            batch[b,i] = (batch[b,i] * (incisorMax - incisorMin) + incisorMin)
            i = i+1
        for v in range(2):
            batch[b,i] = (batch[b,i] * (velumMax - velumMin) + velumMin)
            i = i+1
        for a in range(2):
            batch[b,i] = (batch[b,i] * (0 - aspMin) + aspMin)
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

trainFiles = list()
postfix = ".csv"
prefix = "Train"
for i in range(392,400):
    trainFiles.append( (prefix + str(i) + postfix))


genData = False # <----------Generate new training data
if genData:
    ExamplesPerFile = 1000
    for f in trainFiles:
        tdg.newTrainingData(ExamplesPerFile,spkr,f)


#lets see if we can get it to match a single sound. with only a single frame transition of audio.
vtl.initSpeaker(spkr,False)
[srate,tubecount,vtcount,glotcount] = vtl.getSpeakerConstants()
inSize = int(srate/(framerate)) #I know this works with frameRate of 4 and 1 


#Glottis + vtp + tubeLengths + incisor + velum + aspStrength
outSize = (glotcount + vtcount + tubecount + 3) * 2#Times two for a start and end
outSize +=1 #For the added framerate


    #Here We give TF a list of the files it can read to get our values from, 
    #the proceed to organize them into batches that are sent to TF
maxBatchSz = 100
trainFiles = list() #Select which files we want to train off of
for i in range(50,300): #using files 50-299 as they they are currently at size framerate 4
      trainFiles.append((prefix + str(i) + postfix))
audio,params,audioBatch,paramBatch= readFilesnBatch(trainFiles,maxBatchSz,inSize,outSize)  
print("Batching complete")

t_start = time.process_time()
with tf.Session() as sess:

    print("Entered Session")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    with tf.name_scope("Input_Audio"):
        inputAudio = tf.placeholder(tf.float32,shape=[None,inSize],name = "inputAudio")
    with tf.name_scope("Actual_output"):
        outActual = tf.placeholder(tf.float32,shape=[None,outSize],name = "outActual")
    tf.summary.audio("Input_Audio",inputAudio,srate,10)
    tf.summary.histogram("Real_output",outActual)

    #create Weights (this is basically the network)
    h_size = inSize
    W = list()
    W.append(weight_variable((inSize,h_size)))
    for s in range(2):
        W.append(weight_variable((h_size,h_size)))
    W.append(weight_variable((h_size,outSize)))
    print("Weights Generated")

    #Forward propagation(How to calculate from input to output)
    with tf.name_scope("Generated_Output"):
        outEstimate = forwardprop(inputAudio, W)
    tf.summary.histogram("NetworkOutput",outEstimate)


    #Back Progoation(How to correct the Error from the estimate)
    costList = list([0] * 10)
    costSplit = tf.squared_difference(outActual,outEstimate)
    with tf.name_scope("Cost_scope"):
        cost =tf.reduce_mean(costSplit)#Back prop is here 
        costListTensor = tf.placeholder(tf.float32,[None],name = "Cost_History")
        variable_summary(costListTensor)
        tf.summary.scalar("cost",cost)
    

    with tf.name_scope("GD_Trainer"):
        train_step = tf.train.AdamOptimizer(0.9).minimize(cost)
 

    init = tf.global_variables_initializer()
    sess.run(init)
    print("Session Init complete")

    #AC = 0 #Average Cost, 
    #ACL = list() #AverageCost List
    #costList = list()
    #costListSplit = list()
    epochs = 10
    
    mergedSummaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter('.\TensorBoard',sess.graph)

    for i in range(epochs):
        print(i," of ",epochs)
        x,y,currBatchSize = sess.run([audioBatch,paramBatch,tf.shape(paramBatch)[0]])# here we are assigning batchsize components to the None Dimension for the x,y
        print("currBatchSize ",currBatchSize)
        normalizeOut(y,currBatchSize,tubecount,vtcount,glotcount)

        if i <1: #This is here to push the initial cost into the cost list, then it can be summarized into the tensor below
            Cst = sess.run(cost,feed_dict={inputAudio: x, outActual: y})
            costList[0] = Cst
                    
        sessList = [costSplit, cost, train_step, mergedSummaries]
        CstSplit,Cst,_,summary = sess.run(sessList, 
                                          feed_dict={inputAudio: x, outActual: y, costListTensor: costList})

        costList[i%10] = Cst
        writer.add_summary(summary,i)
        
        #AC = (Cst + (i+1%10)*AC)/(i+1%10)#Average Cost, keep track of the last 10 iterations since the initial learning will dilute the results as time goes on
        #if((i%10) == 0):
        #    ACL.append(AC) #AverageCost List
        #    print("Average Cost: ",AC)
        #    AC = 0
            
        
        #costList.append(Cst)
        #if(i > epochs-11): #I am putting this here so we can see what the variance looks like when we have had a chance to learn something.
        #    costListSplit.append(CstSplit)

        #if (i>epochs-2):
        #    #Calc the variance of the error here, 
        #    muList = [0] * outSize
        #    for c in range(len(costListSplit)):#number of epochs
        #       for i in range(len(costListSplit[c])):#size of the batch
        #           for e in range(len(costListSplit[c][i])):#these are the values in each example
        #               muList[e] += costListSplit[c][i][e]

        #    for i in range(len(muList)):
        #        muList[i] /= (len(costListSplit) * len(costListSplit[0])) #divide by the number of examples for each value, that is epochs*batchSize

        #    sigmaList = [0] * outSize
        #    for i in range(len(muList)):#The number of values
        #       for c in range(len(costListSplit)):#The number of Epochs
        #           for e in range(len(costListSplit[c])):#the sizse of the batch
        #            sigmaList[i] += (costListSplit[c][e][i] - muList[i])**2
        #       sigmaList[i] /= (len(costListSplit) * len(costListSplit[c])) #This is the SigmaSquared (variance) of our costs , its plotted below
             

        #     #Here we will sample an single case, and compare how off it is from its true value, 
        #    outEst = sess.run(outEstimate,feed_dict={inputAudio: x})
        #    denormalizeOut(outEst,currBatchSize,tubecount,vtcount,glotcount)
        #    denormalizeOut(y,currBatchSize,tubecount,vtcount,glotcount)

        #    est = outEst[0,:]
        #    act = y[0,:]
        #    diff = list()
        #    for e in range(len(est)):
        #        diff.append(act[e]-est[e])
        #    indices = range(len(diff))
        #    width = 0.8
        #    indices2 = [i+0.25*width for i in indices]
        #    indices3 = [i+0.25*width for i in indices2]

        #    plt.bar(indices,act,width=width,color='b',label = "Actual Values")
        #    plt.bar(indices2,est,width=0.5*width,color = 'r',label = "Regressed Values")
        #    plt.bar(indices3,diff,width=0.25*width,color='g',label = "Cost")
        #    plt.title("Single comparison of Actual vs estimated Output (denormalized)")
        #    plt.legend()
        #    plt.show()
            
        #    plt.bar(indices,sigmaList,label = "Cost Variance")
        #    plt.title("Variance of Cost")
        #    plt.legend()
        #    plt.show()
           

                             
    t_end = time.process_time()
    print("Total Training Network time = ",t_end-t_start)

#Try giving the network a unadultereed audio file input, and then see how the model and network stack up to replicating it.
#Open and split the test audio into even size chunks
    #print("Writing and synthesizing audio")
    #testAudio = openSplit('Mrgan_Free.wav', inSize)
    #outAudio = list()
    ##for a in testAudio:
    
    #for i in range(10):
    #    testParams = sess.run(outEstimate,feed_dict={inputAudio: np.array(testAudio[i]).reshape(1,inSize)})
    #    AudioLists = tdg.generateAudio(testParams,spkr)
    #    for n in AudioLists:
    #        outAudio.extend(n)

    #TASample =  testAudio[1]
    #plt.plot(outAudio[1],'b',label = "Model output")
    #plt.plot(TASample,'r', label = "Test Audio")
    #plt.title("comparison of generated vs actual audio")
    #plt.legend()
    #plt.show()
    #vtl.list_to_wave('Mrgan_Free_out.wav',outAudio,2,srate)

    coord.request_stop()
    coord.join(threads)

#plt.plot(costList,'b')
#plt.plot(ACL,'r')
#plt.title("Cost and the Average Cost per 10 steps")
#plt.show()








        

