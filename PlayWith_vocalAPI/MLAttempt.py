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

    stops = tf.placeholder(tf.int32,shape = [None], name = "StopPoints_List")
    max = tf.constant([22050],tf.int32,name = 'Max_audioSize')#Todo Deal with the magive 22050, where do we get this value from? 

    #We need to grab the size of audio here, that way we have the approriate stop point for the audio in the sequence.
    size = tf.expand_dims(tf.size(audio),0,name = 'increase_size_of_size(audio)_for_concat')
    stops = tf.concat([stops,size],0,name = 'Concat_stop_points')#We could also look at setting stops.shape = batchsize, and then using scatter_update

     
    #Then perform padding on it
    padSize = tf.subtract(max,tf.size(audio),name = 'Padsize_SubtractCalc')
    padding = tf.fill(padSize,0.0,name = 'FillAudioPadding')
    padAudio = tf.concat([audio,padding],0,name= 'AudioPaddingConcat')
    print(tf.rank(padAudio,name = 'PadAudioRank'))
    padAudio = tf.reshape(padAudio,max,name = 'Reshaping_paddedAudio') #Todo:Make sure that we are actually padding things here correctly. Is it being padded with Zeros to 22050?


    #then feed to the batch
    min_after_dequeue = 1000
    capacity = min_after_dequeue+3 * batchSize
    audioBatch,paramBatch = tf.train.shuffle_batch([padAudio,params],batch_size=batchSize,capacity=capacity,min_after_dequeue=min_after_dequeue,allow_smaller_final_batch=True)
   # audioBatch,paramBatch = tf.train.batch([audio,params],batch_size=batchSize,capacity = capacity,allow_smaller_final_batch=True,dynamic_pad=True)
    return stops,audioBatch,paramBatch

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

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def weight_summaries(var):
  sumry = list()
  #Attach a lot of summaries to a Tensor (for TensorBoard visualization).#
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    sumry.append(tf.summary.scalar('mean', mean))
    with tf.name_scope('var_stddev'):
      variance = tf.reduce_mean(tf.square(var-mean))
      stddev = tf.sqrt(variance)
    sumry.append(tf.summary.scalar('stddev', stddev))
    sumry.append(tf.summary.scalar('variance',variance))
    sumry.append(tf.summary.scalar('max', tf.reduce_max(var)))
    sumry.append(tf.summary.scalar('min', tf.reduce_min(var)))
    sumry.append(tf.summary.histogram('Weight_Values', var))
  return sumry

def variable_summary(var):
    sumry = list()
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        sumry.append( tf.summary.scalar('mean',mean))
        with tf.name_scope('var_stddev'):
            variance = tf.reduce_mean(tf.square(var-mean))
            stddev = tf.sqrt(variance)
        sumry.append(tf.summary.scalar('stddev', stddev))
        sumry.append(tf.summary.scalar('variance',variance))
        sumry.append(tf.summary.scalar('max', tf.reduce_max(var)))
        sumry.append(tf.summary.scalar('min', tf.reduce_min(var)))
    return sumry

#I hope this is correct
#Review how to properly build this
#Contains the activation functions for each layer, including the input,hidden,output layers
#X is the input Data, W[i] are the weights of the i'th layer
#yHat is the prediction/regression result of the network
def forwardprop(X, W):
    sumry = list()
    with tf.name_scope("InLayer"):
        with tf.name_scope("Weights"):
            inputLayer = tf.matmul(X,W[0])
            h = tf.nn.sigmoid(inputLayer,name = "InLayer")
            sumry += weight_summaries(W[0])
            sumry.append(tf.summary.histogram("InputLayer",inputLayer))
            sumry.append(tf.summary.histogram("InputLayer_Activation",h))#This is the activation
        with tf.name_scope("InputBias"):
            biasSize = (W[0]).get_shape()[1].value
            b = bias_variable([biasSize])
            h += b
            sumry.append(tf.summary.histogram("InputBias",b))

    count = 0;
    for w in W[1:len(W)-1]:       
        name = "H-Layer_" + str(count)
        count +=1
        with tf.name_scope(name):
             with tf.name_scope("Weights"):
                layer = tf.matmul(h,w)
                h = tf.nn.sigmoid(layer, name = name)
                sumry += weight_summaries(w)
                sumry.append(tf.summary.histogram(name, layer))
                sumry.append(tf.summary.histogram(name + "_activation",h))
             with tf.name_scope("Bias"):
                biasSize = w.get_shape()[1].value #since W has [input,output]
                b = bias_variable([biasSize])
                h += b
                sumry.append(tf.summary.histogram(name+'Bias',b))
    
    with tf.name_scope("OutputLayer"):
        yhat = tf.matmul(h,W[len(W)-1],name = "OutputLayer")

        sumry += weight_summaries(W[len(W)-1])
        sumry.append(tf.summary.histogram("OutputLayer",yhat))

    return yhat, sumry

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
        #TODO: Normalize Framerate

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
    #TODO: Normalize Framerate




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

networkParams = list()
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
max_steps = int(srate)
seq_width = 1
 


#Glottis + vtp + tubeLengths + incisor + velum + aspStrength
outSize = (glotcount + vtcount + tubecount + 3) * 2#Times two for a start and end
outSize +=1 #For the added framerate


    #Here We give TF a list of the files it can read to get our values from, 
    #the proceed to organize them into batches that are sent to TF
maxBatchSz = 100
maxTestBatchSz = 10
trainFiles = list() #Select which files we want to train off of
testFiles= list()
trainStart = 50
trainEnd = 250
testStart = 250
testEnd = 300

for i in range(trainStart,trainEnd): #using files 50-299 as they they are currently at size framerate 4
      trainFiles.append((prefix + str(i) + postfix))
for i in range(testStart,testEnd):
      testFiles.append((prefix+str(i)+postfix))


stops,audioBatch,paramBatch= readFilesnBatch(trainFiles,maxBatchSz,inSize,outSize)
testStops,testAudioBattch,testParamBatch = readFilesnBatch(testFiles,maxTestBatchSz,inSize,outSize)

print("Batching complete")

t_start = time.process_time()
with tf.Session() as sess:

    trainSummaries = list()
    testSummaries = list()
    
    print("Entered Session")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    with tf.name_scope("Input_Audio"):
        inputAudio = tf.placeholder(tf.float32,shape=[None,inSize],name = "inputAudio") #Todo:change 
        trainSummaries.append(tf.summary.audio("Input_Audio",inputAudio,srate,10))
    with tf.name_scope("Actual_output"):
        outActual = tf.placeholder(tf.float32,shape=[None,outSize],name = "outActual")       
        trainSummaries.append(tf.summary.histogram("Real_output_Train",outActual))
        testSummaries.append(tf.summary.histogram("Real_output_Test",outActual))

    #create Weights (this is basically the network)
    networkDepth = 20
    h_size = inSize
    W = list()
    W.append(weight_variable((inSize,h_size)))  
    for s in range(networkDepth):
        W.append(weight_variable((h_size,h_size)))
    W.append(weight_variable((h_size,outSize)))
    print("Weights Generated")

    #Forward propagation(How to calculate from input to output)
    with tf.name_scope("Generated_Output"):
        outEstimate,sumry = forwardprop(inputAudio, W)
        trainSummaries.append(sumry)
        testSummaries.append(sumry)

    trainSummaries.append(tf.summary.histogram("NetworkOutput_Train",outEstimate))
    testSummaries.append(tf.summary.histogram("NetworkOutput_Test",outEstimate))


    #Back Progoation(How to correct the Error from the estimate)
    costList = list([0] * 10)
    costSplit = tf.squared_difference(outActual,outEstimate)
    with tf.name_scope("Cost_scope"):
        cost =tf.reduce_mean(costSplit)#Back prop is here 
        costListTensor = tf.placeholder(tf.float32,[None],name = "Cost_History")
        variable_summary(costListTensor)
        trainSummaries.append(tf.summary.scalar("cost_while_training",cost))
        testSummaries.append(tf.summary.scalar("cost_while_testing",cost))
    print("Completed Cost")

    

    with tf.name_scope("GD_Trainer"):
        learningRate = 0.9
        train_step = tf.train.AdamOptimizer(learningRate).minimize(cost)
 


    epochs = 10
    testFreq = 100
    
    #textTensor = tf.placeholder(networkParams,tf.string,name = "Network_Parameters")
    textTensor = tf.placeholder(tf.string,shape = [14],name = "Network_Parameters")
    textSumry = tf.summary.text(name = "Network_Parameter_Summary",tensor = textTensor)
    networkParams.append(("FrameRate:"+str(framerate)))
    networkParams.append(("Samplerate:"+str(srate)))
    networkParams.append(("InputSize:"+str(inSize)))
    networkParams.append(("OutputSize:"+str(outSize)))
    networkParams.append(("Training Batch Size:"+str(maxBatchSz)))
    networkParams.append(("Testing Batch Size:"+str(maxTestBatchSz)))
    networkParams.append(("Training files:"+str(trainStart)+"-"+str(trainEnd)))
    networkParams.append(("Testing files:"+str(testStart)+"-"+str(testEnd)))
    networkParams.append(("Hidden Layer Width:"+str(h_size)))
    networkParams.append(("Hidden Layers: "+str(networkDepth)))
    networkParams.append(("Learning Rate:"+str(learningRate)))
    networkParams.append(("Epochs Run:"+str(epochs)))
    networkParams.append(("testFreq:"+str(testFreq)))
    

    trainSumry = tf.summary.merge(trainSummaries)
    testSumry = tf.summary.merge(testSummaries)
    textSumry = tf.summary.merge([textSumry],name = "Network_Text_summary")
    writer = tf.summary.FileWriter('.\TensorBoard',sess.graph,filename_suffix = "Run1")
    saver = tf.train.Saver()
    print("Completed Summaries")

    sess.run(tf.local_variables_initializer())
    init = tf.global_variables_initializer()
    sess.run(init)   
    print("Session Init complete")

    for i in range(epochs):
        print(i," of ",epochs)

        if (i%testFreq):
            x,y,currBatchSize = sess.run([audioBatch,paramBatch,tf.shape(paramBatch)[0]])# here we are assigning batchsize components to the None Dimension for the x,y
            normalizeOut(y,currBatchSize,tubecount,vtcount,glotcount)
            def getStopPoints(x):
                stops = list()
                for sample in x:
                    stops.append(len(sample))
                return np.array(stops).reshape(1,len(stops))
            stops = getStopPoints(x)


            if i <1: #This is here to push the initial cost into the cost list, then it can be summarized into the tensor below
                Cst = sess.run(cost,feed_dict={inputAudio: x, outActual: y})
                costList[0] = Cst
                    
            sessList = [costSplit, cost, train_step, trainSumry]
            CstSplit,Cst,_,sumry = sess.run(sessList, 
                                          feed_dict={inputAudio: x, outActual: y, costListTensor: costList},
                                          options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                          run_metadata=tf.RunMetadata()
                                          )
            costList[i%10] = Cst
        else:
            xTest,yTest = sess.run([testAudioBattch,testParamBatch])
            normalizeOut(yTest,maxTestBatchSz,tubecount,vtcount,glotcount)
            sessList = [cost,testSumry]
            Cst,sumry = sess.run(sessList,feed_dict = {inputAudio:xTest, outActual: yTest})
            ##saver.save(sess,'.\TensorBoard\checkpoints',global_step= i)

        writer.add_summary(sumry,i)
                           
        #TODO: Create a better test where we generate the audio and then can compare it to the inputAudio, perhaps every 1000 steps
                             
    t_end = time.process_time()
    tme = t_end-t_start
    print("Total Training Network time = ",t_end-t_start)
    networkParams.append(("Time Running:"+str(tme)))

    #sumry = sess.run([textSumry],feed_dict= {textTensor:networkParams})
    #writer.add_summary(sumry,0)

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







        

