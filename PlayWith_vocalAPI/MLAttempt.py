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
#the csv columns are organized as the output first (about 146 columns) then the audio input
#In our synthesizer model the params are input and then generate an audio

def readFiles(files,inSize,outSize):
    print("Reading files")
    filename_queue = tf.train.string_input_producer(files)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [tf.constant([0.0],dtype = tf.float32) for x in range(outSize + inSize)]  
      
    #Features are the last inSize columns(the audio)
    row = tf.decode_csv(value, record_defaults=record_defaults,name = "Decode_CSV_op")
    params = tf.stack(row[:outSize])
    audio = tf.stack(row[outSize:]) #TODO: I dont think this will result in shuffled inputs anymore, so we may need to look at reading into a shuffle queue, then feeding those to the batch.Feed queue the row, then proceed.

    #TODO: Does Key need a new naming scheme?
    print("Completed Reading files :: produced key,params,audio")
    return key,params,audio

#Creates a special batch that allow us to splits our audio sequences into smaller sequences to be fed to our RNN
def createBatch(batchSize,key,params,audio,initState,unroll,num_enqueue_threads = 3):

    print("Creating Batch")
    sequence = {"audio":audio}
    context = {"params":params}

    initial_states = initState
    capacity = batchSize * num_enqueue_threads * 20 #TODO: Evaluate this and decide if its appropriate

    batch = tf.contrib.training.batch_sequences_with_states(
        input_key=key,
        input_sequences=sequence,
        input_context=context,
        initial_states=initial_states,
        num_unroll=unroll,
        batch_size=batchSize,
        num_threads=num_enqueue_threads,
        input_length = tf.size(params,name = "ParamsSize_createBatch"),
        capacity=capacity
        )
    #audioBatch = batch.sequences["audio"]
    #paramBatch = batch.context["params"]

    print("Completed Creating Batch")
    return batch

#Will format the the input and output appropriately, and also produce the needed sequenceLengths for already padded audio data
def seqLen_input_output(paramBatch,audioBatch,unroll,maxBatchSz,srate,outSize):
    with tf.name_scope("SequenceLengths"):
        print("Creating Sequence Length Stop Point List")
        stops = [tf.cast(tf.ceil(tf.divide(srate,pb[0][146])),dtype = tf.int32) for pb in tf.split(paramBatch,maxBatchSz,0)]
        #for s in stops:
        #    print(sess.run(s)) #Put here for testing purposes # TODO: why are we getting 11 elements in stops??

    with tf.name_scope("Input_Audio"):
        seq_width = 1 #TODO:Determine if this value is needed, and if it should be defined outside the function
        #input is a list, where each n-th element is the n'th time step for an entire batch. 
        print("building Reshaped input")
        inputAudio = [tf.reshape(i,[maxBatchSz,seq_width]) for i in tf.split(audioBatch,unroll,1)] #Todo This needs to be the same as the unroll for the audioBatch
        print("Input is built")

    with tf.name_scope("Actual_output"):
        print("Building output-Placeholder")
        outActual = tf.placeholder(tf.float32, shape=[None,outSize],name = "outActual")#TODO: Need to feed it paramBatch? 

    return stops,inputAudio,outActual

#Will create a MultiRNNCell populated with cells of the given width and at the given layers deep
def GenRNNCells(maxBatchSz,cellWidth, cellLayers=1):
    #create out RNN cells
    print("Creating RNN Cells")
    LSTMStack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(cellWidth) for _ in range(cellLayers)]) #TODO: WORk this back in?

    initState_values = tf.zeros((cellWidth),dtype = tf.float32) #TODO: this wiull probably need some reworking to work with cellLAyers
    initStates = {"lstm_state":initState_values}

    print("RNN Cells Built")

    #TODO: Generate our state_names in here as well?


    return LSTMStack,initStates

def genRNN(cell,input,batch,stops):
    print("Building RNN")

    #TODO: how to use for a tester as well? use a feed to placeholder? PErhaps make it apart of creating the batch
    rnnOut,states = tf.contrib.rnn.static_state_saving_rnn(
        cell,
        input,
        state_saver=batch,
        sequence_length = stops,
        state_name=("lstm_state","lstm_state"))
        #state_name = "lstm_state"


    #val = tf.transpose(rnnOut,[1,0,2])
    #last = tf.gather(val, int(val.get_shape()[1]) - 1)#Todo: I believe we are collecting the wrong values here, and its creating out output to be [unroll,params] instead of [batch,params]

    last = rnnOut[len(rnnOut)-1]#We collect the states from the last time series?
    print("Completed Building RNN")
    
    return last#Todo: remove val and rnnOut, it is for debugging

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
                    h += b #Can we just use += on a tf sigmoid?
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



#*********************************************************************************#
#*********************************************************************************#
#*********************************************************************************#
#******************************END DEFINITIONS************************************#
#*********************************************************************************#
#*********************************************************************************#
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
framerate = 1

trainFiles = list()
buildFiles =list()
postfix = ".csv"
prefix = "Train"
for i in range(392,400):
    buildFiles.append( (prefix + str(i) + postfix))
genData = False # <----------Generate new training data
if genData:
    ExamplesPerFile = 1000
    for f in buildFiles:
        tdg.newTrainingData(ExamplesPerFile,spkr,f)


vtl.initSpeaker(spkr,False)
[srate,tubecount,vtcount,glotcount] = vtl.getSpeakerConstants()
inSize = int(srate/(framerate)) #I know this works with frameRate of 4 and 1
max_steps = int(srate)
#seq_width = 1
 
#Glottis + vtp + tubeLengths + incisor + velum + aspStrength
outSize = (glotcount + vtcount + tubecount + 3) * 2#Times two for a start and end
outSize +=1 #For framerate


    #Here We give TF a list of the files it can read to get our values from, 
    #the proceed to organize them into batches that are sent to TF
maxBatchSz = 100
maxTestBatchSz = 100
trainFiles = list() #Select which files we want to train off of
testFiles= list()
trainStart = 0
trainEnd = 350
testStart = 350
testEnd = 400

for i in range(trainStart,trainEnd): 
      trainFiles.append((prefix + str(i) + postfix))
for i in range(testStart,testEnd):
      testFiles.append((prefix+str(i)+postfix))


t_start = time.process_time()
with tf.Session() as sess:
    print("Entered Session")

    trainSummaries = list()
    testSummaries = list()


    #Use this to dtermine how wide and how many layers our RNN should be
    cellDepth = 1
    cellWidth = 147*2
    cell,initState = GenRNNCells(maxBatchSz,cellWidth,cellDepth) #cell will goto the rnn, initState will goto createBatch

    miniSequenceSize = 150
    #Read a list of CSV files, and prepare our outputs and inputs
    key,params,audio = readFiles(trainFiles,inSize,outSize)

    #Create a state saving batch from our input & output we read from files
    batch = createBatch(maxBatchSz,key,params,audio,initState,miniSequenceSize,10)

    #Format and gather our sequence lengths, inputs, and output placeholder
    stops, input, outActual = seqLen_input_output(batch.context["params"],batch.sequences["audio"],miniSequenceSize,maxBatchSz,srate,outSize)
 
    #create the rnn
    rnnOut = genRNN(cell,input,batch,stops)

    #create Weights

    networkDepth = 20
    print("Creating Weights")
    h_size = 147*2
    W = list()
    W.append(weight_variable((cellWidth,h_size)))  
    for s in range(networkDepth):
        W.append(weight_variable((h_size,h_size)))
    W.append(weight_variable((h_size,outSize)))
    print("Weights Generated")

    #Forward propagation(How to calculate from input to output)
    with tf.name_scope("Generated_Output"):
       
        outEstimate,sumry = forwardprop(rnnOut, W)
        trainSummaries.append(sumry)
        testSummaries.append(sumry)

    print("Forwardprop set")

    trainSummaries.append(tf.summary.histogram("NetworkOutput_Train",outEstimate))
    testSummaries.append(tf.summary.histogram("NetworkOutput_Test",outEstimate))


    #BackPropogation(How to correct the Error from the estimate)
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
    print("Completed Optimizer")
 
    epochs = 1000
    testFreq = 100

    trainSumry = tf.summary.merge(trainSummaries)
    testSumry = tf.summary.merge(testSummaries)
    writer = tf.summary.FileWriter('.\TensorBoard',sess.graph,filename_suffix = "Run1")
    #saver = tf.train.Saver()
    print("Completed Summaries")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.local_variables_initializer())#TODO: do we need both of these intializations?
    init = tf.global_variables_initializer()
    sess.run(init)   
    print("Session Init complete")

    for i in range(epochs):
        print(i," of ",epochs)       

        #if (i%testFreq):
        if (True):
            y = sess.run(batch.context["params"])
           # normalizeOut(y,maxBatchSz,tubecount,vtcount,glotcount)

            #We need to grab some training batch audio, and the same for training params

            if i <1: #This is here to push the initial cost into the cost list, then it can be summarized into the tensor below
                Cst = sess.run(cost,feed_dict={outActual: y})
                costList[0] = Cst
                    
            sessList = [costSplit, cost, train_step, trainSumry]
            #sessList = [costSplit,cost,train_step]
            CstSplit,Cst,_,sumry = sess.run(sessList, 
                                          feed_dict={outActual: y, costListTensor: costList},
                                          options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                          run_metadata=tf.RunMetadata()
                                          )
            costList[i%10] = Cst
            print("current cost: " + str(Cst))
        #else:
          #  xTest,yTest = sess.run([testAudioBatch,testParamBatch])
          #  normalizeOut(yTest,maxTestBatchSz,tubecount,vtcount,glotcount)
          ##  sessList = [cost,testSumry]
          #  sessList = [cost]
          #  Cst,sumry = sess.run(sessList,feed_dict = {outActual: yTest})
          #  ##saver.save(sess,'.\TensorBoard\checkpoints',global_step= i)

            writer.add_summary(sumry,i)
                           
        #TODO: Create a better test where we generate the audio and then can compare it to the inputAudio, perhaps every 1000 steps
                             
    t_end = time.process_time()
    tme = t_end-t_start
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







        

