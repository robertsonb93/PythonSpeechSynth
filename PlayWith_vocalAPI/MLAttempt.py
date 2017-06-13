import sys
import vtlPythonAPI as vtl
import tensorflow as tf
import numpy as np
import TextScraper as sc
import TrainingDataGen as tdg
import matplotlib.pyplot as plt
import time
import csv



#********************************* DEFINITIONS************************************#
#*********************************************************************************#

#This function is called by newTrainingData()
#This function will write the parameters (about 144) and the produced audio to a csv file
#The columns are organized as the parameters first(the networks output) 
#then follows with the produced audio (the networks input)
def writeCSV(trainSet,filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for t in trainSet:
            gltSize = int(len(t.glottis)/t.numFrames)
            vSize = int(len(t.vtp)/t.numFrames)
            tSize = int(len(t.tubeLengths)/t.numFrames)
            #Incisor and velum are size numFrames
            for n in range(t.numFrames-1):
                glt = [str(g) for g in t.glottis[n*gltSize:(n+1)*gltSize]]
                gltEnd = [str(g) for g in t.glottis[(n+1)*gltSize:(n+2)*gltSize]]
                vtp = [str(v) for v in t.vtp[n*vSize:(n+1)*vSize]]
                vtpEnd = [str(v) for v in t.vtp[(n+1)*vSize:(n+2)*vSize]]
                tbel = [str(tl) for tl in t.tubeLengths[(n)*tSize:(n+1)*tSize]]
                tbelEnd= [str(tl) for tl in t.tubeLengths[(n+1)*tSize:(n+2)*tSize]]
                inc = [str(t.incisor[n])]
                incEnd = [str(t.incisor[n+1])]
                vel = [str(t.velum[n])]
                velEnd = [str(t.velum[n+1])]
                aud = [str(a) for a in t.audio[n*t.samplesPerFrame:(n+1)*t.samplesPerFrame]]
                
                row = glt + gltEnd + vtp + vtpEnd + tbel + tbelEnd + inc + incEnd + vel + velEnd + aud
                writer.writerow(row)
        print("Completed Writing Csv")


#The number of desired examples
#the framerate of the speaker(how fast they talk, i am working with 4-8 per second)
#how many frames to generate per synthesis (they are still split into a start-end pair for the data)
#the speaker file to be used, (see .speaker files)
#the name of the file the data should be written too.
def newTrainingData(examples,frameRate,numFrames,spkr,outputFile):
    maxLen = 0.005 #if they all come out at max length it will be 20cm
    minLen = 0.0025 #if they all come out at min length it will be 10cm
    minFrames = numFrames # min is 2 frames (start and end of an audio)
    maxFrames = numFrames

    #Produce the parameters we use for the generating speech.
    paramset = tdg.generateValues(spkr,examples,minFrames,maxFrames,frameRate,minLen,maxLen)

    trainSet = list()
    count = 0
    t_start = time.process_time()
    avg = 0
    for p in paramset:
        count = count +1
        start = time.process_time() 
        #Produce each parameter set into an audio file so we know the outcome of it.

        trainSet.append(tdg.generateAudio(p,spkr))

        t = time.process_time()-start
        avg = (t+ (count-1)*avg)/count
        print("It took ",t, "to finish one synthesis. Estimated ",avg*(len(paramset)-count), " seconds remaining.")

    t_end = time.process_time()
    print("Total Training generation time = ",t_end-t_start)
    writeCSV(trainSet,outputFile)

#Given a list of param lists will parse the needed params out and feed them into the 
#the synthesizer and then piece the produced audio together and return a list containing the audio
def genAudio(testParams,inSize):
    [srate,tubeSecs,vParam,gParam] = vtl.getSpeakerConstants()
    numFrames = len(testParams)+1
    FPS = 6 #TODO: Calculate this better

    #Grab all the start untill the last element, then add the end
    vOffset = gParam*2
    tOffset = vOffset + vParam*2
    incOffset = tOffset + tubeSecs*2
    velOffset = incOffset + 2 
    glottis=list()
    vtp = list()
    tubeLengths = list()
    incisor = list()
    velum = list()

    for i in range(len(testParams)-1):
        glottis.extend(testParams[i][0:gParam])
        print("glottisLen1: ",len(glottis))

        vtp.extend(testParams[i][vOffset:vOffset+vParam])
        tubeLengths.extend(testParams[i][tOffset:tOffset+tubeSecs])
        incisor.append(testParams[i][incOffset])
        velum.append(testParams[i][(velOffset)])
    #Append The final frames of the audio (all the other ends were the starts of the next frames)
    i = len(testParams)
    glottis.extend(testParams[i-1][gParam:vOffset])
    print("glottisLen2: ",len(glottis))
    vtp.extend(testParams[i-1][vOffset+vParam:tOffset])
    tubeLengths.extend(testParams[i-1][tOffset+tubeSecs:tOffset+tubeSecs*2])
    incisor.append(testParams[i-1][incOffset+1])
    velum.append(testParams[i-1][velOffset+1])


    [a_pre, samples, areas,articulators] = vtl.synthSpeech(vtp,glottis,tubeSecs,numFrames,FPS,srate)
    samplesPerFrame = inSize
    areas = [a / 100 for a in areas] #Cause the return type is in the wrong scale

    #these increment so that it starts from 0:n then n:n*2, hence the prev end is the next start
    audioFrames = [a_pre[x:x + samplesPerFrame] for x in range(0, len(a_pre),samplesPerFrame)]
    areaFrames = [areas[x:x + tubeSecs] for x in range(0,len(areas),tubeSecs)]
    lenFrames = [tubeLengths[x:x + tubeSecs] for x in range(0,len(areas),tubeSecs)]
    articFrames = [articulators[x:x + tubeSecs] for x in range(0,len(articulators),tubeSecs)]

    outAudio = list()
    for i in range(numFrames-1):
        print("Number of frames:" ,numFrames)
        artics = ''.join(art for art in articFrames[i])
        artics = bytearray(map(ord,artics))
        window = i * gParam
        print("Window is: ", window)
        print("Glottis len3 is: ",len(glottis))
        outAudio = outAudio + vtl.addToSynthesis(lenFrames[i],areaFrames[i],artics,incisor[i],velum[i],glottis[window + 5],glottis[window:window + gParam],audioFrames[i])
    return outAudio
  

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
    #audio = tf.stack(np.asarray(a))

    #audioBatch = tf.train.batch([audio],name = 'AudioTestBatch')
    #return audioBatch

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
numFrames = 3

#USe this to generate new training data
newTrainingData(10000,framerate,numFrames,spkr,'Train2.csv')

#lets see if we can get it to match a single sound. with only a single frame transition of audio.
vtl.initSpeaker(spkr,False)
[srate,tubecount,vtcount,glotcount] = vtl.getSpeakerConstants()
inSize = int(srate/(framerate)) #I know this works with frameRate of 4

#Glottis + vtp + tubeLengths + incisor + velum
outSize = (glotcount + vtcount + tubecount + 2) * 2#Times two for a start and end

#Here We give TF a list of the files it can read to get our values from

files = ["Train2.csv","Train1.csv","Train.csv"]
batchSz = 200
audio,params,audioBatch,paramBatch= readFilesnBatch(files,batchSz,inSize,outSize)  

t_start = time.process_time()
with tf.Session() as sess:
    print("Entered Session")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    inputAudio = tf.placeholder(tf.float32,shape=[None,inSize],name = "inputAudio")
    outActual = tf.placeholder(tf.float32,shape=[None,outSize],name = "outActual")

    #create Weights (this is basically the network)
    h_size = 2 * 144
    W = list()
    W.append(weight_variable((inSize,h_size)))
    for s in range(10):
        W.append(weight_variable((h_size,h_size)))
    W.append(weight_variable((h_size,outSize)))

    #Forward propagation(How to calculate from input to output)
    outEstimate = forwardprop(inputAudio, W)

    #Back Progoation(How to correct the Error from the estimate)
    cost =tf.reduce_mean(tf.squared_difference(outActual,outEstimate))
    train_step = tf.train.AdamOptimizer(1).minimize(cost)


    init = tf.global_variables_initializer()
    sess.run(init)
    print("Session Init complete")
    AC = 0
    ACL = list()
    costList = list()
    epochs = 100
    for i in range(epochs):  #It seems this will stop on the min(number of lines or range)
        print(i," of ",epochs)
           
        x,y = sess.run([audioBatch,paramBatch])
       #normalizeOut(y,batchSz,tubecount,vtcount,glotcount)
        _, Cst = sess.run([train_step,cost], feed_dict={inputAudio: x, outActual: y})

        AC = (Cst + (i)*AC)/(i+1)
        ACL.append(AC)
        print("Average Cost: ",AC)
        costList.append(Cst)

        if (i>epochs-2):
            oE = sess.run(outEstimate,feed_dict={inputAudio: x})
           # denormalizeOut(oE,batchSz,tubecount,vtcount,glotcount)
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








        

