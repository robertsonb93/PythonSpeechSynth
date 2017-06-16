import TextScraper as sc
import vtlPythonAPI as vtl
import random as rand
import time
import csv


#GlottisCount is the number of parameters the glottis model has
#Glottis params contains the name,min,max,neutral values for each glottis parameter
def genGlottis(glottisCount,GlottisParams):
    ret = list()
    for i in range(2):
        for gIndex in range(glottisCount):
            min = GlottisParams[1][gIndex]
            max = GlottisParams[2][gIndex]
            ret.append(rand.uniform(min,max))
            
    return ret

#Instead of generating vocal tract parameters on our own, lets just use those(about 64) defined in the speaker file
#Randomly choose one from the return of the speakers defined shapes
#It should be noted, we can also consider getting the min,max for each vocaltract param via vtl.getTractParams(), then produce random values.
def genVTP(spkr,VTPsize):
    VocalShapes = sc.GetShapesfromSpeaker(spkr)  
    ret = list()
    for i in range(2):
        ret += vtl.getVocalTractParamsfromShape(VTPsize, rand.choice(VocalShapes),False)[0]
    return ret

#This function will a list of tubeLengths and a comparable incisor distance
#It is worth researching if tube lengths should be stricly decreasing or not
def genTubeLen(minLen,maxLen,tubeSectionCount):
    tubes = [0] * (2*tubeSectionCount) #Create a list of tubes lengths
    incisorDist = [0] * 2
    for i in range(2):
        dist = 0
        for t in range(tubeSectionCount):
            tubes[t + (tubeSectionCount * i)] += rand.uniform(minLen,maxLen)


            dist += tubes[t + (i*t)]
        incisorDist[i] = dist + rand.uniform(-minLen,minLen) #To add a little variation aside form just distance of tubes
    
    return tubes + incisorDist

#Frankly I dont know what a good range for this is, so lets just try this for now, This is cm^2
#This function is mostly for clarity, and as a placeholder for any more intricate generating that may be discovered
def genVelum():
    return [rand.uniform(0.5,10),rand.uniform(0.5,10)]

def genAspStr():
    return [rand.uniform(-40,0),rand.uniform(-40,0)]


#Function will call appropriate functions to generate the lists that have the start and stop frame
#for sound synthesis, the order of the parameters is
    #   glottis start, end
    #   vocal tract start, end
    #   Lengths for each tube in centimeters start and end
    #   Distance from the glottis to the incisors (front teeth) in centimeters start and end
    #   Then area in cm^2 of the velum start and end
    #   The aspiration strength in dB 
def generateValues(spkr,sets,minTube,maxTube):

    vtl.initSpeaker(spkr,True)
    counts = vtl.getSpeakerConstants() # returns sampleRate, tubeSection count, VocalTract count, Glottis count
    glottisVals = vtl.getGlottisParams(counts[3])
   

    rand.seed(time.time())
    start = time.process_time()
    retparams = list()

    for c in range(sets):#Generate for the desired number of parameter sets
        retparams.append(list())
        retparams[c] += genGlottis(counts[3],glottisVals)#providing numGlottis features and the min,max values of them
        retparams[c] += genVTP(spkr,counts[2])#Providing the speaker filename, and number of vocal tract features
        retparams[c] += genTubeLen(minTube,maxTube,counts[1]) # This return tubes then incisor dist,
        retparams[c] += genVelum() #this returns start,end velum states
        retparams[c] += genAspStr() #A start,end aspiration strength

    end = time.process_time()
    print("Generated ",sets," parameter sets in ", end - start, " seconds.")
    vtl.CloseSpeaker()
    return retparams


#provided a list of lists of parameters, each parameter list contains in this order
#glottis start,end
#vocal tract start,end
#tube lengths, start,end
#incisor distance start,end
#velumArea start,end
#All values must be at the centimeter scale.
#Will return a list of lists of the audio synthesized,
def generateAudio(paramSets,frameRate,spkr):
    vtl.initSpeaker(spkr,False)
    [srate,tubeSecs,vParam,gParam] = vtl.getSpeakerConstants()
    numFrames = 2
    audioRet = list()

    times = [0] * 10
    count = 0
    print("Generating Audio.")
    remainingItems = len(paramSets)
    for pSet in paramSets:
        startTime = time.process_time() 
        gltWindow = gParam*numFrames
        vWindow = vParam*numFrames + gltWindow
        tubeWindow = (tubeSecs*numFrames)+vWindow
        incWindow = numFrames + tubeWindow
        velWindow = numFrames+incWindow
        aspStrWindow = numFrames+velWindow

        glottis = pSet[0:gltWindow]
        vtp = pSet[gltWindow:vWindow]
        tubeLens = pSet[vWindow:tubeWindow]
        incisors = pSet[tubeWindow:incWindow]
        velum = pSet[incWindow:velWindow]
        aspStr = pSet[velWindow:aspStrWindow]

        vtl.resetSynthesis()
        [audioPre, audioSamplesCreated, tubeAreas, articulators] = vtl.synthSpeech(vtp,glottis,tubeSecs,numFrames,frameRate,srate)
        #we need to convert the artics in the a different format, so we can pass them to addToSynthesis
        artics = bytearray(map(ord,''.join(art for art in articulators)))
        #Since addToSynthesis interpolates between the current frame (or default starting) and the provided next frame, 
        # we hand it dummy audio and the starting state and discard the result.
        tubeLenStrt = tubeLens[0:tubeSecs]
        tubeLenEnd = tubeLens[tubeSecs:tubeWindow]
        tubeAreaStrt = tubeAreas[0:tubeSecs]
        tubeAreaEnd = tubeAreas[tubeSecs:tubeWindow]
        glottisStrt = glottis[0:gParam]
        glottisEnd = glottis[gParam:gltWindow]

        
        vtl.addToSynthesis(tubeLenStrt,tubeAreaStrt,artics,incisors[0],velum[0],aspStr[0],glottisStrt,([0] * audioSamplesCreated)) #Dummy audio attached at the end
        #This is the true audio we are looking for form the parameter set we have
        audioRet.append(vtl.addToSynthesis(tubeLenEnd,tubeAreaEnd,artics,incisors[1],velum[1],aspStr[1],glottisEnd,audioPre))


        #This portion is just to provide feedback on how long to expect the rest of the synthesis to take
        times[count] = time.process_time() - startTime
        remainingItems -= 1
        count += 1
        count %= 10
        if count == 9:
                avg = sum(times)/len(times)
                print("Estimated ", avg*remainingItems, " Seconds till completion for ",remainingItems, " remaining items.")

    vtl.CloseSpeaker()

    return audioRet


#This function is called by newTrainingData()
#This function will write the parameters (about 144) and the produced audio to a csv file
#The columns are organized as the parameters first(the networks output) 
#then follows with the produced audio (the networks input)
def writeCSV(trainSets,audioSets,filename):
    print("Proceeding to write training Data to CSV")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for s in range(len(trainSets)):
            row = [str(val) for val in trainSets[s]]
            row += [str(val) for val in audioSets[s]]
            writer.writerow(row)

        print("Completed Writing Csv")


#The number of desired examples
#the framerate of the speaker(how fast they talk, i am working with 4-8 per second)
#the speaker file to be used, (see .speaker files)
#the name of the file the data should be written too.
def newTrainingData(examples,frameRate,spkr,outputFile):
    maxLen = 5 #if they all come out at max length it will be 20cm
    minLen = 2.5 #if they all come out at min length it will be 10cm
    
    #Produce the parameters we use for the generating speech.
    #This is a list of lists where each sublist is the parameters needed to produce an audio
    paramSets = generateValues(spkr,examples,minLen,maxLen)
    audioSets = generateAudio(paramSets,frameRate,spkr)
    writeCSV(paramSets,audioSets,outputFile)