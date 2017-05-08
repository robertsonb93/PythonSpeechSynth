import TextScraper as sc
import vtlPythonAPI as vtl
import random as rand
import time

#NumStates is the number glottis states we want to produce
#GlottisCount is the number of parameters the glottis model has
#Glottis params contains the min,max,neutral values for each glottis parameter
def genGlottis(NumStates,glottisCount,GlottisParams):
    #lets generate a range of possible glottis
    glottisStates = list()
    genParams = list()
    for gc in range(glottisCount):#Iterate across the parameters our glottis model has
        rng = GlottisParams[2][gc] - GlottisParams[1][gc]
        stepsize = rng / NumStates
        currParam = list()

        for step in range(NumStates):
            val = GlottisParams[1][gc] + (step * stepsize)
            if(val <= GlottisParams[2][gc]):
                currParam.append(val)
            else:
                print("Error with parameter ",GlottisParams[0][gc]," being too large (max:actual)",GlottisParams[2][gc],":",val)

        genParams.append(currParam)

    #Huzzah we now have a bunch of auto generated glottis permutations.
    import itertools
    for perm in itertools.product(*(tuple(genParams))): 
        glottisStates.append(list(perm))

    ret = list()
    for n in range(NumStates):
        ret = ret + (rand.choice(glottisStates))
    
    return ret

#Instead of generating vocal tract parameters on our own,
#lets just use those(about 64) defined in the speaker file
#This Function will give the VTP parameters for each different shape, the name
#of the shape is not important
#for the training data and is discarded.
def genVTP(spkr,VTPsize):
    VocalShapes = sc.GetShapesfromSpeaker(spkr)  
    shapeParams = list()
    for shape in VocalShapes:
        shapeParams.append(vtl.getVocalTractParamsfromShape(VTPsize,shape,False)[0])

    
    return shapeParams

#We will generate count many variants, in the range of minFrame to maxFrame
#Furthermore, if not provided with a non-zero frameRate, will be the given
#frameRate
def genFrame(count,minFrame,maxFrame,frameRate=0):
    #We generate our needed variations for frames and lengths of each frame
    #The frameRateVariants could use some work, but i based them on average
    #english syllables, which is 6.19 Syllables per second.
    numFramesVariants = list()
    frameRateVariants = list()
    dist = int((maxFrame - minFrame) / count)

    for i in range(count):      
         numFramesVariants.append(minFrame + (i * dist))
         if(frameRate == 0):
            frameRateVariants.append(rand.uniform(4,8))
         else:
            frameRateVariants.append(frameRate)

    return [numFramesVariants,frameRateVariants]


#This function will produce a variety of tubelength sequences and some similar
#incisor distances.
#This assumes the incisor dist to be the same as the length of the tubes.
#im not actually sure if this is correct, but i believe it is quite close.
#We generate the lengths for each tube (these seem to follow the pattern of
#progressively shrinking)
#This portion is iffy, its easy to create far to many, and I dont a good
#guideline for setting the variation of them
def genTubeLen(variants,minLen,maxLen,tubeSectionCount):
    ret = list() #this will be length tube section Count
    incisorDistVariants = list()
    for v in range(variants):
        dist = 0
        PrevLen = maxLen
        ret.append(PrevLen)
        for tubeSect in range(tubeSectionCount - 1):        
            PrevLen = rand.uniform(minLen,PrevLen) #enforce that no tube is longer then the previous, we can end up will all
                                                   #being minimally short.
            ret.append(PrevLen)
            dist = dist + PrevLen

        incisorDistVariants.append(dist)

    return [incisorDistVariants,ret]



class ParamSet:
    numFrames = 0
    FPS = 0
    glottis = list()
    vtp = list()
    tubeLengths = list()
    incisor = list()
    velum = list()
    samplesPerFrame = 0
    audio = list()


#This function is the entry point for generating test data parameters,
# it will call appropriate sub functions to build the needed test sets.
#If provided a FrameRate, will build the training data without varied
#framerates
#The following data is generated here and returned, and can be used in the
#generating audio.
    #   Values for prev state vocalTractParam (VTP)
    #   Values for prev state GlottisParam
    #   Number of Frames for the sound
    #   How fast the Frames play
    #   Lengths for each tube in meters
    #   Distance from the glottis to the incisors (front teeth) in meters
    #   Then area in m^2 of the velum
def generateValues(spkr,count,minFrames,maxFrames,frameRate,minTube,maxTube):

    vtl.initSpeaker(spkr,True)
    rand.seed(time.time())
    start = time.process_time()
    #we get the audio sampling rate,
    #The number of tube sections that the speaker has.
    #the number of parameters the speaker uses to perform any sound
    #the number of parameters(based on model) the speaker has for their glottis
    [audioSampleRate,tubeSectionCount,vocalTractCount,glottisCount] = vtl.getSpeakerConstants()
    VocalTractParams = vtl.getTractParams(vocalTractCount)
    GlottisParams = vtl.getGlottisParams(glottisCount)
    VTPStates = genVTP(spkr,vocalTractCount) #We only need to call this once since it needs to scrape all the parameters
                                             #from a text file.
    

    retParams = list()
    [numFrames,frameRates] = genFrame(count,minFrames,maxFrames,frameRate)
    for i in range(count):
        newParam = ParamSet()
        newParam.numFrames = numFrames[i]
        newParam.FPS = frameRates[i]
        [newParam.incisor,newParam.tubeLengths] = genTubeLen(newParam.numFrames,minTube,maxTube,tubeSectionCount)
        newParam.glottis = genGlottis(newParam.numFrames,glottisCount,GlottisParams)#This function tries to evenly distribute the states from the glottis
        newParam.vtp = list() #for some reason, it will keep building off the previous newParam .vtp if i
                              #dont do this.
        newParam.velum = list()
        for f in range(newParam.numFrames):
            newParam.vtp = newParam.vtp + (rand.choice(VTPStates))
            newParam.velum.append(rand.uniform(0.0065,0.045))

        retParams.append(newParam)


    vtl.CloseSpeaker()

    end = time.process_time()
    print("Generated ",count," parameter sets in ", end - start, " seconds.")
    return retParams

def generateAudio(param,spkr):
    vtl.initSpeaker(spkr,False)

    [srate,tubeSecs,vParam,gParam] = vtl.getSpeakerConstants()

    [a_pre, samples, areas,articulators] = vtl.synthSpeech(param.vtp,param.glottis,tubeSecs,param.numFrames,param.FPS,srate)
    samplesPerFrame = int(samples / param.numFrames)
    param.samplesPerFrame = samplesPerFrame
    areas = [a / 100 for a in areas]

    #chunk everything into individual frames now.
    audioFrames = [a_pre[x:x + samplesPerFrame] for x in range(0, len(a_pre),samplesPerFrame)]
    areaFrames = [areas[x:x + tubeSecs] for x in range(0,len(areas),tubeSecs)]
    lenFrames = [param.tubeLengths[x:x + tubeSecs] for x in range(0,len(areas),tubeSecs)]
    articFrames = [articulators[x:x + tubeSecs] for x in range(0,len(articulators),tubeSecs)]
 
        #Pass each chunk into the AddToSynth
        #To set the initial state, we need the first hand a blank audio and the
        #first frame elements
    for i in range(param.numFrames):
        artics = ''.join(art for art in articFrames[i])
        artics = bytearray(map(ord,artics))
        window = i * gParam
        if (i == 0):#Provide an empty audio file.
            vtl.addToSynthesis(lenFrames[i],areaFrames[i],artics,param.incisor[i],param.velum[i],param.glottis[5],param.glottis[window:window + gParam],list([0,0,0,0]))
        else:
            param.audio = param.audio + vtl.addToSynthesis(lenFrames[i],areaFrames[i],artics,param.incisor[i],param.velum[i],param.glottis[window + 5],param.glottis[window:window + gParam],audioFrames[i - 1])
  
    vtl.CloseSpeaker()
    return param


    