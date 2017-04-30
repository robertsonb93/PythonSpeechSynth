import TextScraper as sc
import vtlPythonAPI as vtl
import random as rand
import time

def genGlottis(NumStates,glottisCount,GlottisParams):
    print("Generating Glottis")
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

    print("Glottis complete.")
    return glottisStates

#Instead of generating vocal tract parameters on our own, 
#lets just use those(about 64) defined in the speaker file
#This Function will give the VTP parameters for each different shape, the name of the shape is not important
#for the training data and is discarded.
def genVTP(spkr,VTPsize):
    VocalShapes = sc.GetShapesfromSpeaker(spkr)  
    shapeParams = list()
    for shape in VocalShapes:
        shapeParams.append(vtl.getVocalTractParamsfromShape(VTPsize,shape,False)[0])
    
    return shapeParams

def genFrame(count):
        #We generate our needed variations for frames and lengths of each frame
    numFramesVariants = list()
    frameRateVariants = list()
    shift = 1/(count/2)
    for i in range(count):      
        numFramesVariants.append(i+1)  # we will see (1,5) for count = 5
        frameRateVariants.append( (1 / (i+1) + shift)) #we will see (1.4,0.6) for count = 5

    return [numFramesVariants,frameRateVariants]


#This function will produce a variety of tubelength sequences and some similar incisor distances.
#This assumes the incisor dist to be the same as the length of the tubes.
#im not actually sure if this is correct, but i believe it is quite close.
#We generate the lengths for each tube (these seem to follow the pattern of progressively shrinking)
#This portion is iffy, its easy to create far to many, and I dont a good guideline for setting the variation of them
def genTubeLen(minLen,maxLen,tubeSectionCount,variants):
    TubeLengthSets = list() #this will be length  tube section Count
    incisorDistVariants = list()
    for v in range(variants):
        PrevLen = maxLen
        currPerm = list()
        currPerm.append(PrevLen)
        for tubeSect in range(tubeSectionCount-1):
            currPerm.append(PrevLen)
            PrevLen = rand.uniform(minLen,PrevLen) #enforce that no tube is longer then the previous, we can end up will all being minimally short.
        TubeLengthSets.append(currPerm)
        incisorDistVariants.append(sum(currPerm))

    return [TubeLengthSets,incisorDistVariants]

#This function is the entry point for generating test data parameters,
# it will call appropriate sub functions to build the needed test sets.
#The following data is generated here and returned, and can be used in the generating audio.
    #   Values for prev state vocalTractParam (VTP)
    #   Values for prev state GlottisParam
    #   Number of Frames for the sound
    #   How fast the Frames play
    #   Lengths for each tube in meters
    #   Distance from the glottis to the incisors (front teeth) in meters
    #   Then area in m^2 of the velum
def generateValues(spkr,numGlottisStates,maxFrames,tubeVariants,minTube,maxTube,velumVariants):
    vtl.initSpeaker(spkr,True)
    rand.seed(time.time())

    #we get the audio sampling rate,
    #The number of tube sections that the speaker has.
    #the number of parameters the speaker uses to perform any sound
    #the number of parameters(based on model) the speaker has for their glottis
    [audioSampleRate,tubeSectionCount,vocalTractCount,glottisCount] = vtl.getSpeakerConstants()

    VocalTractParams = vtl.getTractParams(vocalTractCount)
    GlottisParams = vtl.getGlottisParams(glottisCount)

    #We Can generate Test data by making permutations of the shapes from the vocal tracts and glottis
    
    glottisStates = genGlottis(numGlottisStates,glottisCount,GlottisParams)
    VTPStates = genVTP(spkr,vocalTractCount)
    [numFrames,frameRates] = genFrame(maxFrames)
    [incisors,TubeLengthSets] = genTubeLen(minTube,maxTube,tubeSectionCount,tubeVariants)

    #we generate some velum openings in m^2, these are supposed to be between the 16th and 17th tubes,
    #which  from examples i have seen range in area of about 0.65 to 4.5 cm^2 or 0.045m^2
    velum = list()
    for i in range(velumVariants):
        velum.append(rand.uniform(0.0065,0.045))



    print("Total glottal states: ", len(glottisStates))
    print("Total vocal tract states: ",len(VTPStates))
    glotVTLsize = (len(glottisStates)*len(VTPStates))**2

    print("Generating ",glotVTLsize, " glottis,VTP variants (with old and new state)")
    print("Total numFrames && frameRate variations ", len(numFrames))
    print("Total incisor distance variants ",len(incisors))
    print("Total TubeLength variations ",len(TubeLengthSets))
    print("Total velum variations ",len(velum))

    print("Total sound permutations possible is ",len(velum)*len(incisors)*
          len(TubeLengthSets)*len(numFrames)*len(frameRates)*glotVTLsize)

    return [glottisStates,VTPStates,numFrames,frameRates,incisors,TubeLengthSets,velum]

#def generateAudio