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
   #     numFramesVariants.append(i+1)  # we will see (1,5) for count = 5
         numFramesVariants.append(2)
         frameRateVariants.append(1)
    #    frameRateVariants.append( (1 / (i+1) + shift)) #we will see (1.4,0.6) for count = 5

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

    return [incisorDistVariants,TubeLengthSets]

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
    glotVTLsize = ((len(glottisStates)**2) *len(VTPStates))

    print("Generating ",glotVTLsize, " glottis,VTP variants (with old and new state)")
    print("Total numFrames && frameRate variations ", len(numFrames))
    print("Total incisor distance variants ",len(incisors))
    print("Total TubeLength variations ",len(TubeLengthSets))
    print("Total velum variations ",len(velum))

    print("Total sound combinations possible is ",len(velum)*len(incisors)*
          len(TubeLengthSets)*len(numFrames)*len(frameRates)*glotVTLsize)

    vtl.CloseSpeaker()

    return [glottisStates,VTPStates,numFrames,frameRates,incisors,TubeLengthSets,velum]



#We need to select a subset of the parameters, since we likely got much more then be used. 
#Returns a list of lists
#each sublist is the parameters used and the audio synthesized from it. order as
#[glottisOld ,glottisNew ,VTP,numFrames ,frameRates ,incisors ,tubeLens ,velum ,audio]
def generateAudio (parameters, sampleCountDesired,spkr):
    #3 main functions used, 
    #vtl.resetSynthesis()
    #vtl.synthSpeech(VTP,glottisParams,40,numFrames,frameRate,sampleRate)
    #vtl.addToSynthesis(tubelengths,tubeAreas,artics,incdist,velum,aspStrength,glottisParams,audio_in)
    vtl.initSpeaker(spkr,False)
    [srate,tubeSecs,vParam,gParam] = vtl.getSpeakerConstants()
    usedList = list()
    #we need to select some parameters, 


    #The parameters in used look like
    #[glottisOld 0,glottisNew 1, ,vtpNew 2,numFrames 3,frameRates 4,incisors 5,tubeLens 6,velum 7]
    for sample in range(sampleCountDesired):
        used = list()
        count =0 
        for p in parameters:
            used.append(rand.choice(p))
            if(count < 1):
                used.append(rand.choice(p)) #This is so we have a second glottis and second VTP for a start/end
            count = count + 1

        vtl.resetSynthesis()
        #Now we can generate the fundamental sound and get the remaining parameters for the tube synth.
        [a_pre,unused,areas,articulators] = vtl.synthSpeech(used[1],used[0],tubeSecs,used[3],used[4],srate)

        #Formating on the articulators and tubeAreas
        artics = ''.join(i for i in articulators)
        artics = bytearray(map(ord,artics))
        tubeAreas = [(x/100) for x in areas] #divide since, it was returned in cm instead of the needed meters

        #dummy sets up the starting state, and the a_post is the actual sound we are interested in.
        dummy = vtl.addToSynthesis(used[6],tubeAreas,artics,used[5],used[7],used[0][5],used[0],a_pre)
        a_post = vtl.addToSynthesis(used[6],tubeAreas,artics,used[5],used[7],used[1][5],used[1],a_pre)
        used.append(a_post)
        
        print("Finished synthesizing ",sample+1, " of ", sampleCountDesired)
        usedList.append(used)
        
    
    return usedList


    