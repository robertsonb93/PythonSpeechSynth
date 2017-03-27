import sys
import vtlPythonAPI as vtl
import random
import time
random.seed(time.time())

def generateTubeLengthsnAreas():
    lengths = list()
    areas = list()
    for i in range(40):
        lengths.append(random.uniform(0.001,4))
        areas.append(random.uniform(0.01,4))
    return (lengths,areas)

def generateArticulates():
    s = "TILN"
    options = bytearray()
    options.extend(map(ord, s))
    articulates = list()
    for i in range(40):
        articulates.append(random.choice(options))
    return articulates

#glottal_rest_displacement_cm, subglottal_pressure_Pa, F0_Hz are the parameters that i know of
def generateGlottis(numGlottis,numFrames):
    glottis = list()
    #for now i believe numGlottis will always be 3.
    for i in range(numFrames):
        glottis.append(int(random.uniform(0.01,20)))
        glottis.append(int(random.uniform(0,10000)))
        glottis.append(int(random.uniform(0,120)))
    return glottis

def generateVocalTract(numVocalTractParams,numFrames):
    params = list()
    for i in range (numVocalTractParams*numFrames):
        params.append(random.uniform(0.01,10))
    return params


spkr ="test1.speaker"
audio = vtl.wav_to_list("Example1-Sprachsynthese-orig.wav")

##vtl.test1(spkr,audio)
##vtl.test2(spkr,audio)

vtl.initSpeaker(spkr,True)
print(vtl.GetAPIVersion())

speakerConsts = vtl.getSpeakerConstants()
sampleRate = speakerConsts[0]
numberTubeSections = speakerConsts[1]
numVocalTractParams = speakerConsts[2]
numGlottisParams = speakerConsts[3]


vocalTractParams = vtl.getTractParams(speakerConsts[2])
#print(vocalTractParams[0])

GlottisParams = vtl.getGlottisParams(speakerConsts[3])
#print(GlottisParams[0])

for i in range(86):
    shapeParams = vtl.getVocalTractParamsfromShape(numVocalTractParams,vocalTractParams[0][i],False)
    if(shapeParams[1] == 0):
        print(vocalTractParams[0][i])#With the anomaly of the 2, it appears that only vowels are defined?

VTPLite = [shapeParams[0][i] for i in range(5)]
magPhase = vtl.getTransferFunctions(VTPLite, 500)#I think this is what this expects, but Im not exactly sure.

vtl.resetSynthesis()

print("Adding synthesis")
#For AddToSynthesis we need to generate some tubeListLengths in meters
for i in range(15):
    tubes = generateTubeLengthsnAreas()
    articulates = generateArticulates()
    incisorGlottisDist = random.uniform(0,1)
    velumArea = random.uniform(0.1,0.35)
    pressure = int(random.uniform(0,3000))
    newGlottisnAudio = vtl.addToSynthesis(sampleRate,tubes[0],tubes[1],articulates,incisorGlottisDist
                                          ,velumArea,pressure,numGlottisParams )
    print(i) #And add to synthesis does a whole lot of zeros for the returns values


#Im not quite sure the relationship between addtoSynthesis and synthblock(inside synth speech)
#but trying to run add to synthesis after synthblock will crash python.

numFrames = 30
frameRate = 1

glottisParams = generateGlottis(numGlottisParams,numFrames)
VTP = generateVocalTract(numVocalTractParams,numFrames)

synthSpeechRet = vtl.synthSpeech(VTP,glottisParams,40,numFrames,frameRate,sampleRate)

vtl.list_to_wave("audioOut.wav",synthSpeechRet[0])

vtl.CloseSpeaker()

