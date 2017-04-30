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
    s = "NTIL"
    options = bytearray()
    options.extend(map(ord, s))
    articulates = list()
    for i in range(12):
        articulates.append(options[0])
    for i in range(24):
        articulates.append(options[1])
   
    articulates.append(options[2])
    articulates.append(options[3])
    articulates.append(options[3])
    #for i in range(40):
       # articulates.append(random.choice(options))
      
    return articulates

#See the speaker file for what the parameters are

def generateGlottis(Gparams,numFrames):
    glottis = list()

    for i in range(numFrames):
        for j in range(len(Gparams[0])-1):
            glottis.append(random.uniform(Gparams[2][j],Gparams[1][j]))
    return glottis

def breathyGlottis(numFrames):#this is pulled from Triangular glottis model in the speaker file#THis is for human use.
    glottis = list()
    for i in range(numFrames):
        glottis.append(120)
        glottis.append(1000.000)
        glottis.append(0.000347)
        glottis.append(0.000298)
        glottis.append(0.000000)
        glottis.append(-35)

    return glottis

def SBGlottis(numFrames):#this is pulled from Triangular glottis model in the speaker file#THis is for human use.
    glottis = list()
    for i in range(numFrames):
        glottis.append(120)
        glottis.append(1000.000)
        glottis.append(0.000245)
        glottis.append(0.000197)
        glottis.append(0.000000)
        glottis.append(-20)

    return glottis

def fullyOpenGlottis(numFrames):#THis is for human use.
    glottis = list()
    for i in range(numFrames):
        glottis.append(120)
        glottis.append(1000.000)
        glottis.append(0.002002)
        glottis.append(0.002002)
        glottis.append(0.000000)
        glottis.append(0)
    return glottis

def generateVocalTract(shapes,numVocalTractParams):#THis is for human use.
    params = list()

    for s in shapes:
        shapeParams = vtl.getVocalTractParamsfromShape(numVocalTractParams,s,False)
        for i in range (len(shapeParams[0])):
            params.append(shapeParams[0][i])
    return params


spkr ="test1.speaker"
audio = vtl.wav_to_list("Example1-Sprachsynthese-orig.wav")

vtl.initSpeaker(spkr,True)
print(vtl.GetAPIVersion())

speakerConsts = vtl.getSpeakerConstants()
sampleRate = speakerConsts[0]
numberTubeSections = speakerConsts[1]
numVocalTractParams = speakerConsts[2]
numGlottisParams = speakerConsts[3]


vocalTractParams = vtl.getTractParams(speakerConsts[2])
GlottisParams = vtl.getGlottisParams(speakerConsts[3])

#We have a list of shapenames such as 'a','e','i','o','u'.. etc, to find them, consult the speaker file.
shapeName = 'aI-end' 
shapeParams = vtl.getVocalTractParamsfromShape(numVocalTractParams,shapeName,False)
if(shapeParams[1] == 0):#did it find a defined name shapename
    print(shapeParams[0])


VTPLite = [shapeParams[0][i] for i in range(5)]
magPhase = vtl.getTransferFunctions(VTPLite, 500)#I think this is what this expects, but Im not exactly sure.

vtl.resetSynthesis()

numFrames = 4
frameRate = 1


bs = ['a','e','i','o','u','y','2'] #this is not comprehensive of basic vowel tracts
tn = ['tt-alveolar-nas(a)','tt-alveolar-nas(i)','tt-alveolar-nas(u)'] #closed tongue nasal passage 

shapes = [tn[1],bs[4]]

glottisParams = fullyOpenGlottis(numFrames)

out = list()
VTP = generateVocalTract(shapes,numVocalTractParams)

import matplotlib.pyplot as plt


tubeLength = 0.0025
for i in range(3):

    
    synthSpeechRet = vtl.synthSpeech(VTP,glottisParams,40,numFrames,frameRate,sampleRate)

    glottisParams = breathyGlottis(numFrames)


    artics = ''.join(i for i in synthSpeechRet[3])
    artics = bytearray(map(ord,artics))

    tubeAreas = list()
    for i in range((int(len(synthSpeechRet[2])))):
        tubeAreas.append( synthSpeechRet[2][i]/100) # because we were returned with cm2, but need to give it in m2

    #tubeAreas[len(tubeAreas)-1] = 0
    velum = tubeAreas[16]
    tubelengths = list()
    for i in range(len(tubeAreas)):
        tubelengths.append(tubeLength)#my reading suggests that the pharynx on a human is between 12-14cm(0.12m) 0.003 * 40 sections = 0.12m
    
    
    print(sum(tubelengths))
    aspStrength = glottisParams[5]
    incdist = 0.1532

    Audio  = vtl.addToSynthesis(tubelengths,tubeAreas,artics,incdist,velum,aspStrength,glottisParams,synthSpeechRet[0])
  
    out = out + Audio



byteRate = 2
vtl.list_to_wave("audioOut.wav",out,byteRate,sampleRate)

plt.plot(out)
plt.show()

vtl.CloseSpeaker()

