import sys
import vtlPythonAPI as vtl
import random
import time
random.seed(time.time())

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

def generateVocalTract(shapes,numVocalTractParams):#THis is for human use.
    params = list()

    for s in shapes:
        shapeParams = vtl.getVocalTractParamsfromShape(numVocalTractParams,s,False)
        for i in range (len(shapeParams[0])):
            params.append(shapeParams[0][i]) #This could be params = params + shapeParams[0]
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
numFrames = 4
frameRate = 1


bs = ['a','e','i','o','u','y','2'] #this is not comprehensive of basic vowel tracts
tn = ['tt-alveolar-nas(a)','tt-alveolar-nas(i)','tt-alveolar-nas(u)'] #closed tongue nasal passage 

shapes = [tn[1],bs[1],bs[4],tn[0]] #we need as many shapes as we have frames. a shape per frame.
print("Shapes are: ",shapes)

glottisParams = breathyGlottis(numFrames)

out = list()
VTP = generateVocalTract(shapes,numVocalTractParams)

tubeLength = 0.0025

for i in range(10):
    synthSpeechRet = vtl.synthSpeech(VTP,glottisParams,40,numFrames,frameRate,sampleRate)

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

    samplesPerFrame = sampleRate * frameRate
    audioFrames = [synthSpeechRet[0][x:x+samplesPerFrame] for x in range(0, len(synthSpeechRet[0]),samplesPerFrame)]
    tubeAreaFrames = [tubeAreas[x:x+numberTubeSections] for x in range(0,len(tubeAreas),numberTubeSections)]
    articFrames = [synthSpeechRet[3][x:x+numberTubeSections] for x in range(0,len(tubeAreas),numberTubeSections)]
    print("Audio Frames: ",len(audioFrames), "\nTubeArea Frames: ",len(tubeAreaFrames), "\nartic Frames: ", len(articFrames))


    for i in range(len(audioFrames)):
          print(len(audioFrames[i]))
          artics = ''.join(i for i in articFrames[i])
          artics = bytearray(map(ord,artics))
          Audio = vtl.addToSynthesis(tubelengths,tubeAreas,artics,incdist,velum,aspStrength,glottisParams,audioFrames[i])
          out = out + Audio
   
    
    


vtl.CloseSpeaker()

byteRate = 2
vtl.list_to_wave("audioOut.wav",out,byteRate,sampleRate)

import matplotlib.pyplot as plt
plt.plot(out)
plt.show()


