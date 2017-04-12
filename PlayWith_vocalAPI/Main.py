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

def breathyGlottis(numFrames):
    glottis = list()
    for i in range(numFrames):
        glottis.append(105.104613)
        glottis.append(1000.000)
        glottis.append(0.000445)
        glottis.append(0.000396)
        glottis.append(0.000025)
        glottis.append(0.885903)
        glottis.append(0.00)

    return glottis

def generateVocalTract(shapes,numVocalTractParams):
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

print("Adding synthesis")
#For AddToSynthesis we need to generate some tubeListLengths in meters
###for i in range(15):
#    tubes = generateTubeLengthsnAreas()
#    articulates = generateArticulates()
#    incisorGlottisDist = random.uniform(0,1)
#    velumArea = random.uniform(0.1,0.35)
#    pressure = int(random.uniform(0,2000))
#   # newGlottisnAudio = vtl.addToSynthesis(sampleRate,tubes[0],tubes[1],articulates,incisorGlottisDist
#  #                                        ,velumArea,pressure,numGlottisParams )
#    print(i) #And add to synthesis does a whole lot of zeros for the returns values


#Im not quite sure the relationship between addtoSynthesis and synthblock(inside synth speech)
#but trying to run add to synthesis after synthblock will crash python.

numFrames = 4
frameRate = 1

glottisParams = generateGlottis(GlottisParams,numFrames)
for i in range(int(len(glottisParams)/6)):
    glottisParams[(i*6)+1] = 1700.00;

blurr = ['a','e','i','o','u','y','2']
shapes = list()
for i in range(numFrames):
    shapes.append('a')
    shapes.append('i')
   # shapes.append('tb-velar-stop(a)')

glottisParams = breathyGlottis(numFrames)

VTP = generateVocalTract(shapes,numVocalTractParams)

synthSpeechRet = vtl.synthSpeech(VTP,glottisParams,40,numFrames,frameRate,sampleRate)


tubes = list()
artics = ''.join(i for i in synthSpeechRet[3])
artics = bytearray(map(ord,artics))

for i in synthSpeechRet[2]:
    tubes.append(random.uniform(0.0045,0.045))



   
incdist1 = sum(tubes)
incdist2 = sum(tubes,0.01)
velum1 = 0.0001
velum2 = 0.0001
newGltAudio  = vtl.addToSynthesis(synthSpeechRet[1],tubes,synthSpeechRet[2],artics,incdist1,velum1,glottisParams[6],numGlottisParams,synthSpeechRet[0])
newGltAudio2  = vtl.addToSynthesis(synthSpeechRet[1],tubes,synthSpeechRet[2],artics,incdist2,velum2,glottisParams[6],numGlottisParams,newGltAudio[1])


synthSpeechRet = vtl.synthSpeech(VTP,glottisParams,40,numFrames,frameRate,sampleRate)

out =  synthSpeechRet[0] + newGltAudio[1] + newGltAudio2[1]
#out = newGltAudio2[1]

byteRate = 2
vtl.list_to_wave("audioOut.wav",out,byteRate,sampleRate)


vtl.CloseSpeaker()

