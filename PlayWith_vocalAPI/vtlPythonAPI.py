import ctypes

import wave #pip install wavefile
c_int = ctypes.c_int32
w_char = ctypes.c_wchar_p
ptr = ctypes.POINTER
double = ctypes.c_double
ref = ctypes.byref
c_char = ctypes.c_char
c_charp = ctypes.c_char_p

MyDllObject = ctypes.cdll.LoadLibrary("../../VocalTractLabApi64")
Initialize = MyDllObject.vtlInitialize 
Close = MyDllObject.vtlClose
GetVersion = MyDllObject.vtlGetVersion
GetConstants = MyDllObject.vtlGetConstants
GetTractParamInfo = MyDllObject.vtlGetTractParamInfo
GetGlottisParamInfo = MyDllObject.vtlGetGlottisParamInfo
GetTractParams = MyDllObject.vtlGetTractParams #int
GetTransferFunction = MyDllObject.vtlGetTransferFunction
SynthBlock = MyDllObject.vtlSynthBlock
TubeSynthesisReset = MyDllObject.vtlTubeSynthesisReset
TubeSynthesisAdd = MyDllObject.vtlTubeSynthesisAdd
ApiTest1 = MyDllObject.vtlApiTest1
ApiTest2 = MyDllObject.vtlApiTest2
GesToWav = MyDllObject.vtlGesToWav #Use this function to turn a gestural Score into an audio file.

#define the types that your C# function return
Initialize.restype = c_int
Close.restype = None

GetVersion.restype = None
GetConstants.restype = None
GetTractParamInfo.restype = None
GetGlottisParamInfo.restype = None
GetTractParams.restype = c_int
GetTransferFunction.restype = None

SynthBlock.restype = c_int
TubeSynthesisReset.restype = None
TubeSynthesisAdd.restype = None

ApiTest1.restype = None
ApiTest2.restype = None
GesToWav.restype = c_int

#define the types that your C# function will use as arguments
Initialize.argtype = [ctypes.c_char_p]
Close.argtype = []

GetVersion.argtype = [c_char]
GetConstants.argtype = [ptr(c_int),ptr(c_int),ptr(c_int),ptr(c_int)]
GetTractParamInfo.argtype = [c_charp,ptr(double),ptr(double),ptr(double)]
GetGlottisParamInfo.argtype = [c_charp,ptr(double)]
GetTractParams.argtype = [c_charp,ptr(double)]
GetTransferFunction.argtype = [ptr(double),c_int,ptr(double),ptr(double)]

SynthBlock.argtype = [ptr(double),ptr(double),ptr(double),c_char,c_int,double,ptr(double),ptr(c_int)]
TubeSynthesisReset.argtype = [c_int, ptr(double),ptr(double),ptr(double),c_char,double,double,double,ptr(double)]
ApiTest1.argtype = [c_char, ptr(double),ptr(c_int)]
ApiTest2.argtype = [c_char, ptr(double),ptr(c_int)]
GesToWav.argtype = [c_char,c_char,c_char,c_char]

def wav_to_list(filename):
    w = wave.open(filename,'r')
    aud = [float(n) for n in w.readframes(w.getnframes())]
    w.close()
    return aud

def list_to_wave(filename,audio):
    w = wave.open(filename,'w')
    #nchannels, sampwidth, framerate, nframes, comptype, compname
    w.setparams((1, 2, 44100, len(audio), 'NONE', 'noncompressed'))
    audStr = (bytes(audio))
    w.writeframes(audStr)
    w.close()

def initSpeaker(SpeakerFile,debugMsg):
    
    s = SpeakerFile.encode('utf-8')
    if(debugMsg == True):
        if (Initialize(s) == 0 ):
            print("Successfully initialized speaker: ",s)
        else:
            print("Failed Initialize on speaker: ",s)
    else:
        Initalize(s)
    #return c_charp(s)

def CloseSpeaker():
    Close()
#audio = wave_to_double("..\\..\\VTL2.1\\Example5-Hallo-synth.wav");

def GetAPIVersion():
    print("Get Version")
    version = ctypes.create_string_buffer(64)
    GetVersion(ctypes.byref(version))
    return version.value

#// ****************************************************************************
#// Returns a couple of constants:
#// o The audio sampling rate of the synthesized signal.
#// o The number of supraglottal tube sections.
#// o The number of vocal tract model parameters.
#// o The number of glottis model parameters.
#// Will be Returned as a list of the parameter values in the above order
#// ****************************************************************************
def getSpeakerConstants():
    SamplingRate = c_int(1)
    TubeSections =  c_int(1)
    VocalTract = c_int(1)
    GlottisParams =  c_int(1)
    GetConstants(ref(SamplingRate),ref(TubeSections),ref(VocalTract),ref(GlottisParams))
    return (SamplingRate.value,TubeSections.value,VocalTract.value,GlottisParams.value)



#// ****************************************************************************
#// Returns for each vocal tract parameter the minimum value, the maximum value,
#// and the neutral value. Each vector passed to this function must have at 
#// least as many elements as the number of vocal tract model parameters.
#// The "names" string receives the abbreviated names of the parameters separated
#// by spaces. This string should have at least 10*numParams elements.
# Requires the number of Vocal Tract Parameters from getSpeakerConstants()
# will return a a set of lists, consisting of the TractNames, the minimum, max, and neutral parameter values
#// ****************************************************************************
def getTractParams(numVocalTractParams):
    aSize = numVocalTractParams*10; 
    tractNames = (c_char *aSize)(0)
    paramMin = (double * aSize)(0)
    paramMax = (double * aSize)(0)
    paramNeutral = (double * aSize)(0)

    GetTractParamInfo(tractNames,paramMin,paramMax,paramNeutral);#If you dont have the size large enough, python will just crash. no erros just die.
    tNames = [tractNames[i] for i in range(tractNames._length_)]
    pMin = [paramMin[i] for i in range(paramMin._length_)]
    pMax = [paramMax[i] for i in range(paramMax._length_)]
    pNeut = [paramNeutral[i] for i in range(paramNeutral._length_)]

    return (tNames,pMin,pMax,pNeut)


#// ****************************************************************************
#// Returns for each glottis model parameter the minimum value, the maximum value,
#// and the neutral value. Each vector passed to this function must have at 
#// least as many elements as the number of glottis model parameters.
#// The "names" string receives the abbreviated names of the parameters separated
#// by spaces. This string should have at least 10*numParams elements.
# Requires the number of GlottisParams from getSpeakerConstants()
#Will return a set of lists, consisting of the GlottisNames, the min,max,and neutral parameter values
#// ****************************************************************************
def getGlottisParams(numGlottisParams):
    aSize = numGlottisParams* 10;
    glottisNames = (c_char *aSize)(0)
    paramMin = (double * aSize)(0)
    paramMax = (double * aSize)(0)
    paramNeutral = (double * aSize)(0)

    GetGlottisParamInfo(glottisNames,paramMin,paramMax,paramNeutral)   
    gNames = [glottisNames[i] for i in range(glottisNames._length_)]
    pMin = [paramMin[i] for i in range(paramMin._length_)]
    pMax = [paramMax[i] for i in range(paramMax._length_)]
    pNeut = [paramNeutral[i] for i in range(paramNeutral._length_)]
    return (gNames,pMin,pMax,pNeut)


#// ****************************************************************************
#// Returns the vocal tract parameters for the given shape as defined in the
#// speaker file.
#// The vector passed to this function must have at least as many elements as 
#// the number of vocal tract model parameters.
#// Returns 0 in the case of success, or 1 if the shape is not defined.
#Requires the number of Vocal Tract parameters from GetConstants(), a shapeName from getTractNames(), and
# a boolean for printing messages on successfully grabbing the shapeName
#Returns a list of the vocal tract parameters for a given shape
#// ****************************************************************************

#Notice VocalTractParameters and given shape, we need a shape name! lets try using names from the GetTractParamInfo
def getVocalTractParamsfromShape(numVocalTractParams,shapeName,debugMsg):
    size = numVocalTractParams;
    sn = c_charp(shapeName)
    vocalParams = (double * size)();
    succes = GetTractParams(shapeName,vocalParams)

    if(debugMsg ==True):
            if(succes > 0):
                print("Shape for ",shapeName," is undefined")
            else: print("Shape for " ,shapeName," is defined")

    vPar = [vocalParams[i] for i in range(vocalParams._length_)]
    return (vPar,succes)

#// ****************************************************************************
#// Calculates the volume velocity transfer function of the vocal tract between 
#// the glottis and the lips for the given vector of vocal tract parameters and
#// returns the spectrum in terms of magnitude and phase.
#// Parameters (in/out):
#// o tractParams (in): Is a vector of vocal tract parameters with 
#//     numVocalTractParams elements.
#// o numSpectrumSamples (in): The number of samples (points) in the requested 
#//     spectrum. This number of samples includes the negative frequencies and
#//     also determines the frequency spacing of the returned magnitude and
#//     phase vectors. The frequency spacing is 
#//     deltaFreq = SAMPLING_RATE / numSpectrumSamples.
#//     For example, with the sampling rate of 22050 Hz and 
#//     numSpectrumSamples = 512, the returned magnitude and phase values are 
#//     at the frequencies 0.0, 43.07, 86.13, 129.2, ... Hz.
#// o magnitude (out): Vector of spectral magnitudes at equally spaced discrete 
#//     frequencies. This vector mus have at least numSpectrumSamples elements.
#// o phase_rad (out): Vector of the spectral phase in radians at equally 
#//     spaced discrete frequencies. This vector mus have at least 
#//     numSpectrumSamples elements.
#// ****************************************************************************
#Requires the VocalTractParameters from getVocalTractParams(), and the number of desired samples to be found
#Returns a tuple of lists, consisting of the magnitude from each samples, and the phase(in radians) of each smaple
def getTransferFunctions(vocalTractParams, numSamplesDesired):
    vtp = vocalTractParams
    tractParams = (double * len(vtp) )(*vtp)
    numSamples = c_int(numSamplesDesired);
    magnitude = (double * numSamplesDesired)();
    phaseRad = (double * numSamplesDesired)();

    GetTransferFunction(tractParams,numSamples,magnitude,phaseRad)
    mag = [magnitude[i] for i in range(magnitude._length_)]
    ph = [phaseRad[i] for i in range(phaseRad._length_)]
    return (mag,ph)


#// ****************************************************************************
#// Synthesize speech with a given sequence of vocal tract model states and 
#// glottis model states, and return the corresponding sequence of tube
#// area functions and the audio signal.
#// Parameters (in/out):
#// o tractParams (in): Is a concatenation of vocal tract parameter vectors
#//     with the total length of (numVocalTractParams*numFrames) elements.
#// o glottisParams (in): Is a concatenation of glottis parameter vectors
#//     with the total length of (numGlottisParams*numFrames) elements.
#// o tubeArea_cm2 (out): Is a concatenation of vocal tract area functions that
#//     result from the vocal tract computations. Reserve (numTubeSections*numFrames)
#//     elements for this vector!
#// o tubeArticulator (out): Is a concatenation of articulator vectors.
#//     Each single vector corresponds to one area function and contains the 
#//     articulators for each tube section (cf. vtlTubeSynthesisAdd(...)).
#//     Reserve (numTubeSections*numFrames) elements for this vector!
#// o numFrames (in): Number of successive states of the glottis and vocal tract
#//     that are going to be concatenated.
#// o frameRate_Hz (in): The number of frames (states) per second.
#// o audio (out): The resulting audio signal with sample values in the range 
#//     [-1, +1] and with the sampling rate audioSamplingRate. Reserve enough
#//     elements for the samples at the given frame rate, sampling rate, and
#//     number of frames!
#// o numAudioSamples (out): The number of audio samples written into the
#//     audio array.
#//
#// Returns 0 in case of success, otherwise an error code > 0.
#// ****************************************************************************

def synthSpeech(vocalTractParams, glottisParams,numTubeSections, numFrames,frameRate,audioSampleRate,):
    tractParams2 = (double * (len(vocalTractParams)))(*vocalTractParams)
    glottisParams2 = (double * (len(glottisParams)))(*glottisParams)

    tubeArea_cm2 = (double * (numTubeSections*numFrames))()
    tubeArticulator =(c_char * (numTubeSections*numFrames))()#could be up to 40 i think
    audio_out = (double * (int(numFrames/frameRate)*audioSampleRate))()
    numAudioSamples = (c_int)()
    nF = c_int(numFrames)
    fR = double(frameRate)

    SynthBlock(tractParams2,glottisParams2,tubeArea_cm2,tubeArticulator,nF,fR,audio_out,ref(numAudioSamples))
    print("NumAudioSamples: ",numAudioSamples.value)
    a_out = [int((audio_out[i]*125) + 125) for i in range(numAudioSamples.value)]
    return (a_out,numAudioSamples.value)

#// ****************************************************************************
#// Resets the synthesis from a sequence of tubes (see vtlTubeSynthesisAdd() below).
#// ****************************************************************************
def resetSynthesis():
    TubeSynthesisReset()

#// ****************************************************************************
#// Synthesizes a speech signal part of numNewSamples samples and returns 
#// the new signal samples in the array audio (the caller must allocate 
#// the memory for the array).
#// For the synthesis of this part, the vocal tract tube is linearly interpolated
#// between the current tube and glottis states and the given new tube and 
#// glottis states.
#// The new tube states is given in terms of the following parameters:
#// o tubeLength_m: Vector of tube sections lengths from the glottis (index 0)
#//     to the mouth (index numTubeSections; see vtlGetConstants()).
#// o tubeArea_m2: According vector of tube sections areas in m^2.
#// o tubeArticulator: Vector of characters (letters) that denote the articulator 
#//     that confines the vocal tract at the position of the tube. We dicriminate
#//     'T' for tongue
#//     'I' for lower incisors
#//     'L' for lower lip
#//     'N' for any other articulator
#// o incisorPos_m: Position of the incisors from the glottis.
#// o velumOpening_m2: Opening of the velo-pharyngeal port in m^2.
#// o aspirationStrength_dB: Aspiration strength at the glottis.
#// 
#// The new glottis model state is given by the vector:
#// o newGlottisParams
#// ****************************************************************************
def addToSynthesis(numNewSamples, tubeLengthsList, tubeAreasList, ArticulatorsList, incisorGlottisDist_meters,
                   velumArea_meters2, aspirationStrengthDecibles,numGlottisParams):
    numNS = c_int(numNewSamples)
    audio = (double * numNewSamples)()
    if(len(tubeLengthsList) != len(tubeAreasList)):
        print("MalFormed tube LengthList and Tube Area List in AddToSynthesis(), Check lengths")

    tubeLengths = (double * len(tubeLengthsList))(*tubeLengthsList)
    tubeAreas = (double * len(tubeAreasList))(*tubeAreasList)
    articulators = (c_char * len(ArticulatorsList))(*ArticulatorsList)
    incisorPos = double(incisorGlottisDist_meters)
    velumOpen = double(velumArea_meters2)
    aStrengthdB = double(aspirationStrengthDecibles)
    newGlottisState = (double * numGlottisParams)()
    TubeSynthesisAdd(numNS,audio,tubeLengths,tubeAreas,articulators,incisorPos,velumOpen,aStrengthdB,newGlottisState)
    a_out = [int((audio[i]*125) + 125) for i in range(audio._length_)]
    newGlottis = [newGlottisState[i] for i in range(newGlottisState._length_)]
    return (newGlottis,a_out)

#These MUST be called before initialize is called
def test1(speakerFile, audioList):
    speaker = speakerFile.encode('utf-8')
    numSamples = c_int(len(audioList))
    audio = (double * numSamples.value)(*audioList)
    ApiTest1(speaker, audio, ref(numSamples))


def test2(speakerFile, audioList):
    speaker = speakerFile.encode('utf-8')
    numSamples = c_int(len(audioList))
    audio = (double * numSamples.value)(*audioList)
    ApiTest2(speaker, audio, ref(numSamples))

#I think the way to work with this API for ML purposes is going to be by loading a speaker, then modifiying it using TubesynthesisAdd, if 
#i understand it correctly(though I probably dont). 
#Perhaps we start with vtlSynthBlock and create the "model" to be synthesized, theen use the tubesynthesis mentioned above