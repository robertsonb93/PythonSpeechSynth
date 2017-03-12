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
#it's important to assign the function to an object
#MyFunctionObject = MyDllObject.MyFunctionName

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
#MyFunctionObject.restype = w_char
Initialize.restype = c_int
Close.restype = None

GetVersion.restype = None
GetConstants.restype = None
GetTractParamInfo.restype = None
GetGlottisParamInfo.restype = None
GetTractParams.restype = c_int
GetTransferFunction.restype = None

SynthBlock.restype = c_int
TubeSynthesisReset.restype =  None
TubeSynthesisAdd.restype = None

ApiTest1.restype = None
ApiTest2.restype = None
GesToWav.restype = c_int


#define the types that your C# function will use as arguments
#MyFunctionObject.argtypes = [w_char]
Initialize.argtype = [ctypes.c_char_p]
Close.argtype = [];

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

def wave_to_double(filename):
    w = wave.open(filename,'r')
    w = [float(n) for n in w.readframes(w.getnframes())]
    return w







#That's it now you can test it
str ="..\\..\\VTL2.1\\test1.speaker"
s = str.encode('utf-8')
speaker = c_charp(s)

audio = wave_to_double("..\\..\\VTL2.1\\Example5-Hallo-synth.wav");

#Audio is a list, convert it to an array of doubles 
audioPtr = (double * len(audio))(*audio)
#ApiTest1(speaker, ctypes.byref(audioPtr), audioPtr._length_)#currently throwing an unhandled OS exception


print(Initialize(speaker)) # 0 means success, >0 means error


print("Get Version")
version = ctypes.create_string_buffer(64)
GetVersion(ctypes.byref(version))
print(version.value)

#// ****************************************************************************
#// Returns a couple of constants:
#// o The audio sampling rate of the synthesized signal.
#// o The number of supraglottal tube sections.
#// o The number of vocal tract model parameters.
#// o The number of glottis model parameters.
#// ****************************************************************************
audioSamplingRate_Int =c_int(1)
numTubeSections_Int =  c_int(1)
numVocalTractParams = c_int(1)
numGlottisParams =  c_int(1) 
GetConstants(ref(audioSamplingRate_Int), 
             ref(numTubeSections_Int),
          ref(numVocalTractParams),
            ref(numGlottisParams))


#// ****************************************************************************
#// Returns for each vocal tract parameter the minimum value, the maximum value,
#// and the neutral value. Each vector passed to this function must have at 
#// least as many elements as the number of vocal tract model parameters.
#// The "names" string receives the abbreviated names of the parameters separated
#// by spaces. This string should have at least 10*numParams elements.
#// ****************************************************************************

aSize = numGlottisParams.value *11; 
tractNames = (c_char *aSize)(0)
paramMin = (double * aSize)(0)
paramMax = (double * aSize)(0)
paramNeutral = (double * aSize)(0)

GetTractParamInfo(tractNames,paramMin,paramMax,paramNeutral);#If you dont have the size large enough, python will just crash. no erros just die.



#// ****************************************************************************
#// Returns for each glottis model parameter the minimum value, the maximum value,
#// and the neutral value. Each vector passed to this function must have at 
#// least as many elements as the number of glottis model parameters.
#// The "names" string receives the abbreviated names of the parameters separated
#// by spaces. This string should have at least 10*numParams elements.
#// ****************************************************************************

aSize = numGlottisParams.value * 10;
glottisNames = (c_char *aSize)(0)
paramMin = (double * aSize)(0)
paramMax = (double * aSize)(0)
paramNeutral = (double * aSize)(0)

GetGlottisParamInfo(glottisNames,paramMin,paramMax,paramNeutral)



#// ****************************************************************************
#// Returns the vocal tract parameters for the given shape as defined in the
#// speaker file.
#// The vector passed to this function must have at least as many elements as 
#// the number of vocal tract model parameters.
#// Returns 0 in the case of success, or 1 if the shape is not defined.
#// ****************************************************************************

#Notice VocalTractParameters and given shape, we need a shape name! lets try using names from the GetTractParamInfo
size = numVocalTractParams.value;

paramList = list();
for i in range(0,tractNames._length_):
    shapeName = ctypes.c_char_p(tractNames[i]) 
    param = (double * size)()
    
    ret = GetTractParams(shapeName,param) ## I am suspicious this is currently broken...
    if(ret <1):#I wonder if this is the right way to do this.
        for p in range(0,size):
             paramList.append(param[p]);
    #print(ret)#This produces a mix of true and false, so i guess its working


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

#dbl*, int, dbl*, dbl*
#use the tractParams that is given by GetTractParams
tractParams = param;
numSamples = c_int(10);
magnitude = (double * numSamples.value)();
phaseRad = (double * numSamples.value)();

GetTransferFunction(tractParams,numSamples,magnitude,phaseRad)


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

#What is numFrames? I think a frame is a sample, so how many samples do we want.
#what is frameRate? how many samples per second does the audio play??
numFrames = len(audio);
frameRate = audioSamplingRate_Int.value;

tractParams2 = (double * (numFrames*numVocalTractParams.value))()
i=0;
for tp in tractParamList:
    tractParams2[i] = tp;

#glottisParams2 = it says a concatentaion of glottis param vectors,, but we dont have these??? Do we make them?





print(Close());

#I think the way to work with this API for ML purposes is going to be by loading a speaker, then modifiying it using TubesynthesisAdd, if 
#i understand it correctly(though I probably dont). 
#Perhaps we start with vtlSynthBlock and create the "model" to be synthesized, theen use the tubesynthesis mentioned above