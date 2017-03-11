import ctypes

import wave #pip install wavefile
c_int = ctypes.c_int32
w_char = ctypes.c_wchar_p
ptr = ctypes.POINTER
double = ctypes.c_double
ref = ctypes.byref

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
SynthBlock.restype = None
TubeSynthesisReset.restype =  None
TubeSynthesisAdd.restype = None
ApiTest1.restype = None
ApiTest2.restype = None
GesToWav.restype = c_int


#define the types that your C# function will use as arguments
#MyFunctionObject.argtypes = [w_char]
Initialize.argtype = [ctypes.c_char_p]
Close.argtype = [];
GetVersion.argtype = [w_char]
GetConstants.argtype = [ptr(c_int),ptr(c_int),ptr(c_int),ptr(c_int)]
GetTractParamInfo.argtype = [w_char,ptr(double),ptr(double),ptr(double)]
GetGlottisParamInfo.argtype = [w_char,ptr(double)]
GetTractParams.argtype = [w_char,ptr(double)]
GetTransferFunction.argtype = [ptr(double),c_int,ptr(double),ptr(double)]
SynthBlock.argtype = [ptr(double),ptr(double),ptr(double),w_char,c_int,double,ptr(double),ptr(c_int)]
TubeSynthesisReset.argtype = [c_int, ptr(double),ptr(double),ptr(double),w_char,double,double,double,ptr(double)]
ApiTest1.argtype = [w_char, ptr(double),ptr(c_int)]
ApiTest2.argtype = [w_char, ptr(double),ptr(c_int)]
GesToWav.argtype = [w_char, w_char,w_char,w_char]

def wave_to_double(filename):
    w = wave.open(filename,'r')
    w = [float(n) for n in w.readframes(w.getnframes())]
    return w







#That's it now you can test it
str ="..\\..\\VTL2.1\\test1.speaker"
s = str.encode('utf-8')
speaker = ctypes.c_char_p(s)

audio = wave_to_double("..\\..\\VTL2.1\\Example5-Hallo-synth.wav");

#Audio is a list, convert it to an array of doubles 
audioPtr = (double * len(audio))(*audio)
#ApiTest1(speaker, ctypes.byref(audioPtr), audioPtr._length_)#currently throwing an unhandled OS exception


print(Initialize(speaker)) # 0 means success, >0 means error
print(Close());

print("Get Version")

version = ctypes.create_string_buffer(64)
GetVersion(ctypes.byref(version)) #After adding this, the GetConstants below sometimes doesnt segfault
print(version.value)



#// ****************************************************************************
#// Returns a couple of constants:
#// o The audio sampling rate of the synthesized signal.
#// o The number of supraglottal tube sections.
#// o The number of vocal tract model parameters.
#// o The number of glottis model parameters.
#// ****************************************************************************
#audioSamplingRate_Int =c_int(1)
#numTubeSections_Int =  c_int(1)
#numVocalTractParams = c_int(1)
#numGlottisParams =  ctypes.c_long(1) #I have tried as being a single integer, with it going as a long, int, int32, 
##numGlottisParams = (c_int * 64)(99)#After trying the single values, tried handing it some arrays as both double and integer sizes 1,16,32,64

#GetConstants(ctypes.byref(audioSamplingRate_Int), 
#             ctypes.byref(numTubeSections_Int),
#           ctypes.byref(numVocalTractParams),
#             ctypes.byref(numGlottisParams),#I have tried re-ordering the params to ensure there isnt a type difference, but to no avail.
#             ) #Well it was working, till I decided to hit reset.... then it changed its mind.. Intermittent SegFaults

##print("Get Constants Post")#IF I keep mashing the reset/rerun button, sometimes i get to see these values except for numGlottisParams
##print(audioSamplingRate_Int.value)
##print(numTubeSections_Int.value)
##print(numVocalTractParams.value)
##print(numGlottisParams.value)#when these do pass through, they all have the correct values except for numGlottisParams(returns 0, not 6)


#// ****************************************************************************
#// Returns for each vocal tract parameter the minimum value, the maximum value,
#// and the neutral value. Each vector passed to this function must have at 
#// least as many elements as the number of vocal tract model parameters.
#// The "names" string receives the abbreviated names of the parameters separated
#// by spaces. This string should have at least 10*numParams elements.
#// ****************************************************************************

#using the test Api, I know we havbe 24 tractParams

aSize = 64
names = (c_int *aSize)(0)
paramMin = (double * aSize)(0)
paramMax = (double * aSize)(0)
paramNeutral = (double * aSize)(0)

#int,dbl,dbl,dbl
GetTractParamInfo(names,paramMin,paramMax,paramNeutral);#I read somewhere that these python are the same as ctypes, hence dont needed byref
                                                            #doesnt mean they dont both segfault..




#I think the way to work with this API for ML purposes is going to be by loading a speaker, then modifiying it using TubesynthesisAdd, if 
#i understand it correctly(though I probably dont). 
#Perhaps we start with vtlSynthBlock and create the "model" to be synthesized, theen use the tubesynthesis mentioned above