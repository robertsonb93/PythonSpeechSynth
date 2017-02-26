import ctypes
import wave #pip install wavefile

MyDllObject = ctypes.cdll.LoadLibrary("../../VocalTractLabApi64")
#it's important to assing the function to an object
#MyFunctionObject = MyDllObject.MyFunctionName
Initialize = MyDllObject.vtlInitialize #int
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
#MyFunctionObject.restype = ctypes.c_wchar_p
Initialize.restype = ctypes.c_int
Close.restype = ctypes.c_void_p
GetVersion.restype = ctypes.c_void_p
GetConstants.restype = ctypes.c_void_p
GetTractParamInfo.restype = ctypes.c_void_p
GetGlottisParamInfo.restype = ctypes.c_void_p
GetTractParams.restype = ctypes.c_int
GetTransferFunction.restype = ctypes.c_void_p
SynthBlock.restype = ctypes.c_void_p
TubeSynthesisReset.restype =  ctypes.c_void_p
TubeSynthesisAdd.restype = ctypes.c_void_p
ApiTest1.restype = ctypes.c_void_p
ApiTest2.restype = ctypes.c_void_p
GesToWav.restype = ctypes.c_int


#define the types that your C# function will use as arguments
#MyFunctionObject.argtypes = [ctypes.c_wchar_p]
Initialize.argtype = [ctypes.c_char_p]
Close.argtype = [];
GetVersion.argtype = [ctypes.c_wchar_p]
GetConstants.argtype = [ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
GetTractParamInfo.argtype = [ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
GetGlottisParamInfo.argtype = [ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_double)]
GetTractParams.argtype = [ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_double)]
GetTransferFunction.argtype = [ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
SynthBlock.argtype = [ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_wchar_p,ctypes.c_int,ctypes.c_double,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int)]
TubeSynthesisReset.argtype = [ctypes.c_int, ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_wchar_p,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.POINTER(ctypes.c_double)]
ApiTest1.argtype = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int)]
ApiTest2.argtype = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int)]
GesToWav.argtype = [ctypes.c_wchar_p, ctypes.c_wchar_p,ctypes.c_wchar_p,ctypes.c_wchar_p]

def wave_to_double(filename):
    w = wave.open(filename,'r')
    w = [float(n) for n in w.readframes(w.getnframes())]
    return w

#That's it now you can test it
str ="..\\..\\VTL2.1\\test1.speaker"
s = str.encode('utf-8')
speaker = ctypes.c_char_p(s)

audio = wave_to_double("..\\..\\VTL2.1\\Example5-Hallo-synth.wav");
audioPtr = ctypes.POINTER(ctypes.c_double)()
audioPtr.contents = ctypes.c_double(audio[0]);


ApiTest1(speaker, audioPtr, len(audio))


print(Initialize(speaker)) # 0 means success, >0 means error

#I think the way to work with this API for ML purposes is going to be by loading a speaker, then modifiying it using TubesynthesisAdD, if 
#i understand it correctly(though I probably dont)