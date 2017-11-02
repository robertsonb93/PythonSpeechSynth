This a Project started by Brandon Robertson for the Gruber Lab in the spring of 2017.
The projects goal is to try and create a human sounding speech generator, the method is by looking at the problem
as a motor control problem, the motor control in this case being the human vocal tract.

The project has several elements to it
The first element is the VTL (Vocal Tract Lab) model, This model is a black box in this project and was sourced from the internet 
and is the work of Peter Birkholz of Technische Universitat Dresden.
Being a black box we are provided only with an API, as is titled VocalTractLabApi64.dll,
naturally there are exposed functions for the API in VocalTractLabAPI64.h .

The machine learning goals are as follows:
-Create ML model capable of learning the relation between an input audio and then output the params to synthesize said audio.
(Which is done using randomized parameters and sound generation from the VTL Model)

-Use Trained ML Model to produce parameters for a real life speaker, have evaluate the results of synthesized params and if they are 
accurate. 

-Start collecting groups of parameters on a speaker to learn what common characteristics there exists between their sounds

-Use characteristics from parameters to synthesize characterised speach

File Descriptions

// vtlPythonAPI.py //
The part where we begin is from the file vtlPythonAPI.py, whos primary purpose is to act as a wrapper for the VTL API. 
The code contains conversion data for every exposed function from the API, and provides a simpler interface so that the API
can be utilized from python alone without having to go back to the API header file. Understand that data which is transfered
across DLL boundaries must be primitive (no data structures like lists), maintaining accurate lengths and pointers is imperative.
Furthermore modifying this code and not keeping accurate lengths/pointers is prone to soft failing and may keep running with
unpredictable results, or failing spontaneously at working pieces of code. If edits are made to this wrapper, ensure that 
changes are small, and ensure significant testing on changes to help keep your sanity. 

// HumanAttempt.py //
This file was primarily used for testing and evaluating the wrapper file. It goes throught eh procedure of starting the model,
providing the model with parameters and synthesizing the said parameters. It is somewhat out of date and some of the knowledge in it
may not accurately represent how the model works, but it does provide a general idea on the flow of the model. Furthmore this script
will allow for a quick to use framework for testing out parameters on the model  if desired. 
It should be noted that there apears to be 2 synthesis occuring, one from vtl.synthSpeech, and later one from vtl.addToSynthesis
My understanding for requiring the 2 is that the prior generates a fundamental frequency, the sound that would be sourced from the
glottis in a human. The later is given the fundamental frequency and runs it through a simulation of the throat after the glottis.

// TextScraper.py //
There are speaker files with the API, these appear to be predefined speakers for the model, (Values on how the throat is shapped etc.)
This file was meant to quickly grab some of the predefined values from the speaker file. Things such as glottis and phoneme shapes

// MLAttempt.py //
As of the time of writing this, this file has been the primary source of work, and works as the program entry point. 
It currently utilizes Tensorflow 1.3 to work on a set of model generated data, utilizing a RNN that feeds into a FFN.
The generated model data is created by TrainingDataGen.py and stores a set of parameters being the starting frame and ending frame
of the voicebox (Frame is the state of a throat, a sound is a transition between states) and is followed by the produced audio.
The goal is for the learner to be capable of learning how parameters create varying sounds, so that a real life speaker can 
be fed into it and the model then used to reproduce an accurate recreation of the sounds. 

// TrainingDataGen.py  //
This file is typically called from MLAttempt.py as a way of auto generating new data sets for training the tensorflow models on.
It produces parameters randomly (within ranges) and then proceeds to synthesis the sounds from those parameters. Both the parameters
and audio are then stored in CSV files which can be read by tensorflow.

// TraningDataRevamp.py //
I didnt quite know what my training data needed to look like the first time I generated it, This file has been rewritten a a few
times and is just used to make modifications to the training data sets. Is run on its own and not from MLAttempt.py


All the files were written in python 3.5 and tensorflow 1.3 and Visual Studio 2015 / 2017