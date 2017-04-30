import re
#the Vocal Tract Shapes are defined in the lines <shape name="NAME"> with the quotations, but only while
#we havent passed the line </vocal_tract_model>
def GetShapesfromSpeaker(speaker):
    f = open(speaker,'r')
    ret = list()
    text = f.readline()
    while  "</vocal_tract_model>" not in text :    
        if "<shape name=" in text:
            word = re.findall('".*"',text)#grab the word between quotes.
            if(len(word) <1):
                print(text)
            ret.append(re.sub('["]','',word[0]))#remove the open and close quotes.
        text = f.readline()

    return ret

def GetGlottisfromSpeaker(speaker):
    f = open(speaker,'r')
    ret = list()
    text = f.readline()
    while "<glottis_model type=" not in text and "selected=\"1\"" not in text:
        text = f.readline()

    while "</glottis_model>" not in text:
        if "<shape name=" in text:
            word = re.findall('".*"',text)#grab the word between quotes.
            if(len(word) <1):
                print(text)
            ret.append(re.sub('["]','',word[0]))#remove the open and close quotes.
        text = f.readline()

    return ret