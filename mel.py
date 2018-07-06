from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os

n=0
x=0

# for file in os.listdir(r"C:\Users\VARUN\PycharmProjects\NLP\telugu"):
for file in os.listdir(r"D:\nlp lang\liddata\train\english"):
    file1 = open("HinTam.txt", "a")
    flist=[]
    n=n+1

    print(file + " written")
    x=0
    (rate, sig) = wav.read(file)
    mfcc_feat = mfcc(sig, rate)
    fbank_feat = logfbank(sig, rate)
    flist.append(fbank_feat[1:2, :])
    for i in flist:
        for j in i:
            for k in j:
                file1.write(str(k)+" ")
                x=x+1
    if(x==39):
        print(file+" has "+str(x)+" features")
    file1.write("\n")

    print("values written")
    # file1.write("\\n\\n")
    file1.close()

#f=open("HindiMfcc.txt","a")
#f.write("number of rows:"+str(n)+"\n"+"number of columns:"+str(x))
#f.close()



