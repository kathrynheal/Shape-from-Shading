"""This gets read by Evaluate.py"""


import os

modelname   = "1580752072.4233093"   #"1572314742.3791122" #   #"1569212026.0217707" #"1569258849.5705116"
iteration   = "9900"                 #"49500"              #   #"49500"#"3300"#
noisevar    = "0_010"


directoryin = str(os.getcwd()+"/Data/Synth")
#directoryin = str(os.getcwd()+"/Data/Perfect/from_036051")             ##GenerateSynthData.nb
#directoryin = str(os.getcwd()+"/Data/Noise_"+noisevar+"/from_036051")
    ##GenerateSynthData.nb
#directoryin = str(os.getcwd()+"/Data/MultiPixel/MultiPixel")
    ##NOT MULTI LIGHTS.
    ##MultiPixel.nb
#directoryin = str(os.getcwd()+"/Data/YUP/YUP")
    ##RealDataProcessing.nb
#directoryin = str(os.getcwd()+"/Data/Real/Real")
    ##RealDataProcessing.nb
#directoryin = str(os.getcwd()+"/Data/testme/testme")
    ##RealDataProcessing.nb
#directoryin = str(os.getcwd()+"/Data/please/please")
    ##RealDataProcessing.nb
#directoryin = str(os.getcwd()+"/Data/realreal/realreal")
    ##RealDataProcessing.nb
print("\n\ndataset: ",directoryin)


nbhd_gr     = 10                   #40 is good
nbhd_sz     = 1                    #2 is good
if directoryin == str(os.getcwd()+"/Data/Noise_"+noisevar+"/from_036051"):
    thisf = 0  #single fixed surface.    #<4 required.
else:
    thisf = 0
noiseon     = False
printmore   = False
printmore2  = False
printmore3  = False
plotson     = False #cumulative & pairwise plots
plotson0    = False #variety plots
plotsonc    = False #coloring plots
finalpaper  = False
allplotsoff = False

lights      = ["1","2","3","4"]
print("lights: ",lights,"\n\n")

wsz = 1
