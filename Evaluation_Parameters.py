"""This gets read by Evaluate.py"""


import os

## EVALUATION PARAMS
modelname   = "1605670402.8486269"  # "1603224725.508597"
iteration   = "99000"
noisevar    = "0_010"

datatype = "photometric"  # "photometric" vs "co-quadratic" vs "training"
#datatype = "co-quadratic"
#datatype = "training"

if datatype == "photometric":
    directoryin = str(os.getcwd()+"/Data/Synth")
    lights      = ["0", "1", "2", "3"]  # ["5", "6", "2", "4", "1", "3", "7", "0"]
elif datatype == "co-quadratic":
    directoryin = str(os.getcwd()+"/Data/MultiPixel/MultiPixel")  ##NOT MULTI LIGHTS. MultiPixel.nb
    lights      = ["0", "1", "2", "3"]  # really, "pixels"
elif datatype == "training":
    directoryin = ""
    lights      = ["0", "1", "2", "3"]

print("lights: ",lights,"\n\n")

#directoryin = str(os.getcwd()+"/Data/Perfect/from_036051")             ##GenerateSynthData.nb
#directoryin = str(os.getcwd()+"/Data/Noise_"+noisevar+"/from_036051")
    ##GenerateSynthData.nb
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
    
print("\n\ndataset: ", directoryin)

nbhd_gr     = 50                   #40 is good
nbhd_sz     = .5                    #2 is good

#if directoryin == str(os.getcwd()+"/Data/Noise_"+noisevar+"/from_036051"):
#    thisf = 0  #single fixed surface.    #<4 required.
#else:
#    thisf = 0

noiseon     = False



## VISUALIZATION PARAMS
plotson     = True  # cumulative & pairwise plots
plotson_col = False  # colored plots
plotson_more = False
finalpaper  = True  # more plots
allplotsoff = False  # overrides the above flags
verbose     = True  # printing statuses
moreverbose = False
wsz = 1
