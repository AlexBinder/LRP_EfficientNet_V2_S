# LRP_EfficientNet_V2_S

This one uses the torchvision implementation, and not like the other repo here, which was based on the version from Luke Melas (thanks to him, anyway!). It should be extendible to other efficientnet versions. Not tested on a GPU, not taken one with me on a "holiday" :) . 

Something does not work, email me to my institutional email (Magdeburg, Leipzig). If I do not reply, its nothing against you, just keep resending until I do.  

Bugs ? Can be. The forward pass seems to have a minor discrepancy, likely causing the upper left corner artefact. If you know how to fix it, tell me. 

Software version this was tested on:
python3
Python 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch,torchvision
>>> torch.__version__
'2.6.0+cpu'
>>> torchvision.__version__
'0.21.0+cpu'
>>> exit()
