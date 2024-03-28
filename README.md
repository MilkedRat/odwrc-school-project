# ODwRC(Object Detection with Reverse Cameras)
This is an object detection program intended\* for use on the cameras you find on the back of those fancy vehicles

\*Not intended, just in practice

# ---Usage---
### train-and-save.py
Running this file will ask 2 things:
-Number of epochs
-Filename to save the build to

The number of epochs is easy, basically, the more epochs, the more accurate, however, the more epochs, the slower it is to build.

The filename can be whatever you like (I recommend keeping consistent filenames), everything is stored inside the "models" folder.

### load-and-compare.py
Running this file will ask 2 things:
-Filename of the build to use
-Filename of the image to use

The filename of the build assumes the "models/" path, pick an existing already-built model.

The filename of the image assumes the "traindata/" path, pick an existing .JPG image
