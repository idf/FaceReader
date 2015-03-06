# FaceReader
Face Recognition  

## Modules
* [Eigenface and Fisherface](http://www.cs.columbia.edu/~belhumeur/journal/fisherface-pami97.pdf): https://github.com/bytefish/facerec
* [Local Binary Pattern](http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html#local-binary-patterns-histograms)
* [GaussianFace](http://arxiv.org/pdf/1404.3840.pdf) 

## Experiment Run
in expr/experiment_setup.py  
To enable ROC: `cv = KFoldCrossValidation(model, k=10, threshold_up=1)`  
To disable ROC: `cv = KFoldCrossValidation(model, k=10, threshold_up=0)`  

## Git Subtree
* util 
