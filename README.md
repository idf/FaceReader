# FaceReader
Face Recognition  

## Modules
* [Eigenface and Fisherface](http://www.cs.columbia.edu/~belhumeur/journal/fisherface-pami97.pdf)
* [Local Binary Pattern](http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html#local-binary-patterns-histograms)
* [Gabor Fisher](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.7675&rep=rep1&type=pdf)
* [Local Gabor Binary Pattern Histogram Sequence (LGBPHS)](http://www.jdl.ac.cn/user/sgshan/pub/ICCV2005-ZhangShan-LGBP.pdf)
* [Local Phase Quantization](http://www.ee.oulu.fi/research/imag/mvg/files/pdf/ICISP08.pdf)
## TODO
* [GaussianFace](http://arxiv.org/pdf/1404.3840.pdf) 

## Experiment Run
in expr/experiment_setup.py  
To enable ROC: `cv = KFoldCrossValidation(model, k=10, threshold_up=1)`  
To disable ROC: `cv = KFoldCrossValidation(model, k=10, threshold_up=0)`  

## Git Subtree
* util 

## Databases
### Yale A 
![](/img/yale_a.png)
