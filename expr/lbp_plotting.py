import sys
import cv2
import numpy as np
from expr.read_dataset import read_images
from facerec_py.facerec.lbp import ExtendedLBP, OriginalLBP
from facerec_py.facerec.preprocessing import LBPPreprocessing

__author__ = 'Danyang'


class LbpIntermidiate(object):
    def __init__(self, lbp=OriginalLBP()):
        self.X, self.y = self.read()
        self.lbp = lbp

    def read(self):
        if len(sys.argv) < 2:
            print "USAGE: experiment_setup.py </path/to/images>"
            sys.exit()
        # Now read in the image data. This must be a valid path!
        X, y = read_images(sys.argv[1])

        X = np.asarray(X)
        y = np.asarray(y)
        return X, y

    def experiment(self):
        hstack = lambda x, y: np.hstack((x, y))
        org_imgs = reduce(hstack, self.X)
        lbp_imgs = reduce(hstack, map(lambda x: self.lbp(x).astype(np.uint8), self.X))
        cv2.imshow("original", org_imgs)
        cv2.imshow("lbp", lbp_imgs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    LbpIntermidiate().experiment()
