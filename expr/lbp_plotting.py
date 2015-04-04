import sys
import cv2
import numpy as np
from expr.feature import GaborFilterCv2
from expr.read_dataset import read_images
from facerec_py.facerec.lbp import ExtendedLBP, OriginalLBP
from facerec_py.facerec.preprocessing import LBPPreprocessing
import matplotlib.pyplot as plt

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

    def draw(self):
        org_imgs = reduce(self.hstack, self.X)
        lbp_imgs = reduce(self.hstack, map(self.lbp_filter, self.X))
        cv2.imshow("original", org_imgs)
        cv2.imshow("lbp", lbp_imgs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def hstack(self, x, y):
        return np.hstack((x, y))

    def lbp_filter(self, x):
        return self.lbp(x).astype(np.uint8)


class GaborLbpIntermidiate(LbpIntermidiate):
    def __init__(self):
        super(GaborLbpIntermidiate, self).__init__()
        self.sz = (16, 16)
        self.py = 0
        self.px = 0

    def square(self, X, row, col):
        for x in X:
            x[row * self.py, col * self.px:(col + 1) * self.px] = 255
            x[(row + 1) * self.py, col * self.px:(col + 1) * self.px] = 255
            x[row * self.py:(row + 1) * self.py, col * self.px] = 255
            x[row * self.py:(row + 1) * self.py, (col + 1) * self.px] = 255
        return X

    def gabor_filter(self, x, idx=34):
        gabor = GaborFilterCv2(8, 5)
        return cv2.filter2D(x, cv2.CV_8UC3, gabor._kernels[idx])

    def draw(self, title, row=12, col=3, gabor_filter=lambda x: x, lbp_filter=lambda x: x):
        # cv2.imshow("gabor", reduce(self.hstack, map(gabor_filter, self.X)))

        imgs_procssed = map(lbp_filter, map(gabor_filter, self.X))

        hists = []
        for i in imgs_procssed:
            hists.append(self.histogram(i, row, col))


        cv2.imshow("original", reduce(self.hstack, self.square(self.X, row, col)))

        plt.subplot()
        plt.title(title)
        plt.ylabel("Number of Pixels")
        plt.xlabel("Gray Level")

        lines = []
        for idx, hist in enumerate(hists):
            line, = plt.plot(range(len(hist)), hist, label="image-%d" % (idx + 1))
            lines.append(line)
        plt.legend(handles=lines)

        plt.show()

    def run(self):
        self.draw("Original")
        self.draw("Lbp", lbp_filter=self.lbp_filter)
        self.draw("Gabor", gabor_filter=lambda x: self.gabor_filter(x, 5))
        self.draw("LGBPHS", lbp_filter=self.lbp_filter, gabor_filter=self.gabor_filter)

    def histogram(self, L, row, col):
        # calculate the grid geometry
        lbp_height, lbp_width = L.shape
        grid_rows, grid_cols = self.sz
        # grid size
        self.py = int(np.floor(lbp_height / grid_rows))
        self.px = int(np.floor(lbp_width / grid_cols))

        C = L[row * self.py:(row + 1) * self.py, col * self.px:(col + 1) * self.px]  # sub-regions
        H = np.histogram(C,
                         bins=16,
                         range=(0, 2 ** self.lbp.neighbors),
                         weights=None,
                         normed=False
        )[0]  # normalized
        # probably useful to apply a mapping?
        return np.asarray(H)


if __name__ == "__main__":
    print __file__
    # LbpIntermidiate().draw()
    GaborLbpIntermidiate().run()
