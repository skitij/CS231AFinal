import os, sys
import time
from skimage import io
from skimage.color import rgb2gray
import numpy as np, numpy.matlib, numpy.ma
from skimage.transform import rescale, resize
import cv2

from math_functions import MathFuncs
from seq_solve import seq_solve
from fileread import fileread

def main():
    np.random.seed(123123)

    seqset = ['cheetah', 'football', 'highjump', 'horjump', 'javelin', 'paper', \
        'singer', 'skating', 'ski', 'soccer', 'sun', 'sylvester', 'vault', 'waves', \
        'marple2', 'tennis', 'monkey5', 'marple17', 'birdhouse', 'head', 'shoe']

    for cur_seq in seqset:
        print("processing sequence %s" % cur_seq)
        data_path = os.path.join('./data/', cur_seq)

        seq = fileread(data_path)
        
        I = None
        img0 = []
        for t in range(0, seq.numframe):
            img = seq.get_image()
            img0.append(img)

            gimg = rgb2gray(img)
            gimg = rescale(gimg, 0.20)
            if t == 0:
                I = np.zeros((gimg.shape[0]* gimg.shape[1], seq.numframe))
            
            I[:, t] = gimg[:, :].reshape(1,-1)
        
        for iter in range(0, 1):
            print('iter %d' % (iter + 1))

            sf = range(1,seq.numframe-1,1)
            p = np.random.permutation(sf)
            p = np.concatenate(([0],p))
            print('randomized order: ', p)

            Jumble = I[:, p]

            start_time = time.time()
            seqconstraint = np.hstack([ np.zeros( (Jumble.shape[1] - 1, 1) ), \
                np.array(list(range(1, Jumble.shape[1] ))).T.reshape(-1,1) ])
            _, outputorder, _ = seq_solve(Jumble, seqconstraint)

            end_time = time.time()

            origorder = np.array(range(0, Jumble.shape[1]))
            print('origorder : ', origorder)
            print('p[outputorder] : ', p[outputorder])
            dist, err = MathFuncs.kendall_distance(origorder, p[outputorder])
            print('sequence: %s kendall_dist: %f runtime: %f secs' % \
                (cur_seq, err, (end_time - start_time)) )

if __name__ == '__main__':
    main()

