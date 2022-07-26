# DCPVO
Deep Central-Peripheral Vision Odometry\n
High resolution input image preprocessed as narrow angle of view (Central vision) and low-res wide angle (Peripheral vision) image.
The image then feeded into deep depth and flow networks was forked from https://github.com/Huangying-Zhan/DF-VO (for fast implementation) and then pass into a deep dense network for (R,T) regression.
