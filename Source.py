'''

Project Title: Real-Time Monitoring and Counting of People

Description: 
	In this program we use Histogram of Oriented Gradients(HOG) and Support Vector Machine(SVM) to detect the body of a human being. An image is a 2-dimensional data. If we consider x-y plane then, a point (x,y) represents a pixel. Every pixel has 3 values i.e [B,G,R]. These values overlap to give the color to each pixel.

	A video is a 3D-data where (x,y) repesents image and z represents the image at time t. We represent a video as a collection, that is collection of frames. Each frame is an image at a particular time t. Inorder to get a single value at each pixel of a frame we convert it to gray scaled image. The gray scale image is filtered to remove noises.
	
Created with vim

Authors: Karthik Bhat K., Kavana M. S., Manasa Bekal, Manisha Sudesh Naik

'''

#!/usr/bin/env python

import numpy as np
import cv2
from random import randint

help_message = '''
USAGE: peopledetect.py <image_names> ...

Press any key to continue, ESC to stop.
'''

def isInside(r, q):
	rx, ry, rw, rh = r
	qx, qy, qw, qh = q
	return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def drawDetections(image, rects, thickness = 1):
	i = 0
	for x, y, w, h in rects:
		# the HOG detector returns slightly larger rectangles than the real objects.
		# so we slightly shrink the rectangles to get a nicer output.
		r = randint(0,255)
		g = randint(0,255)
		b = randint(0,255)
		i = i + 1
		pad_w, pad_h = int(0.15*w), int(0.05*h)
		cv2.rectangle(image, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (r,g,b), thickness)
		cv2.putText(image, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (r,g,b), 1)


if __name__ == '__main__':
	import sys
	from glob import glob
	import itertools as it
	import os
	x = sys.argv[1]
	print help_message
	prevCount = 0
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
	cam = cv2.VideoCapture(x)
	while True:
		_, img = cam.read()
		gray = img
		kernel = np.ones((5,5),np.float32)/25
		img = cv2.filter2D(img,-1,kernel)
		img = cv2.blur(img,(5,5))
		img = cv2.GaussianBlur(img,(5,5),0)
		median = cv2.medianBlur(img,5)
		found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
		found_filtered = []
		for ri, r in enumerate(found):
			for qi, q in enumerate(found):
				if ri != qi and isInside(r, q):
					break
			else:
				found_filtered.append(r)
		drawDetections(gray, found_filtered, 3)
		if len(found) != prevCount:
			print '%d people are detected' % (len(found_filtered))
            	if len(found) > 5:
                	exec('vlc alert.mp3')
		prevCount = len(found)
		cv2.imshow('img', gray)
		out.write(gray)
		if 0xff & cv2.waitKey(5) == ord('q'):
			break
	cv2.destroyAllWindows()
