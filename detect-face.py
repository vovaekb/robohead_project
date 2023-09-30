import sys, time
# imports for face regognition
from optparse import OptionParser
import math
import datetime
import serial

from threading import Thread
import cv2.cv as cv
import cv2
import numpy as np
import video

from servo import Servo

# imports for face regognition
#
#
# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned
# for accurate yet slow object detection. For a faster operation on real video
# images the settings are:
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING,
# min_size=<minimum possible face size
 
min_size = (20, 20)
image_scale = 2
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0

COM = 'COM4'

MORPH_ON = 1
HOUGH_ON = 1
SERVO_ON = 1
BLUR_ON = 1
SHOW_MAIN = 1
SHOW_PATH = 0
SHOW_THRESH = 0
FLIP = 1

COLOR_RANGE={
    'ball_light': (np.array((20, 70, 170), np.uint8), np.array((40, 170, 255), np.uint8)),
    'ball_dark': (np.array((0, 170, 120), np.uint8), np.array((20, 240, 255), np.uint8)),
}

 
def detect_and_draw(img, cascade):
    gray = cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv.CreateImage(
        (cv.Round(img.width / image_scale),
        cv.Round (img.height / image_scale)), 8, 1
    )
 
    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
 
    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)
 
    cv.EqualizeHist(small_img, small_img)
 
    centre = None
 
    if(cascade):
        t = cv.GetTickCount()
        # HaarDetectObjects takes 0.02s
        faces = cv.HaarDetectObjects(
            small_img, cascade, cv.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size
        )
        t = cv.GetTickCount() - t
        if faces:
            for ((x, y, w, h), n) in faces:
                # the input to cv.HaarDetectObjects was resized, so scale the
                # bounding box of each face and convert it to two CvPoints
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)
                # get the xy corner co-ords, calc the centre location
                x1 = pt1[0]
                x2 = pt2[0]
                y1 = pt1[1]
                y2 = pt2[1]
                centrex = x1+((x2-x1)/2)
                centrey = y1+((y2-y1)/2)
                centre = (centrex, centrey)
 
    cv.ShowImage("result", img)
    return centre


def get_delta(loc, span, max_delta, centre_tolerance):
    """How far do we move on this axis to get the webcam
       centred on the face?
       loc is the face's centre for this axis
       span is the width or height for this axis
       max_delta is the max nbr of degrees to move on this axis
       centre_tolerance is the centre region where we don't allow movement
       """
    framecentre = span/2
    delta = framecentre - loc
    if abs(delta) < centre_tolerance: # within X pixels of the centre
        delta = 0 # so don't move - else we get weird oscillations
    else:
        is_neg = delta <= 0
        to_get_near_centre = abs(delta) - centre_tolerance
        if to_get_near_centre > 35:
            delta = 4
        else:
            # move slower if we're closer to centre
            if to_get_near_centre > 25:
                delta = 3
            else:
                # move real slow if we're very near centre
                delta = 1
        if is_neg:
            delta = delta * -1
    return delta

def nothing( *arg ):
    pass

def createPath( img ):
    h, w = img.shape[:2]    
    return np.zeros((h, w, 3), np.uint8)
    

class Tracker(Thread):
    def __init__(self, color, color_2=None, flag=0):
        Thread.__init__(self)
        self.color = color
        self.path_color = cv.CV_RGB(0,110,0)

        self.lastx = 0
        self.lasty = 0

        self.time = 0
        
        self.h_min = COLOR_RANGE[color][0] 
        self.h_max = COLOR_RANGE[color][1]
        if color_2:
            self.h_min_2 = COLOR_RANGE[color_2][0]
            self.h_max_2 = COLOR_RANGE[color_2][1]

        self.flag = flag
        
        if self.flag:
            cv2.namedWindow( self.color )
   
    def poll(self,img):
        par1 = 80#40
        par2 = 50#67
        h, w = img.shape[:2]

        dt = time.clock() - self.time
        self.time = time.clock()

        hsv = cv2.cvtColor(img, cv.CV_BGR2HSV )

        thresh = cv2.inRange(hsv, self.h_min, self.h_max)
        thresh_2 = cv2.inRange(hsv, self.h_min_2, self.h_max_2)
        thresh = cv2.bitwise_or(thresh, thresh_2)

        if MORPH_ON:
            st1 = cv2.getStructuringElement(
                cv2.MORPH_RECT, (21, 21), (10, 10)
            )
            st2 = cv2.getStructuringElement(
                cv2.MORPH_RECT, (11, 11), (5, 5)
            )
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_CLOSE, st1
            )
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, st2
            )

        if BLUR_ON:    
            thresh = cv2.GaussianBlur(thresh, (5, 5), 2)
        
        circles = None
        if HOUGH_ON:
            circles = cv2.HoughCircles( thresh,
                                        cv.CV_HOUGH_GRADIENT,
                                        2, h/4, np.array([]),
                                        par1, par2, 5, 0) 
           
        if circles is not None:
            maxRadius = 0
            x = 0
            y = 0
            found = False
            
            for c in circles[0]:
                found = True
                if c[2] > maxRadius:
                    maxRadius = int(c[2])
                    x = int(c[0])
                    y = int(c[1])
            if found:
                cv2.circle(img, (x, y), 3, (0,255,0), -1)
                cv2.circle(img, (x, y), maxRadius, (255,0,0), 3)
            
                if self.lastx > 0 and self.lasty > 0:
                    cv2.line(self.path, (self.lastx, self.lasty), (x,y), self.path_color, 5)

                xspeed = abs(x - self.lastx)/dt;
                yspeed = abs(y - self.lasty)/dt;
                
                self.lastx = x
                self.lasty = y

                if SERVO_ON:
                    yaw = (x*1./w)*50.0 - 25.0
                    pitch = (y*1./h)*39.0 - 19.5
                    #sctrl.shift(0, yaw)
                    #sctrl.shift(1, -pitch)
                    sctrl.shift(0, (x*1./w)*20-10)
                    sctrl.shift(1, -((y*1./h)*20-10))

        if SHOW_PATH:
            img = cv2.add( img, self.path)

        if FLIP:
            img = cv2.flip(img,0)
            thresh = cv2.flip(thresh,0)

        if self.flag:
            cv2.imshow(self.color, thresh)

        text = '%0.1f' % (1./dt)
        cv2.putText( img, text, (20, 20), cv2.FONT_HERSHEY_PLAIN,
                     1.0, (0, 110, 0), thickness = 2)

        if SHOW_MAIN:
             cv2.imshow('result', img)  

if __name__ == '__main__':
# parse cmd line options, setup Haar classifier
    parser = OptionParser(usage = "usage: %prog [options] [camera_index]")
    parser.add_option("-c", "--cascade",
                      action="store",
                      dest="cascade",
                      type="str",
                      help="Haar cascade file, default %default",
                      default = "/home/vladimir/OpenCV/opencv-2.4.7/data/haarcascades/haarcascade_frontalface_alt.xml")
    (options, args) = parser.parse_args()
 
    cascade = cv.Load(options.cascade)
 
    if len(args) != 1:
        parser.print_help()
        sys.exit(1)
 
    input_name = args[0]
    if input_name.isdigit():
        capture = cv.CreateCameraCapture(
            int(input_name)
        )
    else:
        print "We need a camera input! Specify camera index e.g. 0"
        sys.exit(0)
 
    cv.NamedWindow("result", 1)
 
    if capture:
        frame_copy = None
 
        while True:
            frame = cv.QueryFrame(capture)
            if not frame:
                cv.WaitKey(0)
                break
            if not frame_copy:
                frame_copy = cv.CreateImage(
                    (frame.width,frame.height),
                    cv.IPL_DEPTH_8U, frame.nChannels
                )
            if frame.origin == cv.IPL_ORIGIN_TL:
                cv.Copy(frame, frame_copy)
            else:
                cv.Flip(frame, frame_copy, 0)
 
            centre = detect_and_draw(frame_copy, cascade)
 
            if centre is not None:
                cx = centre[0]
                cy = centre[1]
 
                # modify the *-1 if your x or y directions are reversed!
                xdelta = get_delta(cx, frame_copy.width, 6, 15) * -1
                ydelta = get_delta(cy, frame_copy.height, 1, 25) * -1
 
                # on my camera I introduce a delay after movements
                # else my assembly wobbles and the webcam transmits
                # a non-centred image, so weird oscillations can occur
                total_delta = abs(xdelta)+abs(ydelta)
                if total_delta > 0:
                #    xygo = (xygo[0]+xdelta,xygo[1]+ydelta)
 
                    sleep_for = 1/10.0*min(total_delta, 10)
                    sleep_for = min(sleep_for, 0.4)
 
                #    move_servos(xygo)
                else:
                    sleep_for = 0
 
            if cv.WaitKey(10) >= 0: # 10ms delay
                break
 
    cv.DestroyWindow("result")

    if SERVO_ON:
       sctrl = Servo( com=COM )
       sctrl.setpos(0,45)
       sctrl.setpos(1,45)

