''' Human hand (palm) detection and tracking.
Algorithm states: 
    State 0: human skin tone detection, subtraction of 3 successive frames to obtain movement of skin region. 
                  Detection of vertical upwards movement.
    State 1: classification of front human palm at the potential palm region detected in state 0.
    State 2: tracking the detected palm using template matching.
How to use:
    Raise your hand. If detected hold for a moment. If classified successfully - move and control. 

Created by: Tomislav Kis, tkis.cro@gmail.com

Date: 10 June 2016
'''

# Required moduls
import cv2
import numpy as np
from math import *


# Find the corresponding contour, inputs: array with previously detected contours, testing contour
def contains(contours_dict_array, cTest, xTest = 0, yTest = 0):
    if not type(cTest) is int:
        X, Y, Y_max = 0, 0, height+1
        for x in cTest:
            X += x[0]
            Y += x[1]
            if x[1] < Y_max: Y_max = x[1]
        X /= len(cTest)
        Y /= len(cTest)
    else: X, Y_max = xTest, yTest
    # X: x-center of testing contour, Y: y-center of testing contour, Y_max: y-top of testing contour
    
    if len(contours_dict_array) == 0: 
        return 0, [X, Y_max, -1, Y]    
    c_out, new_out = 0, [X, Y_max, 150, Y]
    for c in contours_dict_array:
        cX, cY = c['cX'], c['cY']
        v = [2.5*(cX - X), 0.5*(cY - Y_max)]
        dist = np.linalg.norm(v)        
        if dist < 90:
            if dist < new_out[2]: c_out, new_out[2] = c, dist
    return c_out, new_out            
    # returns the corresponding contour from the array, if it exists

# Contours analysis
def analyse_contours(contours):
    global contours_all
    for c in contours:
        area = cv2.contourArea(c)
        # Small contours are not relevant
        if area > 1000:
            # Fit ellipse around contour and anlyze it
            ellipse = cv2.fitEllipse(c)
            ellipsePoly =  cv2.ellipse2Poly((int(round(ellipse[0][0])), int(round(ellipse[0][1]))), (int(round(ellipse[1][0]/2)), int(round(ellipse[1][1]/2))), int(round(ellipse[2])), 0, 360, 1)         
            target_cont, new_cont = contains(contours_all, ellipsePoly)
            # If temporary contour does not exists in contours array append it
            if not target_cont:
                contours_all.append({ 'cX0' : new_cont[0], 'cY0' : new_cont[3], 'cX' : new_cont[0], 'cY' : new_cont[1], 'raising' : 0, 'removing' : 0, 'visited' : 1, 'delete' : 0})
                continue
            # If it exists update it
            else:                
                # Update position, set 'visited' flag
                target_cont['cX'], target_cont['cY'] = new_cont[0], new_cont[1]
                target_cont['visited'] = 1
                # Calculate movement vector, difference between temporary and starting position
                v0 = [new_cont[0] - target_cont['cX0'], target_cont['cY0'] - new_cont[1]]
                if v0[0] == 0: continue
                alpha = atan2(v0[1], v0[0])
                dist = np.linalg.norm(v0)
                # Vector angle x-axis:
                if 70 < alpha*180/3.14159 < 110: 
                    target_cont['raising'] = dist * sin(alpha)
    # Garbage collector for unwanted contours                        
    for c in contours_all:
        if c['visited'] == 0: c['delete'] += 1
        else: c['visited'] = 0
        if c['delete'] == 6: contours_all.remove(c) # and c['raising'] < 70
         
# State 0:
def state0():
    global sourceImage, oldImage, oldImageOlder, contours_all
    #sourceImage = cv2.GaussianBlur(sourceImage, (5, 5), 0) # useful but slow

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)   
    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)    
    #skinRegion = cv2.morphologyEx(skinRegion, cv2.MORPH_OPEN, kernel) # better after calculating flow (below) but could be useful here
    
    # Skin region flow
    kk = cv2.bitwise_xor(skinRegion, oldImage)        
    kk2 = cv2.bitwise_or(kk, oldImageOlder)
    kk2 = cv2.morphologyEx(kk2, cv2.MORPH_OPEN, kernel)
    #kk2 = cv2.erode(kk2,kernel2) # simulate MORPH_OPEN with different kernels for erode/dilate
    #kk2 = cv2.dilate(kk2,kernel3) # kernels in main
        
    # Do contour detection on skin region
    img_contours, contours, hierarchy = cv2.findContours(kk2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)   
    analyse_contours(contours)  
    
    # Save previous skinRegion frames
    oldImageOlder = kk
    oldImage = skinRegion

    # Draw points and vectors, update state and possible hand
    for c in contours_all:
        cv2.circle(sourceImage, (c['cX'], c['cY']), 5, (255, 0, 0), -1)
        if c['raising'] < 200: cv2.line(sourceImage, (c['cX0'], c['cY0']), (c['cX'], c['cY']), (255, 255, 255), 2)
        else: cv2.line(sourceImage, (c['cX0'], c['cY0']), (c['cX'], c['cY']), (0, 0, 255), 2)

        if c['raising'] > 200 and c['delete'] == 4: 
            global state1_counter
            state1_counter = 0
            return 1, c
    return 0, []           

# State 1:
def state1():
    # Global counters
    global sourceImage, hand, contours_all, state1_counter, state2_ready, state2_counter
    global crop_T, crop_B, crop_L, crop_R
    # Draw the rectangle around the detected hand
    if state1_counter == 0:
        crop_T = hand['cY'] - 40 if (hand['cY'] - 40 >= 0) else 0 
        crop_B = hand['cY'] + 120 if (hand['cY'] + 120 < height) else height - 1
        crop_L = hand['cX'] - 60 if (hand['cX'] - 60 >= 0) else 0
        crop_R = hand['cX'] + 60 if (hand['cX'] + 60 < width) else width - 1
    cv2.rectangle(sourceImage, (crop_L, crop_T), (crop_R, crop_B), (0, 0, 255), 1)
    # Detect palms in the smaller image around the detected hand
    img_state1 = sourceImage[crop_T : crop_B, crop_L : crop_R]
    img_YCrCb = cv2.cvtColor(img_state1, cv2.COLOR_BGR2YCR_CB)
    img_skinRegion = cv2.inRange(img_YCrCb, min_YCrCb, max_YCrCb)    
    img = cv2.bitwise_and(img_state1, img_state1, mask = img_skinRegion)
    palms = palm_cascade.detectMultiScale(img, 1.3, 2)
    # State analysis
    if len(palms) > 0: state2_ready += 1        
    state1_counter += 1    
    if state1_counter > 20 and state2_ready > 9:        
        # If classifier detected palms in 9 of 20 frames go to state 2     
        # Calculates x and y projection from the skinRegion image
        hist_x = cv2.reduce(img_skinRegion, 0, 0, dtype=cv2.CV_32S)
        hist_x = cv2.divide(hist_x, 255)
        hist_y = cv2.reduce(img_skinRegion, 1, 0, dtype=cv2.CV_32S)
        hist_y = cv2.divide(hist_y, 255)
        wL, wR = 120, 0
        mean = cv2.mean(hist_x)[0] * 1.15
        center_val = cv2.sumElems(hist_x)[0] / 2
        center_dist = center_val
        center_tmp = 0
        center_idx = 0
        for i, h in enumerate(hist_x[0]):
            center_tmp += h
            if abs(center_tmp - center_val) < center_dist: center_dist, center_idx = abs(center_tmp - center_val), i
            if h > mean and i < wL: wL = i
            if h > mean and i > wR: wR = i            
        wT, wB = 160, 0
        mean = cv2.mean(hist_y)[0] * 1.2
        for i, h in enumerate(hist_y):
            if h > mean and i < wT: wT = i
            if h > mean and i > wB: wB = i
        # Set position, width and height of the palm calculated from projections and mean values
        x, y, w, h = wL, wT, wR - wL, wR - wL
        if w > 80: w, h = 80, 80
        # Update the center of the palm with meanShift algorithm
        track_window = (x, y, w, h)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 23, 1)            
        ret, track_window = cv2.meanShift(img_skinRegion, track_window, term_crit)
        x, y, w_tmp, h_tmp = track_window
        y = int(y - h * 0.1)
        hand['bounding'] = (x + crop_L, y + crop_T, w, h)
        state2_ready = 0
        state2_counter = 0
        # Returns state 2 and hand dictionary with position, width and height of the palm
        return 2
    elif state1_counter > 20: 
        state2_ready = 0
        contours_all, hand = [], []  
        imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)      
        skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)    
        global oldImage, oldImageOlder
        oldImageOlder = skinRegion
        oldImage = skinRegion
        return 0
    return 1

# State 2:
def state2():
    global sourceImage, hand, contours_all, state0_ready, state2_counter
    global img_hand, w0, h0, width, height, x, y, x_tmp_old, y_tmp_old
    # Initialize the representing image of the palm
    if state2_counter == 0:    
        x_tmp_old, y_tmp_old = 100, 100
        x, y, w, h = hand['bounding']
        x1 = x if (x >= 0) else 0
        y1 = y if (y >= 0) else 0
        x2 = x + w if (x + w < width) else width - 1
        y2 = y + h if (y + h < height) else height - 1
        img_hand = sourceImage[y1 : y2, x1 : x2]
        w0, h0 = int(w * 1.25), int(h * 1.25)
    img_hand_YCrCb = cv2.cvtColor(img_hand,cv2.COLOR_BGR2YCR_CB)   
    skinRegion = cv2.inRange(img_hand_YCrCb,min_YCrCb,max_YCrCb)    
    img_hand = cv2.bitwise_and(img_hand, img_hand, mask=skinRegion)  
    # Calculate boundaries of the search image
    x3 = x - w0 if (x - w0 >= 0) else 0
    y3 = y - h0 if (y - h0 >= 0) else 0
    x4 = x + w0 + w0 if (x + w0 + w0 < width) else width - 1
    y4 = y + h0 + h0 if (y + h0 + h0 < height) else height - 1
    img_search = sourceImage[y3 : y4, x3 : x4]        
    img_YCrCb = cv2.cvtColor(img_search, cv2.COLOR_BGR2YCR_CB)
    skinRegion = cv2.inRange(img_YCrCb, min_YCrCb, max_YCrCb)          
    # Template matching on the relevant part of the sourceImage surrounding the detected palm
    img_search = cv2.bitwise_and(img_search, img_search, mask = skinRegion)                  
    result = cv2.matchTemplate(img_search, img_hand, 0)
    result = cv2.normalize(result, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)        
    # New position of the palm on the sourceImage calculated from detected relevant position in the search image
    x = minLoc[0] + x3 if (minLoc[0] + x3 < width) else width - 1
    y = minLoc[1] + y3 if (minLoc[1] + y3 < height) else height -1   
    x2 = x + w0 if (x + w0 < width) else width - 1
    y2 = y + h0 if (y + h0 < height) else height - 1
    x_tmp = int(round(x + w0 / 2)) if (x + w0 / 2 < width) else width - 1
    y_tmp = int(round(y + h0 / 2)) if (y + h0 / 2 < height) else height -1      
    
    # Tracking lost or hand static leads to state 0
    if abs(x_tmp - x_tmp_old) < 15 and abs(y_tmp - y_tmp_old) < 15: state0_ready += 1
    elif state0_ready > 3: state0_ready -= 3
    else: state0_ready = 0
    if state0_ready > 20: 
        contours_all, hand = [], []
        state0_ready = 0
        return 0, 0, 0
    if not (state2_counter % 7): 
        state2_counter = 0
        x_tmp_old, y_tmp_old = x_tmp, y_tmp  
    img_hand = sourceImage[y : y2, x : x2] 
    # Circle on the postion of the detected hand, mirrored
    cv2.circle(sourceImage, (x_tmp, y_tmp), 7, (0, 0, 255), -1)        
    state2_counter += 1         
    return 2, x_tmp, y_tmp
              

# Global variables
contours_all = []
# Constants for finding range of skin color in YCrCb
min_YCrCb = np.array([80, 133, 77],np.uint8) # or [0, 133, 77]
max_YCrCb = np.array([255, 173, 127],np.uint8)

# Create a window to display the camera feed, fullscreen, without title bar
cv2.namedWindow('Camera Output', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Camera Output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Get pointer to video frames from primary device
videoFrame = cv2.VideoCapture(0)

# Initialize hand haar cascade classifier
palm_cascade = cv2.CascadeClassifier('../haarcascades/Hand.Cascade.1.xml') 
print 'Palm classifier initialized:', not palm_cascade.empty()

# Grab video frame, decode it and return next video frame
readSucsess, sourceImage = videoFrame.read()
height, width, channels = sourceImage.shape
print height, width, channels

# Convert to YCbCr color space and detect human skin tone region
imageYCrCb = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2YCR_CB)
oldImage = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
oldImageOlder = oldImage
# old and older image for flow analysis

# Main kernel:
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# Testing kernels in state 0:
#kernel2 = np.array( [ [0, 1, 0], [0, 1, 0], [0, 1, 0] ], np.uint8 )
#kernel3 = np.array( [ [0, 0, 0], [1, 1, 1], [0, 0, 0] ], np.uint8) 

# Initialize flags:
step, hand = 0, []
state = 0
state1_counter = 0
state2_ready = 0
state2_counter = 0
state0_ready = 0
fist_pressed = 0
keyPressed = -1 # -1 indicates no key pressed

# Process the video frames
while(keyPressed != 27): # any key pressed has value >= 0
    # Calculate every 3rd frame (30fps camera). Optimizes the algorithm without any significant loss.
    step += 1
    if step % 3 !=0 and state==0: continue
    step = 0
    
    # Grab a video frame, decode it and return next video frame
    readSucsess, sourceImage = videoFrame.read()
            
    if state == 0: state, hand = state0()
    elif state == 1: state = state1()          
    elif state == 2: state, x_tmp, y_tmp = state2()                              
                           
    # Check for user input to close the program
    keyPressed = cv2.waitKey(1) & 0xFF # wait 1 milisecond in each iteration of while loop              
              
    cv2.imshow('Camera Output', sourceImage)
    
#print doc
#print state
#print contours_all
# Close window and camera after exiting the while loop
cv2.destroyAllWindows()
videoFrame.release()
