## Human hand (palm) detection and tracking.

Created by: Tomislav Kis, tkis.cro@gmail.com

Date: 10 June 2016

### Algorithm states

State 0: human skin tone detection, subtraction of 3 successive frames to obtain movement of skin regions, detection of vertical upward movement

State 1: classification of front human palm at the potential palm region

State 2: tracking the detected palm using template matching

### Requirements

Python2 with: OpenCV, Numpy

 **Note:** Can be easily modified for Python3 

### How to use
Raise your hand. If detected hold for a moment. If classified successfully move and control. 

### ToDo

`hand.png` as cursor

Fist detection instead of space button for simulating mouse button click