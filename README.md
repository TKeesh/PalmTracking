## Human hand (palm) detection and tracking.

### Algorithm states: 

State 0: human skin tone detection, subtraction of 3 successive frames to obtain movement of skin region. Detection of vertical upwards movement.

State 1: classification of front human palm at the potential palm region detected in state 0.

State 2: tracking the detected palm using template matching.

### How to use:
    Raise your hand. If detected hold for a moment. If classified successfully move and control. 

Created by: Tomislav Kis, tkis.cro@gmail.com

Date: 10 June 2016

### ToDo

`hand.png` as cursor

Fist detection instead of space button for simulating mouse button click