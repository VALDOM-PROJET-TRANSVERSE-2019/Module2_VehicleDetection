# The big picture
This repository is part of the ‘projet transverse’ of the [Advanced Master ValDom](http://www.enseeiht.fr/fr/formation/masteres-specialises/valorisation-des-donnees-massives.html) which is co-accredited by INP-ENSEEIHT and INSA Toulouse.

This year (2019/2020) the goal of the project is to develop a video analysis service. The main functionality is the recognition and tracking of vehicles in order to be able to estimate the emission rate (Co2) produced by traffic in the areas concerned.

# Module2_VehicleDetection
This module is responsible, from the images extracted by the [preprocessing module](https://github.com/VALDOM-PROJET-TRANSVERSE-2019/Module1_VideoPreprocessing), for detecting the vehicles on the images and for generating the corresponding metadata.


# Interface
## Input
- frames_path: a string with the path where are the frames which will be used to process

## Output
- frame_contours: the list containing the bounding boxes in the corresponding frame


Team:
Achraf HAMID
Aymen GHARBI
Chouaib NEMRI

