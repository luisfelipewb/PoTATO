#!/bin/bash        ../runs/

image_list=(
"exp02_frame00554_"
"exp03_frame00128_"
"exp03_frame00225_"
"exp04_frame00109_"
"exp04_frame01021_"
"exp04_frame01568_"
"exp04_frame01856_"
"exp05_frame009963_"
"exp05_frame042448_"
"exp05_frame042483_"
"exp06_frame025794_"
"exp07_frame021778_"
"exp07_frame031498_"
"exp07_frame032023_"
"exp07_frame033238_"
"exp07_frame041963_"
"exp07_frame041973_"
"exp07_frame049233_"
)

ext_list=("mono" "rgb" "rgbdif" "dolp" "pol" "pauli")

file_list=()
for ext in ${ext_list[@]}
do
	for image in ${image_list[*]}
       	do 
		file_list+="output/crop_bbox/$image$ext.png "
	done
done

montage $file_list -geometry 100x100+1+1 -tile x6 output/crop_tiled.jpg




































