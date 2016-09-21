#!/bin/bash
# 
#Run the object tracking experiments with different videos
#Usage : test.sh[-w workspace][-d dataPath]

usage() 
{
cat << EOF
usage : $0 [ -w WORKSPACE ][ -d DATAPATH ] 
options : -w path to folder that contains build file (mandatory field) 
          -d path to folder containing the videos (mandatory field) 
          -h show this message 
EOF
}

while getopts w:d:h option; do
  case "${option}" in
    w) WORKSPACE=${OPTARG} ;;
    d) DATAPATH=${OPTARG} ;;
    h) usage exit ;;
  esac
done

if [[ -z $WORKSPACE ]] || [[ -z $DATAPATH ]]; then
  usage
  exit    
fi

videos=("Lemming")

images_subfolder="img/"

groundtruth_file="groundtruth_rect.txt"

initialization_modes=("File")

filters=("NoFilter")

particle_filter="sirparticle-filter"

#appearance_models=("BRISK" "histogram" "ORB" "SIFT" "SURF")

feature_detectors=("BRISK" "SIFT")
feature_descriptors=("BRISK" "SIFT")

cd $WORKSPACE
for video in "${videos[@]}"; do 
  input_path="${DATAPATH}/${video}/"        
  for initalizationMode in "${initialization_modes[@]}"; do    
    for filter in "${filters[@]}"; do   
      for detector in "${feature_detectors[@]}"; do 
        for descriptor in "${feature_descriptors[@]}"; do          
        ./jgtracker track $input_path $images_subfolder $groundtruth_file ${initalizationMode}  ${filter} -1 ${detector} ${descriptor} >> error_file.txt
        done
      done
    done
  done 
done


#echo "I need this to work!!!"
