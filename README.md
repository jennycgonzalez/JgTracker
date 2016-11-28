# Jgtracker

Jgtracker is a real time video tracking system that employs a particle filter to estimate the centroid location of an object and keypoint tracking and matching algorithms to estimate its scale and rotation. This implementation runs on Linux.

# License

Jgtracker is freely available under the BSD license. 

## Install Dependencies

```bash
sudo apt-get install build-essential
sudo apt-get install libtcmalloc-minimal4 libboost-system-dev libboost-filesystem-dev libboost-iostreams-dev
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

```

## Download and Install OpenCV with Extra Modules

```bash
git clone https://github.com/Itseez/opencv_contrib.git
cd opencv_contrib
git checkout 1d90d67016a83f9ed5c2eb2db1c864ddd27a93ea
cd ..
wget https://github.com/Itseez/opencv/archive/3.1.0.zip
unzip 3.1.0.zip
cd opencv-3.1.0
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DWITH_IPP=OFF -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j4
sudo make install
sudo ldconfig
```
## Compile and Run

```bash
qmake
make
./jgtracker track path-to-video-folder delay-in-miliseconds
```
When the value of "delay-in-miliseconds" is greater than zero, a pause of that amount of time is made between each frame. When its value is zero, it makes
the system wait infinitely for a key stroke before displaying the next frame. Finally, when it has a negative value the system does not display the frames while processing the video.

The video folder must have the following structure:

```.
├── config.ini
├── groundtruth.txt
├── img
│   ├── 0001.jpg
│   ├── 0002.jpg
│   ├── 0003.jpg
│   ├── 0004.jpg
│   ├── ...
```

The "config.ini" and "groundtruth.txt" files for 9 videos of the Visual Tracker Benchmark http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html are given in the "configuration-files" folder. 

## The Ground Truth File

For any video the "groundtruth.txt" file is required to initialize the target in the first frame. Spaces and not commas are used to separate the values in this file. In each row, the first two columns contain
the X and Y coordinates of the upper-left corner of the bounding box that surrounds the target. On the other hand, the third and fourth columns indicate the width and height of the bounding box.  

## The Tracking Results

When the system is done processing the video,
it saves the results in a text file named "jg-result-test-x.txt" inside the video folder, as shown in the following tree:

```.
├── config.ini
├── groundtruth.txt
├── img
│   ├── 0001.jpg
│   ├── 0002.jpg
│   ├── 0003.jpg
│   ├── 0004.jpg
│   ├── ...
│   └── 0725.jpg
└── results
    └── BRISK-BRISK
        └── jg-result-test-1.txt
```

The results file employs the same format as the ground truth file to define the bounding box surrounding the target in each frame.
The folder's name "BRISK-BRISK" comes from the keypoint extractor and descriptor methods given in the "config.ini" file:

```bash
keypoint_extractor=BRISK
keypoint_descriptor=BRISK
```




