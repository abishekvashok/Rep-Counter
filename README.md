## RepCounter

This repository is a fork of posenet-python(https://github.com/rwightman/posenet-python)
RepCounter uses posenet to analyse different points and count repetitions. This is highly
skilled in counting no of reps of exercises.

### Working demos

![Demo1](./demo/demo1.gif)
![Demo2](./demo/demo2.gif)
![Demo3](./demo/demo3.gif)

### Install

A suitable Python 3.x environment with a recent version of Tensorflow is required.
Development and testing was done with Conda Python 3.6.8 and Tensorflow 1.12.0 on Linux.
All prerequisitites can be found in the `requirements.txt` file. It is adviced that you
install them in a virtual environment.
* Installation in Virtual Environment

   It is recommended that you use [`virtualenv`](https://virtualenv.pypa.io/en/stable/installation/)
   and [`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/install.html) to maintain
   a clean Python 3 environment. Create a `virtualenv` and install the requirements:

  ```sh
  $ mkvirtualenv -p python3 repCounter
  (badgeyay) $ deactivate         			# To deactivate the virtual environment
  $ workon repCounter             			# To activate it again
  (badgeyay) $ pip3 install -r requirements.txt 	# Install the requirements
  ```
* System Wide Installation

  ```sh
  pip3 install -r requirements.txt
  ```
  Note: This might change versions of exisiting python packages and hence is *not recommended*.
### Usage

Run the app in eye candy mode with:
```sh
python3 run.py
```
Usage information: 
```sh
usage: run.py [-h] [--variance VARIANCE] [--model MODEL] [--cam_id CAM_ID]
              [--cam_width CAM_WIDTH] [--cam_height CAM_HEIGHT]
              [--scale_factor SCALE_FACTOR] [--file FILE]

optional arguments:
  -h, --help            show this help message and exit
  --variance VARIANCE   The tolerance for the model in integers
  --model MODEL         The model to use, available versions are 101 (def.),
                        102, 103 etc
  --cam_id CAM_ID       The respective cam id to use (default 0)
  --cam_width CAM_WIDTH
                        The width of the webcam in pixels (def. 1280)
  --cam_height CAM_HEIGHT
                        The height of the webcam in pixels (def. 780)
  --scale_factor SCALE_FACTOR
                        The scale factor to use (default: .7125)
  --file FILE           Use the video file at specified path instead of live
                        cam
```
The first time the apps is run (or the library is used) model weights will be downloaded from the TensorFlow.js version and converted on the fly.

The model can be specified with the '--model` argument by using its ordinal id (0-3) or integer depth multiplier (50, 75, 100, 101). The default is the 101 model.

Count can be reset by pressing `r` or `R` on the keyboard.
Exit the app by pressing the `q` or `Q` on the keyboard.

### Credits

This repository is a fork of posenet-python (https://github.com/rwightman/posenet-python)

The original model, weights, code, etc. was created by Google and can be found at https://github.com/tensorflow/tfjs-models/tree/master/posenet

This work is in no way related to Google.

