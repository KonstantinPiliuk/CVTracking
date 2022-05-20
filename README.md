# CVTracking
Approach to tracking data extraction from football broadcast

## Scope
The project is dedicated on tracking data extraction from football broadcast video files.
The repository contains last progress and cannot be used as a final solution for custom needs.
Currently the algorith can be used on a given toy video file only. Its usage with other videos is likely to end up with incosistent results.

## Documentation
Detailed description of the algorithm and some documentation for code are provided [here](https://github.com/KonstantinPiliuk/CVTracking/wiki).

## Installation
To use the algorythm all dependencies are to be installed into python environment. It can be done with pip.
1. Clone repository

`git clone https://github.com/KonstantinPiliuk/CVTracking.git`

`cd CVTracking`

2. Install dependecies

`pip install -r requirements.txt`

## Inference
To run the inference `inference.py` script should be called

`python inference.py --vid 'inputs/broad_test_5sec.mp4' --w 'inputs/coords_model.pt' --log 'output.csv'`

The mandatory arguments are:

`--vid` - path to video file with football TV broadcast

`--w`   - pretrained weights for keypoints detection **(by default uses weights in inputs/coords_model.pt)**

`--log` - output style. Two options are available: csv file (default) and logging to sql database. 
To get csv output provide filename e.g. 'output'. For sql output provide 'sql' string.

 Output as .csv includes players postions only after the whole funnel. Output in sql includes output of all inner stages.
 
 Optional arguments specify sql connection via SQLAlchemy if `--log` == 'sql':
 
 `--d`  - SQLAlchemy dialect
 
 `--h`  - host and port address e.g. h.h.h.h:pppp
 
 `--db` - database name
 
 `--u`  - username
 
 `--p`  - password
