#!/bin/bash
pip install openmim
mim install mmcv-full==1.7.1
pip install -r requirements.txt
cd main/transformer_utils && python setup.py install