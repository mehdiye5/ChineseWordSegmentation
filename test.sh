#!/bin/bash
cp default.py answer/zhsegment.py
cp default.ipynb answer/zhsegment.ipynb
python3 zipout.py
python3 check.py