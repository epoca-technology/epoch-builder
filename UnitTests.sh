#!/bin/bash
export PYTHONPATH=$(pwd)/dist
clear
python3 -m unittest discover -s dist/tests -p '*_test.py'