#!/bin/bash
export PYTHONPATH=$(pwd)/dist
python3 -m unittest discover -s dist/tests -p '*_test.py'