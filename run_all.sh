#!/bin/bash

source ~/venv/potato/bin/activate
cd ./experiments
./train_yolo.sh
./train_det2.sh
cd ..

echo "Done"
