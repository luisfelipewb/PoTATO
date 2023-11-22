#!/bin/bash

source ~/venv/yolo-v5/bin/activate
cd ./experiments
./train_yolo.sh
deactivate

source ~/venv/potato/bin/activate
./train_det2.sh
cd ..
deactivate

echo "Done"
