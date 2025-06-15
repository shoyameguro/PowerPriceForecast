#!/bin/bash
# predict.sh
python -m src.interface.predict --models output/models --input data/test/test.pkl --out output/submissions/submission.csv
