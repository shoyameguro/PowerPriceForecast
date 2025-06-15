#!/bin/bash
# predict.sh
python -m src.interface.predict --models output/models --input data/test/test.feather --out output/submissions/submission.csv
