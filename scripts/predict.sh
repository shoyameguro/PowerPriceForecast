#!/bin/bash
# predict.sh
python -m src.interface.predict --models output/models --input data/processed/test.feather --out output/submissions/submission.csv