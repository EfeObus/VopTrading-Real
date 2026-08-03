#!/bin/bash

echo "Running Unit tests"

pytest --random-order --cov=voptrade --cov-config=.coveragerc tests/
