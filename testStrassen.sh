#!/bin/bash

gcc -Wall -std=c99 -fopenmp -O2 -o ss strassen.c

./ss 512