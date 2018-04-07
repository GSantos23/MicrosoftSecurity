#!/bin/bash

DATA_DIR=../dataTest1
BIN_DIR=../bin
SRC_DIR=../src

TEXT_DATA=$DATA_DIR/2012.txt
VECTOR_DATA=$DATA_DIR/vector.bin

if [ ! -e $VECTOR_DATA ]; then
  if [ ! -e $TEXT_DATA ]; then
		sh ./create-data.sh
	fi
  echo ----------------------------------------------------------------------
  echo -- Training vectors --
  time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size 100 -window 5 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 10
fi
