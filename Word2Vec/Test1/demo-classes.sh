#!/bin/bash

DATA_DIR=../dataTest1
BIN_DIR=../bin
SRC_DIR=../src

pushd ${SRC_DIR} && make; popd
sh $DATA_DIR/create-data.sh

TEXT_DATA=$DATA_DIR/2012.txt
CLASSES_DATA=$DATA_DIR/someWords.txt

pushd ${SRC_DIR} && make; popd
  
if [ ! -e $CLASSES_DATA ]; then
  if [ ! -e $TEXT_DATA ]; then
		sh ./create--data.sh
	fi
  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors...
  time $BIN_DIR/word2vec -train $TEXT_DATA -output $CLASSES_DATA -cbow 0 -size 100 -window 5 -negative 25 -hs 0 -sample 1e-4 -threads 20 -iter 15 -classes 22
fi

sort $CLASSES_DATA -k 2 -n > $DATA_DIR/classes.sorted.txt

echo The word classes were saved to file $DATA_DIR/classes.sorted.txt
