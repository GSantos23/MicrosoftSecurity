#!/bin/bash

DATA_DIR=../dataTest1
BIN_DIR=../bin
SRC_DIR=../src

VECTOR_DATA=$DATA_DIR/vector.bin

pushd ${SRC_DIR} && make; popd
sh ./create-data.sh

set -x
$BIN_DIR/distance $VECTOR_DATA

# This file compile de .c files to star word2vec algortihm
