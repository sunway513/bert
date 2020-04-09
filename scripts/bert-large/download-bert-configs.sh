#!/bin/bash
DUMP_DIR=/tmp
cd ../..

# download bert base
MODEL_NAME=bert_base
mkdir -p configs/$MODEL_NAME
wget -c https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -P $DUMP_DIR
unzip -q -o $DUMP_DIR/uncased_L-12_H-768_A-12.zip -d $DUMP_DIR
cp $DUMP_DIR/uncased_L-12_H-768_A-12/bert_config.json configs/$MODEL_NAME/bert_config.json
cp $DUMP_DIR/uncased_L-12_H-768_A-12/vocab.txt configs/$MODEL_NAME/vocab.txt

# download bert large
MODEL_NAME=bert_large
mkdir -p configs/$MODEL_NAME
wget -c https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip -P $DUMP_DIR
unzip -q -o $DUMP_DIR/uncased_L-24_H-1024_A-16.zip -d $DUMP_DIR
cp $DUMP_DIR/uncased_L-24_H-1024_A-16/bert_config.json configs/$MODEL_NAME/bert_config.json
cp $DUMP_DIR/uncased_L-24_H-1024_A-16/vocab.txt configs/$MODEL_NAME/vocab.txt
