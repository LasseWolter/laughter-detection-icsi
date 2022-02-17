#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage concat_laughs.sh <out-dir>" 
  exit
fi

sox $(for f in $1/chan*.wav; do echo -n "$f break.wav "; done) ${1}/all.wav
echo "Stored concatenated laughs in ${1}/all.wav"

