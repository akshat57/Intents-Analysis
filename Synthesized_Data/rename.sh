#!/bin/sh
cd test_female_2
rm 9.wav

for i in 1 2 3 4 5 6 7 8
do
  mv $i.wav 0$i.wav
done
