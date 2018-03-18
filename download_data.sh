#!/bin/bash
# For each Mesh model of Shape-Net-Core download 1 point-cloud with 2048 points
# sampled uniformly at random (around 1.4GB).
wget https://www.dropbox.com/s/vmsdrae6x5xws1v/shape_net_core_uniform_samples_2048.zip?dl=0
mv shape_net_core_uniform_samples_2048.zip\?dl\=0 shape_net_core_uniform_samples_2048.zip
unzip shape_net_core_uniform_samples_2048.zip
rm shape_net_core_uniform_samples_2048.zip
mkdir -p data
mv shape_net_core_uniform_samples_2048 data
