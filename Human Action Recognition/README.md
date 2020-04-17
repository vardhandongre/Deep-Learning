# Human Action Recognition 
Understanding spatial and temporal information learning. 

In this work:
  * Single Frame Model: Fine-tuned a 50-layer ResNet model (pretrained on ImageNet) on single UCF-101 video frames
  * Sequence Model: Fine-tuned a 50-layer 3D ResNet model (pretrained on Kinetics) on UCF-101 video sequences

## Dataset: UCF-101 human action recognition dataset
Original Report for Dataset: [Report](https://www.crcv.ucf.edu/papers/UCF101_CRCV-TR-12-01.pdf)
Dataset consists of 13,320 videos between ~2-10 seconds long of humans performing one of 101 possible actions. The dimensions of each frame are 320 by 240.
