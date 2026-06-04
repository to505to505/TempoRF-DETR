"""Stenosis-DetNet - sequence-aware coronary stenosis detection.

Reimplements Pang et al. 2021 (CMIG 89:101900). Main pieces:
SFFBoxHead (self-attention fusion of box features across the T-frame
window), VideoFasterRCNN (torchvision Faster-R-CNN + per-frame SFF), and
detnet.sca (cross-frame clustering + interpolation of missing frames).
Training scaffold mirrors psstt so the two trainers stay comparable.
"""
