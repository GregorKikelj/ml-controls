"""
  Generated using Konverter: https://github.com/ShaneSmiskol/Konverter
"""

import numpy as np

wb = np.load('export/model_s2_weights.npz', allow_pickle=True)
w, b = wb['wb']

def predict(x):
  x = np.array(x, dtype=np.float32)
  l0 = np.dot(x, w[0]) + b[0]
  l0 = np.maximum(0, l0)
  l1 = np.dot(l0, w[1]) + b[1]
  l1 = np.maximum(0, l1)
  l2 = np.dot(l1, w[2]) + b[2]
  l2 = np.maximum(0, l2)
  l3 = np.dot(l2, w[3]) + b[3]
  l3 = np.tanh(l3)
  return l3
