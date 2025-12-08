import pickle
import numpy as np

with open("calibrate.pkl", "rb") as f:
    data = pickle.load(f)
    
for cam_id, matrix in data.items():
    print(f"--- Camera {cam_id} ---")
    print(matrix)
    print("Position (x,y,z):", matrix[:3, 3])
