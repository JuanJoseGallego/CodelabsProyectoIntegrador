import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from time import time

# Carga imagen (sube archivos en Colab o usa una URL y descárgala)
img_path = "grupo.jpg"

img = cv2.imread(img_path)  # reemplaza por tu archivo
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detector = MTCNN()   # puedes ajustar min_face_size
t0 = time()
res = detector.detect_faces(img_rgb)
t1 = time()

print(f"Detected: {len(res)} rostro(s) • tiempo: {(t1 - t0)*1000:.1f} ms")
for r in res:
    print(r['confidence'], r['box'], r['keypoints'].keys())