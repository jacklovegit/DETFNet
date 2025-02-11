import cv2
import os

label_path = 'data/stare/annotations/validation'

save_label = 'data/stare/annotations/validation'
for i in os.listdir(label_path):
    label = cv2.imread(os.path.join(label_path,i),0)
    # print(set(label.flatten()))
    label[label==255] = 1

    cv2.imwrite(os.path.join(save_label,i),label)