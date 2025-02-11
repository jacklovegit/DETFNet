import os
import cv2

ann_path = r'data\CHASEDB1\annotations'
ann_list = ['training', 'validation']
for split in ann_list:
    print('Processing')
    path = os.path.join(ann_path,split)
    for i in os.listdir(path):
        mask = cv2.imread(os.path.join(path,i),0)
        mask[mask==255] =1
        # if '1st' in i:
        cv2.imwrite(os.path.join(path,i),mask)
        print("Saved mask ",i)
