import os

import cv2

video_writer = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
images_path = r"C:\Users\30505\Desktop\dataset\train\MOT16-02\img1"#mot16数据集路径
images_list = os.listdir(images_path)
images_list.sort()

for image_name in images_list:
    image = cv2.imread(os.path.join(images_path, image_name))
    video_writer.write(image)
    show = cv2.resize(image, (1280, 720))
    cv2.imshow("test", show)
    if cv2.waitKey(10) != ord('q'):
        pass
