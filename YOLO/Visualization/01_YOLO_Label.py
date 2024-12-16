import os
import cv2

images_folder = r'D:\images'
labels_folder = r'D:\labels'
output_folder = r'D:\im_labels'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]

for image_file in image_files:
    label_file = os.path.splitext(image_file)[0] + '.txt'

    image_path = os.path.join(images_folder, image_file)
    image = cv2.imread(image_path)

    img_height, img_width = image.shape[:2]

    label_path = os.path.join(labels_folder, label_file)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.split())

                x_center, y_center, width, height = x_center * img_width, y_center * img_height, width * img_width, height * img_height

                x_min, y_min = int(x_center - width / 2), int(y_center - height / 2)
                x_max, y_max = int(x_center + width / 2), int(y_center + height / 2)

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, image)

cv2.destroyAllWindows()
