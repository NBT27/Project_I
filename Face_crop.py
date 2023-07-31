from mtcnn import MTCNN
import cv2
import os
# import uuid


detector = MTCNN()
def detect_face(image, folder_out, file_name):
    img = cv2.imread(image)
    faces = detector.detect_faces(img)
    for face in faces:
        bounding_box = face['box']
        im  = img[ bounding_box[1]:bounding_box[1]+bounding_box[3],
                bounding_box[0]:bounding_box[0]+bounding_box[2]]
        
        # file_path = folder_out +"/"+ '{}.jpg'.format(uuid.uuid1())
        file_path = os.path.join(folder_out, file_name)
        cv2.imwrite(file_path, im)
        print(file_path)
        
# image_folder_in = ".\data_in\Amber Heard"
# image_folder_out =".\data_out\Amber"   
        
def crop_face_one_folder(root_folder, image_folder_out):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        folder_out = os.path.join(image_folder_out, folder_name)
        
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path,image_name)
            
            detect_face(image_path,folder_out,image_name)

# đếm số thư mục có nhiều hơn 1 ảnh
def count_directories_with_multiple_files(root_dir):
    counts = {}
    for root, dirs, files in os.walk(root_dir):
        # Đếm số lượng tệp trong thư mục hiện tại
        count = len(files)
        # Lưu trữ số lượng tệp trong dictionary
        counts[root] = count
    # Đếm số thư mục có nhiều hơn một tệp
    num_directories_with_multiple_files = sum(1 for count in counts.values() if count > 1)
    return num_directories_with_multiple_files

root_folder = "C:/Users/NguyenBaThanh/Desktop/Project I/Project/lfw"
image_folder_out = "C:/Users/NguyenBaThanh/Desktop/Project I/Project/lfw"

crop_face_one_folder(root_folder, image_folder_out)

# count = count_directories_with_multiple_files(root_folder)

# print(count)