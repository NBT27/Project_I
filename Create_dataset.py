import os
import random
import numpy as np
import cv2
import tensorflow as tf
import uuid
from PIL import Image

def load_lfw_dataset(data_dir):
    # Load the LFW dataset from the specified directory
    images = []
    labels = []
    
    for person_dir in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_dir)
        
        if not os.path.isdir(person_path):
            continue
        
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            
            # Read the image and store it in the dataset
            # Read in image from file path
            byte_img = tf.io.read_file(image_path)
            # Load in the image 
            img = tf.io.decode_jpeg(byte_img)
    
            # Preprocessing steps - resizing the image to be 100x100x3
            img = tf.image.resize(img, (100,100))
            # Scale image to be between 0 and 1 
            img = img / 255.0
            images.append(img)
            # Assign a unique label to each person
            person_label = person_dir
            labels.append(person_label)
    
    return images, labels

def create_siamese_pairs(images, labels):
    pairs_anchor = []
    pairs_val = []
    target = []
    
    # Create positive pairs (same person)
    for i in range(len(images) - 41):
        for j in range(i+1, i + 41):
            if labels[i] == labels[j]:
                pairs_anchor.append((images[i]))
                pairs_val.append((images[j]))
                target.append(1)

    # Create negative pairs (different persons)
    for i in range(len(images) - 41):
        for j in range(i+1, i + 41):
            if labels[i] != labels[j]:
                pairs_anchor.append((images[i]))
                pairs_val.append((images[j]))
                target.append(0)
    
    return pairs_anchor, pairs_val, target

def load_data(data_dir):
    # Tạo bộ dữ liệu tổng hợp
    images, labels = load_lfw_dataset(data_dir)
    pairs_anchor, pairs_val, target = create_siamese_pairs(images, labels)

    # Shuffle the pairs and target arrays in the same order
    combined = list(zip(pairs_anchor, pairs_val, target))
    random.shuffle(combined)
    pairs_anchor[:], pairs_val[:], target[:] = zip(*combined)
    pairs_anchor = np.array(pairs_anchor)
    pairs_val = np.array(pairs_val)
    labels_dataset = np.array(target)
    
    return pairs_anchor, pairs_val, labels_dataset

def convert_to_grayscale(image_path):
    
    for person_dir in os.listdir(data_dir):
        # gray_path = 'C:/Users/NguyenBaThanh/OneDrive/Project I/Project/Gray_data'
        person_path = os.path.join(data_dir, person_dir)
        
        if not os.path.isdir(person_path):
            continue
        
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            # image_path = person_path +"/"+ '{}.jpg'.format(uuid.uuid1())
            
            # Mở ảnh gốc
            image = Image.open(image_path)
            
            # Chuyểnổi sang ảnh xám
            grayscale_image = image.convert("L")
            
            # Lưu ảnh x
            grayscale_image.save(image_path)
    
            print("Đã chuyển đổi ảnh", image_path)


# data_dir = 'C:/Users/NguyenBaThanh/Desktop/Test_data/data_img'
data_dir = 'C:/Users/NguyenBaThanh/OneDrive/Project I/Project/LFW_dataset/over_40_pic_in_folder'

# convert_to_grayscale(data_dir)

pairs_anchor, pairs_val, labels_dataset = load_data(data_dir)

# anchor_dataset = tf.convert_to_tensor(pairs_anchor)
# pairs_dataset = tf.convert_to_tensor(pairs_val)
# labels_dataset = tf.convert_to_tensor(labels_dataset)

# Tạo tập dữ liệu 2 lớp được gán nhãn
# data = tf.data.Dataset.zip((
#     tf.data.Dataset.from_tensor_slices(anchor_dataset),
#     tf.data.Dataset.from_tensor_slices(pairs_dataset),
#     tf.data.Dataset.from_tensor_slices(labels_dataset)
# ))

np.save('LFW_pairs_anchor_over_40_pic_in_folder.npy', pairs_anchor)
np.save('LFW_pairs_val_over_40_pic_in_folder.npy', pairs_val)
np.save('LFW_target_over_40_pic_in_folder.npy', labels_dataset)


