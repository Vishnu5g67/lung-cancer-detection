import os
import cv2
import numpy as np

def generate_sample_features(num_samples=100):
    """Generates synthetic CT scan feature vectors and sample images for immediate testing."""
    os.makedirs("features", exist_ok=True)
    os.makedirs("testSamples", exist_ok=True)
    
    np.random.seed(42)
    X = np.random.rand(num_samples, 64, 64, 3).astype('float32')
    # Normal (0) and Abnormal (1) labels
    Y = np.array([0 if i < num_samples // 2 else 1 for i in range(num_samples)])
    
    # Introduce distinctive pattern for abnormal class
    for i in range(num_samples // 2, num_samples):
        X[i, 20:44, 20:44, :] += 0.5
        X[i] = np.clip(X[i], 0.0, 1.0)
        
    np.save("features/X.txt.npy", X)
    np.save("features/Y.txt.npy", Y)
    print("Features saved: features/X.txt.npy and features/Y.txt.npy (Shape: {})".format(X.shape))
    
    # Generate 2 sample test images
    normal_img = (np.random.rand(400, 400, 3) * 120 + 30).astype(np.uint8)
    cv2.circle(normal_img, (200, 200), 120, (180, 180, 180), -1)
    cv2.imwrite("testSamples/normal_sample.png", normal_img)
    
    abnormal_img = (np.random.rand(400, 400, 3) * 120 + 30).astype(np.uint8)
    cv2.circle(abnormal_img, (200, 200), 120, (180, 180, 180), -1)
    cv2.circle(abnormal_img, (230, 170), 35, (255, 255, 255), -1)  # simulated nodule/lesion
    cv2.imwrite("testSamples/abnormal_sample.png", abnormal_img)
    print("Sample test CT images generated in testSamples/ directory.")

def process_image_dataset(dataset_dir):
    """
    Processes a directory containing Normal/ and Abnormal/ folders of CT scan images.
    """
    os.makedirs("features", exist_ok=True)
    categories = ['Normal', 'Abnormal']
    data = []
    labels = []
    
    for label_idx, cat in enumerate(categories):
        cat_path = os.path.join(dataset_dir, cat)
        if not os.path.exists(cat_path):
            print(f"Folder {cat_path} not found.")
            continue
        for img_name in os.listdir(cat_path):
            img_path = os.path.join(cat_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                img = img.astype('float32') / 255.0
                data.append(img)
                labels.append(label_idx)
                
    if len(data) > 0:
        X = np.array(data)
        Y = np.array(labels)
        np.save("features/X.txt.npy", X)
        np.save("features/Y.txt.npy", Y)
        print(f"Processed {len(data)} images successfully.")
    else:
        print("No images found in dataset folder. Generating sample features instead...")
        generate_sample_features()

if __name__ == '__main__':
    generate_sample_features()
