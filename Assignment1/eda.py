import os
import matplotlib.pyplot as plt
from PIL import Image



def eda(data_dir, name):
    # List all subdirectories (classes) in the dataset directory
    classes = os.listdir(data_dir)

    # Display the number of classes
    print("Number of classes:", len(classes))

    # Display the number of images in each class
    total = 0
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        num_images = len(os.listdir(cls_dir))
        print(f"Class: {cls}, Number of images: {num_images}")
        total += num_images
    print("Total number of images: ", total)

    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(classes, [len(os.listdir(os.path.join(data_dir, cls))) for cls in classes])
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.savefig(name+"_class_distribution.png")

    # Display sample images from each class
    plt.figure(figsize=(12, 8))
    for i, cls in enumerate(classes, 1):
        cls_dir = os.path.join(data_dir, cls)
        img_file = os.listdir(cls_dir)[0]  # Display the first image from each class
        img_path = os.path.join(cls_dir, img_file)
        img = Image.open(img_path)
        plt.subplot(2, 3, i)
        plt.imshow(img)
        plt.title(cls)
        plt.axis('off')
    plt.savefig(name+"_sample_images.png")


    # Calculate and visualize image sizes
    image_sizes = []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for img_file in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_file)
            img = Image.open(img_path)
            width, height = img.size
            image_sizes.append((width, height))

    widths, heights = zip(*image_sizes)
    plt.figure(figsize=(10, 6))
    plt.hist(widths, bins=20, alpha=0.5, label='Width')
    plt.hist(heights, bins=20, alpha=0.5, label='Height')
    plt.xlabel('Image Size')
    plt.ylabel('Frequency')
    plt.title('Image Size Distribution')
    plt.legend()
    plt.savefig(name + "_image_size_distribution.png")




train_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Intel_Image_classification", "seg_train", "seg_train")
eda(train_dir, "train")

test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Intel_Image_classification", "seg_test", "seg_test")
eda(test_dir, "test")

