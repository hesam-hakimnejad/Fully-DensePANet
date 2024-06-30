from PIL import Image
import os

def concatenate_images(folder1, folder2, output_folder):
    # Get the list of image filenames in each folder
    images1 = os.listdir(folder1)
    images2 = os.listdir(folder2)

    # Iterate through images in the first folder
    for image1 in images1:
        # Check if the image exists in the second folder
        if image1 in images2:
            # Open images from both folders
            with Image.open(os.path.join(folder1, image1)) as img1, \
                 Image.open(os.path.join(folder2, image1)) as img2:
                # Concatenate images horizontally
                width = img1.width + img2.width
                height = max(img1.height, img2.height)
                new_img = Image.new('RGB', (width, height))
                new_img.paste(img1, (0, 0))
                new_img.paste(img2, (img1.width, 0))

                # Save the concatenated image
                output_path = os.path.join(output_folder, image1)
                new_img.save(output_path)
                print(f"Concatenated and saved: {output_path}")

# Example usage:
folder1 = 'data1024_1'
folder2 = 'data1024_2'
output_folder = 'train1024'
concatenate_images(folder1, folder2, output_folder)
