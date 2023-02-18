from PIL import Image
import os
import shutil

def crop_gauge_from_frame(path, gauge_name, x1, y1, x2, y2):
    '''
    Crops a gauge from a video frame and saves the cropped image to a directory.
    '''
    # Define the parent directory and video name
    parent_dir, vid_name = os.path.split(path)

    # Define the path to the directory where the cropped images will be saved
    gauge_images_dir = os.path.join('gauge_images', gauge_name, vid_name)

    # Remove the directory if it already exists
    if os.path.exists(gauge_images_dir):
        shutil.rmtree(gauge_images_dir)
        print(f"Removed existing directory: {gauge_images_dir}")

    # Create a new directory to store the cropped images
    os.makedirs(gauge_images_dir)
    print(f"Created new directory: {gauge_images_dir}")

    # Iterate through the files in the directory and crop each image
    for filename in os.listdir(path):
        if filename == ".DS_Store":  # Ignore Mac OS metadata files
            continue
            
        # Load the image and crop it
        filepath = os.path.join(path, filename)
        with Image.open(filepath) as image:
            cropped_image = image.crop((x1, y1, x2, y2))

        # Save the cropped image to the output directory
        output_path = os.path.join(gauge_images_dir, filename)
        cropped_image.save(output_path)

# Define the coordinates of the airspeed gauge
x1 = 1111
y1 = 510
x2 = x1+80
y2 = y1+80

for i in os.listdir('cockpitview'):
    crop_gauge_from_frame(os.path.join('cockpitview', i), 'airspeed',
                          x1=x1, y1=y1, x2=x2, y2=y2)
