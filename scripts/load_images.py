import os
import shutil

# Define the source and destination directories
source_dir = '~/cmu/kantor_lab/image_data2'
source_dir = os.path.expanduser(source_dir)
destination_dir = 'fb_images'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Iterate over each subdirectory in the source directory
for folder_name in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, folder_name)
    
    # Check if it is indeed a directory
    if os.path.isdir(subdir_path):
        # Construct the path to the image file within this subdirectory
        src_file = os.path.join(subdir_path, 'ximea.png')
        
        # Check if the image file exists
        if os.path.isfile(src_file):
            # Define the new destination path, renaming the file to the folder name
            dest_file = os.path.join(destination_dir, f"{folder_name}.png")
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")
        else:
            print(f"No ximea.png found in {subdir_path}")
