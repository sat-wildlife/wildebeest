import os

folder_path = r'data\images'  # Folder where image files are stored
train_file_path = r'train.txt'  # Path to save the list of all image file paths

# Initialize a list to store all file paths
all_file_paths = []

# Walk through the directory and its subdirectories to collect all file paths
for root, dirs, files in os.walk(folder_path):
    for name in files:
        # Construct the full file path and append its absolute path to the list
        file_path = os.path.join(root, name)
        all_file_paths.append(os.path.abspath(file_path))

# Write the collected file paths to train.txt
with open(train_file_path, 'w') as train_file:
    for path in all_file_paths:
        # Write each file path on a new line
        train_file.write(path + '\n')
