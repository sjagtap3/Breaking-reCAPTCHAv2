import os

# Define the folder path
folder_path = 'datasets/segment/valid'

# Output file to save the image paths
output_file = 'valid.txt'

# Open the file in write mode
with open(output_file, 'w') as file:
    # Loop through all files in the folder
    for i, filename in enumerate(os.listdir(folder_path)):
        # Ignore the non-image files
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue
        # Get the full path of the file
        file_path = os.path.join(folder_path, filename)
        # Write the file path to the output file
        file.write(file_path + '\n')    

print(f"File paths saved to {output_file}")