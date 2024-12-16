from PIL import Image
import os

def create_image_collage(folder_path, images_per_row):
    # Get all image files in the folder
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found in the folder.")
        return

    # Determine the dimensions of each image (based on the first image)
    img_width, img_height = Image.open(images[0]).size

    # Calculate the dimensions of the collage
    total_images = len(images)
    rows = (total_images + images_per_row - 1) // images_per_row
    collage_width = img_width * images_per_row
    collage_height = img_height * rows

    # Create a new blank collage image
    collage = Image.new('RGB', (collage_width, collage_height), 'white')

    # Place each image onto the collage
    for index, image in enumerate(images):
        img = Image.open(image)
        x = (index % images_per_row) * img_width
        y = (index // images_per_row) * img_height
        collage.paste(img, (x, y))

    # Save the collage
    collage.save(os.path.join(folder_path, 'collage_23.jpg'))
    print("Collage created and saved successfully.")

# Example usage
folder_path = r''  # Replace with the path to your image folder
images_per_row = 20  # Specify the number of images per row
create_image_collage(folder_path, images_per_row)
