from PIL import Image
import csv
import os

def get_image_dimensions(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return width, height


def convert_labels_to_yolo(csv_file, output_dir, img_dir):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if present
        
        for row in reader:
            image_name, h, w, xmin, ymin, xmax, ymax = row
            
            # Calculate normalized bounding box coordinates
            file_path = os.path.join(img_dir, image_name)
            img_width, img_height = get_image_dimensions(file_path)
            x_center = (float(xmin) + float(xmax)) / 2 / img_width
            y_center = (float(ymin) + float(ymax)) / 2 / img_height
            width = (float(xmax) - float(xmin)) / img_width
            height = (float(ymax) - float(ymin)) / img_height
            
            # Write label in YOLO format
            label_content = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            
            # Save the label file
            image_name_without_ext = os.path.splitext(image_name)[0]
            label_file = os.path.join(output_dir, f"{image_name_without_ext}.txt")
            
            if os.path.exists(label_file):
                with open(label_file, 'a') as f:
                    f.write('\n' + label_content)
            else:
                with open(label_file, 'w') as f:
                    f.write(label_content)
                    

def delete_last_line(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Process only text files
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Remove the last line
            lines = lines[:-1]

            with open(file_path, 'w') as f:
                f.writelines(lines)