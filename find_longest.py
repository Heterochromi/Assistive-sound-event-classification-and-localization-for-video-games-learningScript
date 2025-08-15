import os
from PIL import Image
def find_longest_width(folder_path="mel_spectrogrames"):
    """
    Find the longest width among all images in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing images. Defaults to "mel_spectrogrames".
    
    Returns:
        tuple: (max_width, filename) - The maximum width found and the filename that has it.
               Returns (0, None) if no valid images are found.
    """
    max_width = 0
    max_width_file = None
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return 0, None
    
    # Get all files in the folder
    files = os.listdir(folder_path)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    
    processed_count = 0
    
    for filename in files:
        # Check if file has an image extension
        _, ext = os.path.splitext(filename.lower())
        if ext in image_extensions:
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    processed_count += 1
                    
                    if width > max_width:
                        max_width = width
                        max_width_file = filename
                        
                    print(f"Processed: {filename} - Width: {width}px, Height: {height}px")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"\nTotal images processed: {processed_count}")
    if max_width_file:
        print(f"Longest width found: {max_width}px in file '{max_width_file}'")
    else:
        print("No valid images found.")
    
    return max_width, max_width_file

# Example usage
if __name__ == "__main__":
    max_width, filename = find_longest_width()
    
    if filename:
        print(f"\nResult: The image with the longest width is '{filename}' with {max_width} pixels.")
    else:
        print("\nNo images found or processed.")