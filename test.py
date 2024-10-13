from pathlib import Path

def convert_path(filepath):
    # Convert the Path object or string path to a string
    filepath_str = str(filepath)
    
    # Replace backslashes with forward slashes
    filepath_str = filepath_str.replace('\\', '/')
    
    # Return the modified string path (can use Path(filepath_str) if needed)
    return filepath_str

# Example usage
original_path = r'C:\Users\abhis\OneDrive\Desktop\Abhishek\Computer Vision\HW1\code\aquarium\sun_asgtepdmsxsrqqvy.jpg'
new_path = convert_path(original_path)

print(f"Original path: {original_path}")
print(f"Converted path: {new_path}")
