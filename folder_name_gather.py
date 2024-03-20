import os
import json
import re

def get_folder_names(directory_path):
    folder_names = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
    return folder_names

def write_to_json(folder_names, json_file):
    with open(json_file, 'w') as f:
        json.dump(folder_names, f)


def change_folder_names_in_html(html_path,directory_path):
    
    folder_names = get_folder_names(directory_path)
    
        # Read the original HTML file
    with open(html_path, 'r') as file:
        html_content = file.read()

    # Define the regular expression pattern to match var dates =
    pattern = r'var\s+dates\s*=\s*\[.*?\];'

    # Replace all occurrences of the pattern with the new dates array
    new_html_content = re.sub(pattern, f'var dates = {folder_names};', html_content)

    # Write the modified HTML content to a new HTML file
    with open(html_path, 'w') as file:
        file.write(new_html_content)
        
if __name__ == "__main__":
    directory_path = "/path/to/your/directory"  # Change this to your directory path
    json_file = "folder_names.json"  # JSON file to store folder names

    folder_names = get_folder_names(directory_path)
    write_to_json(folder_names, json_file)
    