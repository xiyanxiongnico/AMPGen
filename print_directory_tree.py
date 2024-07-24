import os

def print_directory_tree(path, prefix=""):
    # Get a list of all the items in the directory
    items = os.listdir(path)
    
    # Sort items to list directories first, then files
    items.sort(key=lambda x: (os.path.isfile(os.path.join(path, x)), x.lower()))

    # Loop through all the items
    for index, item in enumerate(items):
        # Determine if the item is the last one in the directory
        is_last = index == len(items) - 1
        
        # Print the item with the appropriate prefix
        print(prefix + ("└── " if is_last else "├── ") + item)
        
        # If the item is a directory, recursively print its contents
        if os.path.isdir(os.path.join(path, item)):
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_directory_tree(os.path.join(path, item), new_prefix)

# Specify the path of the directory
project_path = "/Users/nicholexiong/Downloads/AMPGen/generated&scripts"

# Print the directory structure
print_directory_tree(project_path)