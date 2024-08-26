import os
from glob import glob
import shutil

def delete_non_bag_files(directory, action_ls):
    dirs =  sorted(os.listdir(directory))
    for dir in dirs:
        if dir[:7] == 'subject':
            dir_path = os.path.join(directory, dir)
            actions = os.listdir(dir_path)
            for action in actions:
                if action in action_ls:
                    action_path = os.path.join(dir_path, action)
                    views = os.listdir(action_path)
                    for view in views:
                        view_path = os.path.join(action_path, view)
                        if not view_path.split('.')[-1] == 'bag':
                            print(f"Deleting: {view_path}")
                            shutil.rmtree(view_path)
                    # Uncomment the line above after you're sure the script targets the correct files.

# Example usage:
directory = '/mnt/data/backups/MEH/'
actions = ['cut_paper', 'fold_paper', 'pour_water', 'read_book', 'staple_paper', 'write_with_pen', 'write_with_pencil', 'pinch_and_drag', 'pinch_and_hold', \
    'swipe', 'tap', 'touch']
delete_non_bag_files(directory, actions)
 