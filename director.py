
import os
import subprocess

# Try to delete output.csv
# If the file does not exist, ignore the error
try:
    os.remove("output.csv")
except FileNotFoundError:
    pass

# Try to delete authorpicked.csv, blog_type_picked.csv, and fine_pick.csv
# If any of the files do not exist, ignore the error
try:
    os.remove("authorpicked.csv")
except FileNotFoundError:
    pass

try:
    os.remove("blog_type_picked.csv")
except FileNotFoundError:
    pass

try:
    os.remove("fine_pick.csv")
except FileNotFoundError:
    pass

# Run blog_type.py
try:
    subprocess.run(["python", "blog_type.py"])
except Exception as e:
    print(f"Error running blog_type.py: {e}")

# Run author.py
try:
    subprocess.run(["python", "author.py"])
except Exception as e:
    print(f"Error running author.py: {e}")

# Run fine.py
try:
    subprocess.run(["python", "fine.py"])
except Exception as e:
    print(f"Error running fine.py: {e}")

# For every record in input.csv, run steps 1 and 2

with open("input.csv", "r") as input_file:
    for line in input_file:
        # Try to delete output.csv
        # If the file does not exist, ignore the error
        try:
            os.remove("output.csv")
        except FileNotFoundError:
            pass

        # Try to delete authorpicked.csv, blog_type_picked.csv, and fine_pick.csv
        # If any of the files do not exist, ignore the error
        try:
            os.remove("authorpicked.csv")
        except FileNotFoundError:
            pass

        try:
            os.remove("blog_type_picked.csv")
        except FileNotFoundError:
            pass

        try:
            os.remove("fine_pick.csv")
        except FileNotFoundError:
            pass