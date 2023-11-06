import csv

# Reading data from the CSV file
with open('author.csv', mode='r') as file:
    reader = csv.DictReader(file)
    data = [row for row in reader]

# Selecting one author (the first one in this case)
author = data[0]

# Creating the author's point of view
point_of_view = f"My name is {author['author name']}. I am {author['age']} years old, and I attended {author['school attended']}. Currently, I work as a {author['job']}. I live in {author['location']} and I am a {author['gender']}."

# Printing the author's point of view
print(point_of_view)
