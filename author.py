import pandas as pd
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords

# Load author data 
try:
  author_df = pd.read_csv('author.csv')
except Exception as e:
  print(f"Error loading author CSV: {e}")
  exit()

# Match authors function  
def match_authors(blog_type, keyword, author_df):

  matches = []

  for index, row in author_df.iterrows():

    details = row['author details']
    
    # Fuzzy matching on blog type
    if fuzz.partial_ratio(blog_type, details) > 80:
      matches.append(row)  

    # Remove stopwords from details
    details_no_stopwords = [word for word in details.split() 
                             if word not in stopwords.words('english')]
    
    # Check if keyword is present
    if keyword in details_no_stopwords:
      matches.append(row)

  return pd.DataFrame(matches)
  
# Get author - try match first else random  
def get_author(blog_type, keyword, author_df):

  matches = match_authors(blog_type, keyword, author_df)

  if matches.empty:
    print("No matches, picking random author")
    author = author_df.sample(1)
  else:
    print("Found matching authors, picking one") 
    author = matches.sample(1)

  return author

if __name__ == '__main__':

  # Get author 
  author = get_author(blog_type, keyword, author_df)

  # Export picked author   
  try:
    author.to_csv('authorpicked.csv', index=False)
  except Exception as e:
    print(f"Error exporting author CSV: {e}")

  # Print confirmation
  print("Author selection completed successfully")