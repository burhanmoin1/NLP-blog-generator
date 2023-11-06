import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Reading the CSV file
data = pd.read_csv('blog_titles.csv')

# Filling missing values with an empty string
data['Blogs'].fillna('', inplace=True)

X = data['Title']
y = data['Blogs']

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # Remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Concatenate the title and blog content for each entry
data['Combined_Text'] = data['Title'] + ' ' + data['Blogs']

# Extract the combined text
combined_text = data['Combined_Text'].tolist()

# Preprocess the combined text
preprocessed_combined_text = [preprocess_text(text) for text in combined_text]


# Tokenize the preprocessed combined text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_combined_text)
total_words = len(tokenizer.word_index) + 1

# Create input sequences using the tokenized data
input_sequences = []
for line in preprocessed_combined_text:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = 392
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Create predictors and label
import numpy as np
input_sequences = np.array(input_sequences)
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

# Convert the label to a categorical format
label = tf.keras.utils.to_categorical(label, num_classes=total_words)

# Build the RNN model
model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(predictors, label, epochs=5, verbose=1)

# Save the trained model
model.save('blog_generating_model.h5')

model = load_model('blog_generating_model.h5')

# Seed text for the blog
seed_text = "Forrest Gump"
def generate_text(model, tokenizer, max_sequence_len, seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Example usage of the generate_text function
generated_blog = generate_text(model, tokenizer, max_sequence_len+1, seed_text, 100)
print(generated_blog)
