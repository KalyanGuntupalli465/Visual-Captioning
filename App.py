import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from tqdm.notebook import tqdm
from PIL import Image
import os

# Function to generate caption for the image


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

current_directory = os.path.dirname(os.path.abspath(__file__))
features = os.path.join(current_directory, 'features.pkl')
with open(features, 'rb') as f:
    features = pickle.load(f)

# Load the tokenizer from a saved file
tokenizer_path = os.path.join(current_directory, 'tokenizer1.pkl') # Replace with the actual path
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Load the mapping of image to captions
caption_path = os.path.join(current_directory, 'captions.txt')
with open(caption_path, 'r') as f:
    next(f)
    captions_doc = f.read()

mapping = {}
for line in tqdm(captions_doc.split('\n')):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

# Load the trained captioning model
model = load_model(os.path.join(current_directory, 'captioning_model.h5'))
max_length = 35

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'st'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'end':
            break
    return in_text


def generate_caption(image_file):
    image_id = os.path.splitext(image_file)[0]
    captions = mapping.get(image_id, [])
    
    if not captions:
        st.write("oopsss!!!! Cannot predict for this")

    else:
        st.write('---------------------Actual---------------------')
        for caption in captions:
            st.write(caption)
        y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
        st.write('--------------------Predicted--------------------')
        return y_pred

def main():
    st.title("Visual Caption Generator")

    # File uploader widget to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])
    

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        st.image(image, caption="", use_column_width=True)

        # Button to generate caption for the uploaded image
        if st.button("Generate Caption"):
            # Call function to generate caption
            caption = generate_caption(uploaded_file.name)
            # Display the generated caption
            st.write(caption)

if __name__ == "__main__":
    main()
