import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from tqdm.notebook import tqdm
from PIL import Image
import os

# Function to generate caption for the image

st.set_page_config(page_title='Visual Captioning')

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
    st.write("These are some suggested images,, click on them to see the predicted captions.... ")
    st.write("Scroll down for the custom images:wink:")
    images = []
    image_names = []
    folder_path=os.path.join(current_directory, 'Display')
    t=(200,200)
    for filename in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, filename))
        img=img.resize(t)
        images.append(img)
        image_names.append(filename)
    a=None
    b=None
    st.write("<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)
    for i in range(len(images)):
        if (i % 5) == 0:
            row = st.columns(5)
        with row[i % 5]:
            st.image(images[i], use_column_width=True)
            if st.button("Generate",key={image_names[i]}):
                a=images[i]
                b=image_names[i]
                placeholder = st.empty()
                placeholder.info("Please scroll down !!.")
                
    if a != None and b != None:
        st.image(a, caption="", use_column_width=True)
        k=generate_caption(b)
        st.write(k)
        
    st.write("<br>", unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    st.markdown("## Choose an Image")
    uploaded_file = st.file_uploader("", type=["jpg"])
    st.write("Note: Custom images should be from FLICKR 8K dataset")
    

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="", use_column_width=True)
        if st.button("Generate Caption"):
            caption = generate_caption(uploaded_file.name)
            st.write(caption)

    

    footer = """
        <hr>
        <style>
.footer {
    position: absolute;
    bottom: 2000;
    width: 100%;
    background-color: #f1f1f1;
    text-align: center;
    padding: 10px 0;
}
</style>
        <div style="text-align: center;">
         <h3>Developed by!!!!</h3>
         <p>Mohan Kalyan Guntupalli</p>
        </div>
                """
    st.markdown(footer, unsafe_allow_html=True)
    
    

if __name__ == "__main__":
    main()
