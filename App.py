import base64
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from tqdm.notebook import tqdm
from PIL import Image
import os
import io



st.set_page_config(page_title='Visual Captioning',layout="wide")

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

#Function to predict caption
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

#Function to generate caption
def generate_caption(image_file):
    image_id = os.path.splitext(image_file)[0]
    captions = mapping.get(image_id, [])
    st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
         /* Adjust the height as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    if not captions:
        st.markdown(f"<p class='center'>oopsss!!!! Cannot predict for this</p>", unsafe_allow_html=True)
        return ""
    else:        

        st.markdown("<h6 style='text-align: center;'>---------------------Actual--------------------- </h6>", unsafe_allow_html=True)
        
        for caption in captions:
            st.markdown(f"<p class='center'>{caption}</p>", unsafe_allow_html=True)
        y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
        st.markdown("<h6 style='text-align: center;'>---------------------Predicted--------------------- </h6>", unsafe_allow_html=True)
        return y_pred
    
    

    
def main():
    st.markdown('<h1 style="text-align: center;">Visual Caption Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #787878;">Aims to generate captions for the images 😊</p>', unsafe_allow_html=True)    
    st.write("These are some suggested images,, click on them to see the predicted captions.... ")
    st.write("Scroll down for the custom images:wink:")
    #Displaying suggested images
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
        if (i % 10) == 0:
            row = st.columns(10)
        with row[i % 10]:
            st.image(images[i], use_column_width=True)
            if st.button("Generate",key={image_names[i]}):
                a=images[i]
                b=image_names[i]
                placeholder = st.empty()
                placeholder.info("Please scroll down !!.")
    
                
    if a != None and b != None:
        desired_width = 500  # Change this to your preferred width
        image_resized = a.resize((desired_width, int(desired_width * a.height / a.width)))
        buffered = io.BytesIO()
        image_resized.save(buffered, format="JPEG")
        image_str = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpeg;base64,{image_str}" alt="Uploaded Image" style="max-width: {desired_width}px;">
        </div>
        """,
        unsafe_allow_html=True
        )

        k=generate_caption(b)
        st.markdown(f"<p class='center'>{k}</p>", unsafe_allow_html=True)
    
    st.write("<br>", unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)

    #Custom images
    st.markdown("## Choose an Image")
    uploaded_file = st.file_uploader("", type=["jpg"])
    st.write("Note: Custom images should be from FLICKR 8K dataset")
    

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        desired_width = 500  # Change this to your preferred width
        image_resized = image.resize((desired_width, int(desired_width * image.height / image.width)))
        buffered = io.BytesIO()
        image_resized.save(buffered, format="JPEG")
        image_str = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpeg;base64,{image_str}" alt="Uploaded Image" style="max-width: {desired_width}px;">
        </div>
        """,
        unsafe_allow_html=True
        )

        
    
        if st.button("Generate Caption"):
            caption = generate_caption(uploaded_file.name)
            st.markdown(f"<p class='center'>{caption}</p>", unsafe_allow_html=True)

    

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
