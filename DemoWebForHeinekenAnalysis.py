import easyocr
import cv2
from matplotlib import pyplot as plt
from transformers import pipeline
from PIL import Image
import numpy as np
import os
from groq import Groq
import requests
import json
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import glob
import torch
from pymongo import MongoClient
import base64
from io import BytesIO
import urllib

# MongoDB Atlas connection
client = MongoClient("mongodb+srv://cakoipro123456:khuong1182004@cluster0.auqviui.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['image_analysis_db']
history_collection = db['analysis_history']

# Function to load image icons from URL
def load_image_url(url):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        return None
    return Image.open(response.raw)

# Load image icons
upload_icon = load_image_url("https://example.com/path_to_upload_icon.png")
processing_icon = load_image_url("https://example.com/path_to_processing_icon.png")
analysis_icon = load_image_url("https://example.com/path_to_analysis_icon.png")

# Initialize the OCR reader
ocr_reader = easyocr.Reader(["en", "vi"])

# Functions for image processing
def get_image_caption(image):
    caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    return caption_pipeline(image)[0]['generated_text']

def perform_ocr(image):
    result = ocr_reader.readtext(np.array(image))
    return result

def analyze_image_information(image_description, ocr_results):
    prompt = f"""
    Analyze the following image information and provide insights based on the criteria given below:

    Image Description:
    {image_description}

    OCR Results:
    {ocr_results}

    Criteria:
    1. Brand Logos: Identify any brand logos mentioned in the description or OCR results:  Heineken, Tiger, Bia Việt, Larue, Bivina, Edelweiss and Strongbow and other competitor or non-competitor brands
    2. Products: Mention any products such as beer kegs and bottles.
    3. Customers: Describe the number of customers, their activities, and emotions.
    4. Promotional Materials: Identify any posters, banners, and billboards.
    5. Setup Context: Determine the scene context (e.g., bar, restaurant, grocery store, or supermarket).
    6. Evaluate the success of the event.
    7. Follow up with marketing staff.
    8. Evaluate the level of presence in the store.

    Insights:
    """
    client = Groq(api_key="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")
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    chat_completion = client.chat.completions.create(**data)
    return chat_completion.choices[0].message.content

def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return image
    cap.release()
    return None

# Functions for video processing
def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f"frame{count:04d}.jpg"), frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_video_description(frames_folder):
    frame_paths = sorted(glob.glob(os.path.join(frames_folder, '*.jpg')))
    descriptions = []
    for frame_path in frame_paths:
        caption = generate_caption(frame_path)
        descriptions.append(caption)
    detailed_description = " ".join(descriptions)
    return detailed_description

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

def summarize_text(text, max_length=130, min_length=30):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    return summary

def extract_text_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    extracted_text = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = ocr_reader.readtext(frame)
        for (bbox, text, prob) in result:
            extracted_text.append(text)
    cap.release()
    cv2.destroyAllWindows()
    return extracted_text

def analyze_video_information(video_description, video_ocr_results):
    prompt = f"""
    Analyze the following image information and provide insights based on the criteria given below:

    Image Description:
    {image_description}

    OCR Results:
    {ocr_results}

    Criteria:
    1. Brand Logos: Identify any brand logos mentioned in the description or OCR results:  Heineken, Tiger, Bia Việt, Larue, Bivina, Edelweiss and Strongbow.
    2. Products: Mention any products such as beer kegs and bottles.
    3. Customers: Describe the number of customers, their activities, and emotions.
    4. Promotional Materials: Identify any posters, banners, and billboards.
    5. Setup Context: Determine the scene context (e.g., bar, restaurant, grocery store, or supermarket).
    6. Evaluate the success of the event.
    7. Follow up with marketing staff.
    8. Evaluate the level of presence in the store.

    Insights:
    """
    client = Groq(api_key="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")
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    chat_completion = client.chat.completions.create(**data)
    return chat_completion.choices[0].message.content

# Streamlit app
st.set_page_config(layout="wide", page_icon="📷", page_title="Image and Video Analysis App")
st.markdown(
    """
    <style>
    .main {
        background-color: black;
    }
    .stButton button {
        border-radius: 8px;
    }
    .stHeader, .stSubheader, .stMarkdown {
        color: white;
        background-color: black;
    }
    .title {
        color: #FFA500;
        font-size: 24px;
        font-weight: bold;
    }
    .description {
        color: #00FF00;
        font-size: 18px;
    }
    .analysis {
        color: #1E90FF;
        font-size: 18px;
    }
    .delete-button {
        background-color: #FF4500;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #2E2E2E;
    }
    .sidebar .sidebar-content .nav-item {
        color: #FFFFFF;
        font-size: 20px;
    }
    .sidebar .sidebar-content .nav-item:hover {
        background-color: #1E1E1E;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #2E2E2E;
    }
    .sidebar .sidebar-content .nav-item {
        color: #FFFFFF;
        font-size: 20px;
        padding: 10px;
        border-radius: 5px;
    }
    .sidebar .sidebar-content .nav-item:hover {
        background-color: #1E1E1E;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Home")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Main", "History"], format_func=lambda x: "🏠 Main" if x == "Main" else "📜 History")

if app_mode == "Main":
    st.markdown("<div class='title'>📷 Image and Video Analysis App</div>", unsafe_allow_html=True)
    st.markdown("<div class='description'>Analyze images and videos to get detailed insights using AI models.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        st.header("📤 Upload Image or Video")
        if upload_icon:
            st.image(upload_icon, width=150)
        uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]

        if file_type == 'image':
            with col2:
                st.header("🔍 Description")
                image = Image.open(uploaded_file).convert("RGB")

                if processing_icon:
                    st.image(processing_icon, width=100)

                st.subheader("Image Description")
                image_description = get_image_caption(image)
                st.markdown(f"<div class='description'>{image_description}</div>", unsafe_allow_html=True)

                ocr_result = perform_ocr(image)

                image_np = np.array(image)
                boxes = [line[0] for line in ocr_result]
                texts = [line[1] for line in ocr_result]

                for box, text in zip(boxes, texts):
                    top_left = (int(box[0][0]), int(box[0][1]))
                    bottom_right = (int(box[2][0]), int(box[2][1]))
                    cv2.rectangle(image_np, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(image_np, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                st.image(image_np, caption='Uploaded Image', use_column_width=True)

            with col3:
                st.header("📊 Analysis")
                if analysis_icon:
                    st.image(analysis_icon, width=100)

                ocr_results = ' '.join([line[1] for line in ocr_result])
                analysis = analyze_image_information(image_description, ocr_results)
                st.markdown(f"<div class='analysis'>{analysis}</div>", unsafe_allow_html=True)

                # Save to history
                img_base64 = convert_image_to_base64(image)
                history_collection.insert_one({
                    "type": "image",
                    "description": image_description,
                    "ocr": ocr_results,
                    "analysis": analysis,
                    "image_base64": img_base64
                })

        elif file_type == 'video':
            video_path = os.path.join(".", uploaded_file.name)
            if not os.path.exists("temp"):
                os.makedirs("temp")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with col2:
                st.header("🔍 Description")
                st.video(video_path, format="video/mp4")

                if processing_icon:
                    st.image(processing_icon, width=100)

                st.subheader("Video Description")

                first_frame = extract_first_frame(video_path)
                if first_frame:
                    st.image(first_frame, caption='First Frame of Video', use_column_width=True)
                    image_description = get_image_caption(first_frame)
                    st.markdown(f"<div class='description'>{image_description}</div>", unsafe_allow_html=True)

                frames_folder = 'frames'
                extract_frames(video_path, frames_folder)

                st.write("Generating video description...")
                detailed_description = generate_video_description(frames_folder)

                st.write("Summarizing video description...")
                summary = summarize_text(detailed_description, max_length=150, min_length=50)
                st.write(summary)

                video_ocr_texts = extract_text_from_video(video_path)

            with col3:
                st.header("📊 Analysis")
                if analysis_icon:
                    st.image(analysis_icon, width=100)

                video_ocr_results = ' '.join(video_ocr_texts)
                analysis = analyze_video_information(summary, video_ocr_results)
                st.markdown(f"<div class='analysis'>{analysis}</div>", unsafe_allow_html=True)

                # Save to history
                img_base64 = convert_image_to_base64(first_frame)
                history_collection.insert_one({
                    "type": "video",
                    "description": summary,
                    "ocr": video_ocr_texts,
                    "analysis": analysis,
                    "image_base64": img_base64
                })

elif app_mode == "History":
    st.title("📜 Analysis History")
    st.markdown("### View the history of analyzed images and videos.")

    # Fetch history from MongoDB
    history_items = list(history_collection.find().sort("_id", -1))

    for item in history_items:
        st.markdown(f"<div class='title'>Type: {item['type'].capitalize()}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='description'>Image/Video Description:</div>", unsafe_allow_html=True)
        st.write(item['description'])
        st.markdown(f"<div class='analysis'>Analysis:</div>", unsafe_allow_html=True)
        st.write(item['analysis'])

        if item['type'] == 'image' or item['type'] == 'video':
            image_data = base64.b64decode(item['image_base64'])
            image = Image.open(BytesIO(image_data))
            st.image(image, caption='Analyzed Image/Video Frame')

        st.write("---")
