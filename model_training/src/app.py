# model_training/src/app.py (text to sign)


import streamlit as st
from inference import build_video_map, text_to_sign_video

st.set_page_config(page_title="ISL Translator", page_icon="🧏‍♀️")
st.title("🧏‍♀️ Indian Sign Language Translator (Text → Sign)")
st.write("Type any English sentence to see its sign language translation below.")

# Load video map once
video_map = build_video_map()

user_input = st.text_input("Enter text:")

if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter some text to translate!")
    else:
        st.success("✅ Translating...")
        clip = text_to_sign_video(user_input, video_map)

        if clip:
            output_path = "output.mp4"
            clip.write_videofile(output_path, codec="libx264", fps=24, logger=None)
            st.video(output_path)
        else:
            st.warning("⚠️ No matching signs found.")
