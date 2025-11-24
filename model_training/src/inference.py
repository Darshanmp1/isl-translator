import os
import re
from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips, ColorClip

import streamlit as st

# ===============================
# 🔧 CONFIGURATION
# ===============================
VIDEO_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_videos"

# ===============================
# 🧠 HELPER FUNCTIONS
# ===============================
@st.cache_resource
def build_video_map():
    """Map each word/letter (without .mp4) to its video path."""
    if not VIDEO_DIR.exists():
        os.makedirs(VIDEO_DIR, exist_ok=True)
        st.warning(f"⚠️ Created empty video folder: {VIDEO_DIR}")
        return {}

    video_map = {
        os.path.splitext(f)[0].lower(): str(VIDEO_DIR / f)
        for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(".mp4")
    }
    # No info message — keep UI clean
    return video_map


def clean_text(text: str) -> str:
    """Remove special characters and lowercase text."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).lower().strip()


def add_pause(duration=0.4, size=(640, 480), color=(255, 255, 255)):
    """Return a blank clip used as a short pause between signs."""
    return ColorClip(size=size, color=color, duration=duration)


# ===============================
# 🎥 MAIN FUNCTION
# ===============================
def text_to_sign_video(text: str, video_map: dict):
    """
    Convert text into a concatenated ISL video:
    • Plays full-word videos when available
    • Otherwise spells each letter
    • Adds a short pause between words
    """
    words = clean_text(text).split()
    clips = []

    for word in words:
        if word in video_map:
            clips.append(VideoFileClip(video_map[word]))
        else:
            for char in word:
                if char.isalnum() and char.lower() in video_map:
                    clips.append(VideoFileClip(video_map[char.lower()]))

        # Pause after each word
        clips.append(add_pause(0.4))

    if not clips:
        st.warning("⚠️ No matching sign videos found for your input.")
        return None

    try:
        return concatenate_videoclips(clips, method="compose")
    except Exception as e:
        st.error(f"❌ Error generating video: {e}")
        return None
