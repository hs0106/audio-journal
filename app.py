import streamlit as st
import numpy as np
import requests
import json
import os
from datetime import datetime
import tempfile
from pathlib import Path

# Get API key from environment variables or Streamlit secrets
def get_api_key():
    # Try to get from Streamlit secrets (for cloud deployment)
    if 'HF_API_KEY' in st.secrets:
        return st.secrets['HF_API_KEY']
    # Try to get from environment variables (for local development)
    elif 'HF_API_KEY' in os.environ:
        return os.environ['HF_API_KEY']
    # Fallback for development (NOT RECOMMENDED FOR PRODUCTION)
    else:
        st.warning("⚠️ No API key found. Features requiring API calls may not work.")
        return None

HF_API_KEY = get_api_key()
WHISPER_URL = "https://api-inference.huggingface.co/models/openai/whisper-tiny"
BART_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

# Where journal entries are stored - use a file for local dev, or session state for Streamlit Cloud
LOG_FILE = "journal_entries.json"

# Initialize session state for storing entries
if 'journal_entries' not in st.session_state:
    # Try to load from file first (for local development)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            st.session_state.journal_entries = json.load(f)
    else:
        st.session_state.journal_entries = []

# JSON utilities
def load_logs():
    return st.session_state.journal_entries

def save_entry(entry):
    # Add to session state
    st.session_state.journal_entries.append(entry)
    
    # Also try to save to file (for local development)
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(st.session_state.journal_entries, f, indent=2)
    except:
        # Streamlit Cloud might not allow writing to files
        pass

def summarize_text(text):
    if not HF_API_KEY:
        return "[Summary unavailable: API key missing]"
    
    response = requests.post(BART_URL, headers=HEADERS, json={"inputs": text})
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
            return result[0]["summary_text"]
    return "[Summary failed]"

def transcribe_audio(audio_file):
    """Transcribe audio file using Whisper API"""
    if not HF_API_KEY:
        return "[Transcription unavailable: API key missing]"
    
    with open(audio_file, "rb") as f:
        response = requests.post(WHISPER_URL, headers=HEADERS, data=f)
    
    if response.status_code == 200:
        result = response.json()
        if "text" in result:
            return result["text"]
    return None

# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'summary' not in st.session_state:
    st.session_state.summary = None

# --- Streamlit UI ---
st.set_page_config(page_title="Audio Journal", layout="centered")
st.title("🗣️📝 Audio & Text Journal")

tab1, tab2, tab3, tab4 = st.tabs(["🎤 Record Audio", "📁 Upload Audio", "✍️ Manual Entry", "📖 View Entries"])

# --- Tab 1: Record Audio ---
with tab1:
    st.subheader("🎤 Record your journal entry")
    
    # Status indicator
    status_indicator = st.empty()
    
    # Audio recorder
    st.write("Click the microphone button and start speaking")
    recorded_audio = st.audio_input("Record your journal entry")
    
    if recorded_audio is not None:
        # Display audio player for the recorded audio
        st.audio(recorded_audio)
        
        if st.button("🔄 Process Recording", key="process_recording"):
            # Save the recorded audio temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(recorded_audio)
                temp_filename = tmp_file.name
            
            status_indicator.info("🔄 Processing audio...")
            
            # Transcribe the audio
            transcript = transcribe_audio(temp_filename)
            
            # Clean up temp file
            try:
                os.unlink(temp_filename)
            except:
                pass
            
            if transcript:
                st.session_state.transcription = transcript
                
                st.subheader("📝 Transcription")
                st.write(transcript)
                
                status_indicator.info("🔄 Summarizing...")
                summary = summarize_text(transcript)
                st.session_state.summary = summary
                
                st.subheader("🧠 Summary")
                st.write(summary)
                
                # Save button
                if st.button("💾 Save Journal Entry", key="save_recording"):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_entry({
                        "timestamp": timestamp,
                        "transcript": transcript,
                        "summary": summary
                    })
                    status_indicator.success("✅ Journal entry saved!")
            else:
                status_indicator.error("❌ Transcription failed")

# --- Tab 2: Audio Upload ---
with tab2:
    st.subheader("📁 Upload your audio journal entry")
    
    # Status indicator
    upload_status_indicator = st.empty()
    
    # File uploader for audio
    st.write("Upload an audio file (.wav, .mp3, .m4a, .ogg, etc.)")
    uploaded_file = st.file_uploader("Choose an audio file", type=None, accept_multiple_files=False)
    
    if uploaded_file is not None:
        # Display audio player for the uploaded file
        st.audio(uploaded_file)
        
        # Process button
        if st.button("🔄 Process Audio", key="process_upload"):
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_filename = tmp_file.name
            
            upload_status_indicator.info("🔄 Processing audio...")
            
            # Transcribe the audio
            transcript = transcribe_audio(temp_filename)
            
            # Clean up temp file
            try:
                os.unlink(temp_filename)
            except:
                pass
            
            if transcript:
                st.session_state.transcription = transcript
                
                st.subheader("📝 Transcription")
                st.write(transcript)
                
                upload_status_indicator.info("🔄 Summarizing...")
                summary = summarize_text(transcript)
                st.session_state.summary = summary
                
                st.subheader("🧠 Summary")
                st.write(summary)
                
                # Save button
                if st.button("💾 Save Journal Entry", key="save_upload"):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_entry({
                        "timestamp": timestamp,
                        "transcript": transcript,
                        "summary": summary
                    })
                    upload_status_indicator.success("✅ Journal entry saved!")
            else:
                upload_status_indicator.error("❌ Transcription failed")
    
    # Audio recording guide
    with st.expander("How to record audio for your journal using external apps"):
        st.markdown("""
        ### How to record audio for your journal:
        
        1. **Use your smartphone:**
           - On iPhone: Use the Voice Memos app
           - On Android: Use the Voice Recorder app
           - Record your journal entry and save it
           - Transfer to your computer or email it to yourself
           
        2. **Use your computer:**
           - Windows: Use the built-in Voice Recorder app
           - Mac: Use the Voice Memos app
           - Save the recording and upload it here
           
        3. **Use an online recorder:**
           - Go to [Vocaroo](https://vocaroo.com)
           - Record your journal entry
           - Download the audio file and upload it here
           
        4. **Use a dedicated app:**
           - Many free recording apps are available for all platforms
           - Record in a quiet place for best results
           - Save in common formats like MP3 or WAV
        """)

# --- Tab 3: Manual Entry ---
with tab3:
    st.subheader("✍️ Write your journal entry")
    manual_input = st.text_area("Write your thoughts...", height=200)

    if st.button("🧠 Summarize & Save", key="save_manual"):
        if manual_input.strip() == "":
            st.warning("Please write something first.")
        else:
            manual_status_indicator = st.empty()
            manual_status_indicator.info("Summarizing...")
            summary = summarize_text(manual_input)
            st.subheader("🧠 Summary")
            st.write(summary)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_entry({"timestamp": timestamp, "transcript": manual_input, "summary": summary})
            manual_status_indicator.success("✅ Journal saved!")

# --- Tab 4: View Entries ---
with tab4:
    st.subheader("📖 Past Journal Entries")
    logs = load_logs()
    
    # Add export functionality
    if logs:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"You have {len(logs)} journal entries")
        with col2:
            export_data = json.dumps(logs, indent=2)
            st.download_button(
                label="📥 Export Entries",
                data=export_data,
                file_name="journal_entries.json",
                mime="application/json"
            )
            
        # Add search functionality
        search_term = st.text_input("🔍 Search your entries")
        if search_term:
            filtered_logs = [entry for entry in logs if search_term.lower() in entry['transcript'].lower()]
            if filtered_logs:
                st.write(f"Found {len(filtered_logs)} matching entries")
                for entry in reversed(filtered_logs):
                    with st.expander(f"{entry['timestamp']}"):
                        st.markdown(f"**📝 Transcript:**\n\n{entry['transcript']}")
                        st.markdown(f"**🧠 Summary:**\n\n{entry['summary']}")
            else:
                st.info("No matching entries found.")
        else:
            # Show all entries
            for entry in reversed(logs):
                with st.expander(f"{entry['timestamp']}"):
                    st.markdown(f"**📝 Transcript:**\n\n{entry['transcript']}")
                    st.markdown(f"**🧠 Summary:**\n\n{entry['summary']}")
    else:
        st.info("No journal entries yet. Start by recording or writing an entry!")