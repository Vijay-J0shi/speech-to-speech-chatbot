import streamlit as st
import whisper
from langchain_groq import ChatGroq
from gtts import gTTS
from pydub import AudioSegment
from audiorecorder import audiorecorder
import os
import tempfile


import base64

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

# Load ChatGroq model
@st.cache_resource
def load_chat_model():
    return ChatGroq(
        temperature=0,
        groq_api_key="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",  # Replace with your actual Groq API key
        model_name="llama-3.1-70b-versatile"
    )

chat_model = load_chat_model()

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # List to hold the conversation history

# Streamlit interface
st.title("Speech-to-Speech Chatbot")
st.write("Record your audio using the buttons below, and the chatbot will respond with speech!")

st.title("Audio Recorder")
audio_bytes = audiorecorder("Click to record", "Click to stop recording")


if audio_bytes:
    # Save the recorded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_bytes.export().read())
        input_audio_path = temp_audio_file.name

    # Convert audio to WAV format if necessary
    audio = AudioSegment.from_file(input_audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Ensure mono and 16kHz
    audio.export(input_audio_path, format="wav")

    st.audio(input_audio_path, format="audio/wav")

    # Transcribe audio with Whisper
    st.write("Transcribing audio...")
    result = model.transcribe(input_audio_path)
    transcription = result["text"]
    st.write(f"You: {transcription}")

    # Add user input to conversation history
    st.session_state.conversation.append(f"You: {transcription}")

    st.write("Generating chatbot response...")
    full_conversation = "\n".join(st.session_state.conversation)

    # Add prompt to limit the response to 25 words
    prompt_with_limit = full_conversation + "\nPlease respond in 25 words or fewer."
    response = chat_model.invoke(prompt_with_limit)
    chatbot_response = response.content

    st.write(f"Chatbot: {chatbot_response}")

    # Add chatbot response to conversation history
    st.session_state.conversation.append(f"Chatbot: {chatbot_response}")

    # Convert response to speech
    st.write("Converting response to speech...")
    speech = gTTS(text=chatbot_response, lang="en", slow=False, tld="com.au")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_response_audio_file:
        original_audio_path = temp_response_audio_file.name
        speech.save(original_audio_path)
        # Display audio player
    
    # Adjust playback speed using Pydub
    audio = AudioSegment.from_file(original_audio_path)
    faster_audio = audio.speedup(playback_speed=1.25)  # Speed up the audio by 1.25x
    audio_data = faster_audio.export(format="wav").read()
    st.audio(audio_data,autoplay=True)
    # Use the autoplay_audio function
    
    st.write("### Conversation History:")
    for line in st.session_state.conversation:
        st.write(line)

    # Cleanup temporary files
    if os.path.exists(input_audio_path):
        os.remove(input_audio_path)
    if os.path.exists(original_audio_path):
        os.remove(original_audio_path)