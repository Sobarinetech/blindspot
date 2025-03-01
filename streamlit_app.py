import streamlit as st
import google.generativeai as genai
import requests
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import langid
from collections import Counter
from gtts import gTTS
import os

# Download necessary corpora for NLTK
nltk.download('vader_lexicon')

# Configure the API keys securely using Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# App Title and Description
st.title("AI-Powered Ghostwriter & Audio Transcription")
st.write("Generate creative content and ensure its originality, step by step.")
st.write("Additionally, transcribe and analyze audio files for insights.")

# Add custom CSS to hide the header and the top-right buttons
hide_streamlit_style = """
    <style>
        .css-1r6p8d1 {display: none;} /* Hides the Streamlit logo in the top left */
        .css-1v3t3fg {display: none;} /* Hides the star button */
        .css-1r6p8d1 .st-ae {display: none;} /* Hides the Streamlit logo */
        header {visibility: hidden;} /* Hides the header */
        .css-1tqja98 {visibility: hidden;} /* Hides the header bar */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Step 1: Input Prompt
st.subheader("Step 1: Enter your content idea")
prompt = st.text_input("What would you like to write about?", placeholder="e.g. AI trends in 2025")

# Function to handle web search
def search_web(query):
    """Searches the web using Google Custom Search API and returns results."""
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": st.secrets["GOOGLE_API_KEY"],
        "cx": st.secrets["GOOGLE_SEARCH_ENGINE_ID"],
        "q": query,
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        return response.json().get("items", [])
    else:
        st.error(f"Search API Error: {response.status_code} - {response.text}")
        return []

# Function to regenerate content for originality
def regenerate_content(original_content):
    """Generates rewritten content based on the original content to ensure originality."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Rewrite the following content to make it original and distinct:\n\n{original_content}"
    response = model.generate_content(prompt)
    return response.text.strip()

# Function to transcribe audio
def transcribe_audio(file):
    """Transcribes the audio using the Hugging Face API."""
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
    API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
    HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
    try:
        data = file.read()
        response = requests.post(API_URL, headers=HEADERS, data=data)
        if response.status_code == 200:
            return response.json()  # Return transcription
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# Enhanced sentiment analysis with VADER
def analyze_vader_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# Function for keyword extraction using CountVectorizer
def extract_keywords(text):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(stop_words='english', max_features=10)  # Extract top 10 frequent words
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords

# Function to calculate word frequency
def word_frequency(text):
    words = text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(20)
    return most_common_words

# Function to generate a word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# Detect language of the text
def detect_language(text):
    lang, confidence = langid.classify(text)
    return lang, confidence

# Function to convert text to speech using gTTS
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    return "output.mp3"

# Step 2: Content Generation and Search for Similarity
if prompt.strip():
    if st.button("Generate Content"):
        with st.spinner("Generating content... Please wait!"):
            try:
                # Generate content using Generative AI
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                generated_text = response.text.strip()

                # Display the generated content with feedback
                st.subheader("Step 2: Your Generated Content")
                st.write(generated_text)

                # TTS for generated content
                audio_file = text_to_speech(generated_text)
                st.audio(audio_file, format="audio/mp3")

                # Check for similar content online (Step 3)
                st.subheader("Step 3: Searching for Similar Content Online")
                search_results = search_web(generated_text)

                if search_results:
                    st.warning("We found similar content on the web:")

                    # Display results in a compact, user-friendly format
                    for result in search_results[:3]:  # Show only the top 3 results
                        with st.expander(result['title']):
                            st.write(f"**Source:** [{result['link']}]({result['link']})")
                            st.write(f"**Snippet:** {result['snippet'][:150]}...")  # Shortened snippet
                            st.write("---")

                    # Option to regenerate content for originality
                    regenerate_button = st.button("Regenerate Content for Originality")
                    if regenerate_button:
                        with st.spinner("Regenerating content..."):
                            regenerated_text = regenerate_content(generated_text)
                            st.session_state.generated_text = regenerated_text
                            st.success("Content successfully regenerated for originality.")
                            st.subheader("Regenerated Content:")
                            st.write(regenerated_text)

                            # TTS for regenerated content
                            audio_file = text_to_speech(regenerated_text)
                            st.audio(audio_file, format="audio/mp3")
                else:
                    st.success("Your content appears to be original. No similar content found online.")

            except Exception as e:
                st.error(f"Error generating content: {e}")
else:
    st.info("Enter your idea in the text box above to start.")

# Option to clear the input and reset the app
if st.button("Clear Input"):
    st.session_state.generated_text = ""
    st.experimental_rerun()  # Reset the app state

# Audio Transcription and Analysis Section
st.subheader("Audio Transcription & Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file (e.g., .wav, .flac, .mp3)", type=["wav", "flac", "mp3"])

if uploaded_file is not None:
    # Display uploaded audio
    st.audio(uploaded_file, format="audio/mp3", start_time=0)
    st.info("Transcribing audio... Please wait.")
    
    # Transcribe the uploaded audio file
    result = transcribe_audio(uploaded_file)
    
    # Display the result
    if "text" in result:
        st.success("Transcription Complete:")
        transcription_text = result["text"]
        st.write(transcription_text)
        
        # TTS for transcription
        audio_file = text_to_speech(transcription_text)
        st.audio(audio_file, format="audio/mp3")

        # Sentiment Analysis (VADER)
        vader_sentiment = analyze_vader_sentiment(transcription_text)
        st.subheader("Sentiment Analysis (VADER)")
        st.write(f"Positive: {vader_sentiment['pos']}, Neutral: {vader_sentiment['neu']}, Negative: {vader_sentiment['neg']}")

        # Language Detection
        lang, confidence = detect_language(transcription_text)
        st.subheader("Language Detection")
        st.write(f"Detected Language: {lang}, Confidence: {confidence}")

        # Keyword Extraction
        keywords = extract_keywords(transcription_text)
        st.subheader("Keyword Extraction")
        st.write(keywords)

        # Word Frequency Analysis
        word_freq = word_frequency(transcription_text)
        st.subheader("Word Frequency Analysis")
        st.write(word_freq)

        # Word Cloud Visualization
        wordcloud = generate_word_cloud(transcription_text)
        st.subheader("Word Cloud")
        st.image(wordcloud.to_array())

        # Add download button for the transcription text
        st.download_button(
            label="Download Transcription",
            data=transcription_text,
            file_name="transcription.txt",
            mime="text/plain"
        )
        
        # Add download button for analysis results
        analysis_results = f"""
        Sentiment Analysis (VADER):
        Positive: {vader_sentiment['pos']}, Neutral: {vader_sentiment['neu']}, Negative: {vader_sentiment['neg']}
        
        Language Detection:
        Detected Language: {lang}, Confidence: {confidence}
        
        Keyword Extraction:
        {keywords}
        """
        st.download_button(
            label="Download Analysis Results",
            data=analysis_results,
            file_name="analysis_results.txt",
            mime="text/plain"
        )

    elif "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.warning("Unexpected response from the API.")
