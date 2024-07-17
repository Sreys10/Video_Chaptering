import streamlit as st
import re
import csv
import pandas as pd
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

API_KEY = 'AIzaSyBp5lDIb7ufnqH7kdlJ_GqPugqrSZfUwrM'  # Replace with your actual YouTube API key

# Function definitions for YouTube API and transcript handling
def get_video_id(url):
    # Function logic to extract video ID from YouTube URL
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id_match.group(1) if video_id_match else None

def get_video_title(video_id):
    # Function logic to fetch video title from YouTube
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    request = youtube.videos().list(part='snippet', id=video_id)
    response = request.execute()
    title = response['items'][0]['snippet']['title'] if response['items'] else 'Unknown Title'
    return title

# Function to fetch video transcript from YouTube
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

# Function to save transcript to CSV file
def save_to_csv(title, transcript, filename):
    transcript_data = [{'start': entry['start'], 'text': entry['text']} for entry in transcript]
    df = pd.DataFrame(transcript_data)
    df.to_csv(filename, index=False)
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title:', title])

def model(filename):
    try:
        transcript_df = pd.read_csv(filename)
    except FileNotFoundError:
        st.error(f"File '{filename}' not found. Please check the file path.")
        return

    # Clean the 'tart' column to remove any non-numeric entries
    transcript_df['start'] = pd.to_numeric(transcript_df['start'], errors='coerce')
    transcript_df = transcript_df[pd.notnull(transcript_df['start'])]

    if transcript_df.empty:
        st.warning('Transcript DataFrame is empty after cleaning.')
        return

    # Calculate the text length for each transcript segment
    transcript_df['text_length'] = transcript_df['text'].apply(len)

    # most common words
    vectorizer = CountVectorizer(stop_words='english')
    word_counts = vectorizer.fit_transform(transcript_df['text'])
    word_counts_df = pd.DataFrame(word_counts.toarray(), columns=vectorizer.get_feature_names_out())
    common_words = word_counts_df.sum().sort_values(ascending=False).head(20)

    # topic Modeling using NMF
    n_features = 1000
    n_topics = 10
    n_top_words = 10

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(transcript_df['text'])
    nmf = NMF(n_components=n_topics, random_state=42).fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names_out()

    def display_topics(model, feature_names, no_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            topics.append(" ".join(topic_words))
        return topics

    topics = display_topics(nmf, tf_feature_names, n_top_words)

    # get topic distribution for each text segment
    topic_distribution = nmf.transform(tf)

    # align the lengths by trimming the extra row in topic_distribution
    topic_distribution_trimmed = topic_distribution[:len(transcript_df)]

    # compute the dominant topic for each text segment
    transcript_df['dominant_topic'] = topic_distribution_trimmed.argmax(axis=1)

    # analyze the content of each text segment to manually identify logical breaks
    logical_breaks = []

    for i in range(1, len(transcript_df)):
        if transcript_df['dominant_topic'].iloc[i] != transcript_df['dominant_topic'].iloc[i - 1]:
            logical_breaks.append(transcript_df['start'].iloc[i])

    # consolidate the logical breaks into broader chapters
    threshold = 60  # seconds
    consolidated_breaks = []
    last_break = None

    for break_point in logical_breaks:
        if last_break is None or break_point - last_break >= threshold:
            consolidated_breaks.append(break_point)
            last_break = break_point
    # merge consecutive breaks with the same dominant topic
    final_chapters = []
    last_chapter = (consolidated_breaks[0], transcript_df['dominant_topic'][0])

    for break_point in consolidated_breaks[1:]:
        current_topic = transcript_df[transcript_df['start'] == break_point]['dominant_topic'].values[0]
        if current_topic == last_chapter[1]:
            last_chapter = (last_chapter[0], current_topic)
        else:
            final_chapters.append(last_chapter)
            last_chapter = (break_point, current_topic)

    final_chapters.append(last_chapter)  # append the last chapter

    # Convert the final chapters to a readable time format
    chapter_points = []
    chapter_names = []

    for i, (break_point, topic_idx) in enumerate(final_chapters):
        chapter_time = pd.to_datetime(break_point, unit='s').strftime('%H:%M:%S')
        chapter_points.append(chapter_time)

        # get the context for the chapter name
        chapter_text = \
        transcript_df[(transcript_df['start'] >= break_point) & (transcript_df['dominant_topic'] == topic_idx)][
            'text'].str.cat(sep=' ')

        # extract key phrases to create a chapter name
        vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
        tfidf_matrix = vectorizer.fit_transform([chapter_text])
        feature_names = vectorizer.get_feature_names_out()
        chapter_name = " ".join(feature_names)

        chapter_names.append(f"Chapter {i + 1}: {chapter_name}")

    # display the final chapter points with names
    st.header("Chapter Points with Names:")
    for time, name in zip(chapter_points, chapter_names):
        st.write(f"{time} - {name}")



def main():
    st.title('Chapterizing YouTube Video!!')

    # Input field for YouTube URL
    url = st.text_input('Enter YouTube Video URL:')
    if not url:
        st.warning('Please enter a YouTube video URL.')
        return

    # Extract video ID from URL
    video_id = get_video_id(url)
    if not video_id:
        st.error('Invalid YouTube URL.')
        return

    # Fetch video title
    title = get_video_title(video_id)
    st.subheader(f'Video Title: {title}')

    # Fetch video transcript
    transcript = get_video_transcript(video_id)
    if not transcript:
        st.warning('No transcript available for this video.')
        return

    # Save transcript to CSV
    filename = f"{video_id}_transcript.csv"
    save_to_csv(title, transcript, filename)

    # Perform machine learning modeling on the transcript data
    model(filename)

    # Download button for transcript
    st.download_button(
        label='Download Transcript',
        data=pd.read_csv(filename).to_csv().encode('utf-8'),
        file_name=filename,
        mime='text/csv'
    )

if __name__ == '__main__':
    main()
