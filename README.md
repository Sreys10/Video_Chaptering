# Video Chaptering with YouTube Transcripts

This project provides a way to segment YouTube videos into chapters based on their content. By inputting a YouTube video link, the application transcribes the video and uses machine learning techniques to identify logical breaks in the content, resulting in a chapter-based breakdown with timestamps and chapter names.

## Features

- **YouTube Video Transcription:** Extracts the transcript of a given YouTube video.
- **Topic Modeling:** Uses machine learning algorithms to identify different topics within the video.
- **Chapter Generation:** Breaks down the video into chapters based on topic changes, providing timestamps and chapter titles.
- **Downloadable Transcript:** Offers the option to download the transcript in CSV format.

## Libraries Used

- **Streamlit:** For creating the web application interface.
- **Google API Client:** To fetch video details from YouTube.
- **YouTube Transcript API:** To retrieve the transcript of the video.
- **scikit-learn:** For machine learning tasks such as topic modeling.
- **pandas:** For handling and processing transcript data.
- **NMF (Non-negative Matrix Factorization):** For topic modeling and identifying different segments of the video.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Sreys10/Video_Chaptering.git
    cd video-chaptering
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter the YouTube video URL in the input field and press Enter.

4. The application will fetch the video title and transcript, process the transcript to identify chapters, and display the chapter points with names.

5. You can download the transcript by clicking the "Download Transcript" button.

## Code Explanation

### Main Application (`main.py`)

- **Extract Video ID:** The `get_video_id` function extracts the video ID from the provided YouTube URL using a regular expression.
- **Fetch Video Title:** The `get_video_title` function retrieves the video title using the YouTube Data API.
- **Fetch Video Transcript:** The `get_video_transcript` function gets the transcript of the video using the YouTube Transcript API.
- **Save Transcript to CSV:** The `save_to_csv` function saves the transcript to a CSV file along with the video title.
- **Modeling and Chapter Generation:** The `model` function performs text analysis on the transcript to identify topics using NMF and generates logical chapter breaks.

### Machine Learning and Text Analysis

- **Text Vectorization:** Uses `CountVectorizer` to convert transcript text into a matrix of token counts.
- **Topic Modeling:** Applies NMF to identify topics within the transcript.
- **Chapter Identification:** Determines logical breaks in the transcript based on topic changes and consolidates them into chapters.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

