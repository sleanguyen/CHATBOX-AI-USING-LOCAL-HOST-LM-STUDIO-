================================================================
PROJECT: SPOTIFY DATA AI ASSISTANT (MVP)
MODULE: AIGW201 - INTRODUCTION TO AI
================================================================

STUDENT INFORMATION
-------------------
Name: Nguyễn Lương Gia Bảo
Student ID: baonlggcs240315

1. PROJECT DESCRIPTION
----------------------
This project is an AI-powered chatbot designed to assist managers with analyzing Spotify music data. 
It utilizes a local Large Language Model (LLM) via LM Studio to process natural language queries and 
integrates a Random Forest Machine Learning model for song popularity prediction.

Key Features:
- Conversational AI: Chat naturally with the assistant in English.
- Data Analysis: Automatically generates statistics, summaries, and visualizations (histograms, heatmaps).
- Machine Learning: Predicts song popularity (Low/Medium/High) based on audio features.
- Function Calling: The AI intelligently decides when to call Python functions to fetch data or plots.

2. PREREQUISITES & REQUIREMENTS
-------------------------------
To run this application, the following are required:

A. Software:
   - Python 3.8 or higher
   - LM Studio (for running the local LLM server)

B. Python Libraries:
   - streamlit
   - pandas
   - matplotlib
   - seaborn
   - scikit-learn
   - requests

3. INSTALLATION GUIDE
---------------------
Step 1: Ensure all source files are in the same directory:
   - streamlit_app.py
   - data_analyzer.py
   - lm_chatbot.py
   - SpotifyFeatures.csv (The dataset)

Step 2: Install the required Python packages. 
Open your terminal/command prompt in the project folder and run:
   
   pip install streamlit pandas matplotlib seaborn scikit-learn requests

   (Note: If 'pip' is not recognized, use: python -m pip install ...)

4. LM STUDIO CONFIGURATION (CRITICAL)
-------------------------------------
This project requires a local API server running on LM Studio.

1. Open LM Studio.
2. Download and Load a model that supports instructions and tool use.
   Recommended Model: "Meta-Llama-3-8B-Instruct" (Quantization Q4_K_M is suggested).
3. Navigate to the "Local Server" tab (chip icon).
4. Ensure the server settings are:
   - URL: http://localhost
   - Port: 1234
5. Click "Start Server" (Green button).

5. HOW TO RUN THE APPLICATION
-----------------------------
1. Open your terminal or command prompt.
2. Navigate to the project directory.
3. Run the following command:

   streamlit run streamlit_app.py

   (Or if that fails: py -m streamlit run streamlit_app.py)

4. The application will open automatically in your web browser (usually at http://localhost:8501).

6. USAGE EXAMPLES
-----------------
Once the app is running, you can ask questions like:

- General Chat:
  "Hello, who are you?"
  "Tell me a fun fact about music."

- Data Analysis (Triggers Functions):
  "Can you give me a summary of the dataset?"
  "Show me the correlation heatmap."
  "Analyze the popularity distribution."
  "Analyze the energy feature."

- Prediction (Triggers ML Model):
  "Predict the popularity of a song with danceability 0.8 and energy 0.9."

7. FILE STRUCTURE
-----------------
- streamlit_app.py  : Main application entry point and User Interface logic.
- data_analyzer.py  : Handles data processing, visualization, and the Random Forest model.
- lm_chatbot.py     : Manages connection to LM Studio API and handles function calling logic.
- SpotifyFeatures.csv : The dataset file used for analysis and training.

================================================================
END OF README
================================================================
