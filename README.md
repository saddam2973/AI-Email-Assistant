# AI-Powered Email Communication Assistant

This project is my submission for the **AI Engineer Fresher Challenge (Linkenite Oy)**.  
An assistant that automatically filters, categorizes, prioritizes support emails and generates AI-based draft replies, presented in a user-friendly dashboard.

## Features
- Filters emails with support-related keywords.
- Classifies emails by category (Billing, Account, Technical, General).
- Prioritizes urgent requests using keyword detection.
- Performs sentiment analysis on email text.
- Extracts key info like contact details and customer requests.
- Provides AI-generated draft responses.
- Interactive Streamlit dashboard to manage emails and track status.

## Tech Stack
- Python (pandas, nltk, dateutil)
- Streamlit for UI
- Rule-based NLP with placeholder for LLM integration

## How to Run
git clone https://github.com/saddam2973/AI-Email-Assistant
cd AI-Email-Assistant
python -m venv .venv
..venv\Scripts\activate # Windows
pip install -r requirements.txt
python main.py # Process and classify emails
streamlit run app.py # Start the dashboard
## Demo Video
🎥 Watch the demo video here: [Demo Link]([https://drive.google.com/file/d/1ThKwcTcE9LBDMUWptLma22dKjj_iIzzD/view?usp=drive_link])

## Repository Contents
- `main.py`: Batch email processing and classification
- `app.py`: Streamlit dashboard UI
- `Sample_Support_Emails_Dataset.csv`: Input dataset
- `requirements.txt`: Dependencies
- `Documentation.md`: Detailed project documentation
