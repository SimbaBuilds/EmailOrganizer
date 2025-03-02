# Email Organizer

A Python tool that fetches unread emails from Gmail and categorizes them using machine learning.

## Features

- Authenticates with Gmail API
- Fetches unread emails from your inbox (excludes spam and archived emails)
- Categorizes emails using machine learning (K-means clustering)
- Outputs categorized emails to a CSV file
- Provides a summary of email categories in the console

## Requirements

- Python 3.6+
- Google account with Gmail
- Google Cloud Platform project with Gmail API enabled
- OAuth 2.0 credentials for Gmail API

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/SimbaBuilds/EmailOrganizer.git
   cd EmailOrganizer
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up Google API credentials:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Gmail API
   - Create OAuth 2.0 credentials (Desktop application)
   - Download the credentials JSON file and save it as `credentials.json` in the project directory

## Usage

Run the script:
```
python email_analyzer.py
```

The first time you run the script, it will open a browser window for you to authenticate with your Google account. After authentication, the script will:

1. Fetch all unread emails from your inbox (excluding spam and archived emails)
2. Categorize them using machine learning
3. Display the categories in the console
4. Save the categorized emails to a CSV file (`email_categories.csv`)

## Output

The script generates a CSV file with the following columns:
- Category
- From
- Subject
- Date

## License

MIT 
