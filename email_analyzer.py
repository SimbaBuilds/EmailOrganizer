#!/usr/bin/env python3
"""
Email Analyzer: Fetches unread emails from Gmail and categorizes them.
"""
import os
import pickle
import re
import time
from typing import List, Dict, Any
import base64
import webbrowser
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    """Authenticate with Gmail API and return the service object."""
    creds = None
    
    # Check if token.pickle exists
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If credentials are invalid or don't exist, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Use a more permissive approach for testing
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            
            # Set up the flow to handle external apps better
            flow.run_local_server(
                port=0,
                prompt='consent',
                authorization_prompt_message="Please authorize this application to access your Gmail account. When you see the 'Google hasn't verified this app' screen, click 'Advanced' and then 'Go to [App Name] (unsafe)' to proceed."
            )
            creds = flow.credentials
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    # Build and return the Gmail service
    return build('gmail', 'v1', credentials=creds)

def get_unread_email_count(service, user_id='me'):
    """Get the total count of unread emails, excluding spam and archived emails."""
    response = service.users().messages().list(
        userId=user_id,
        q='is:unread in:inbox -in:spam',
        maxResults=1
    ).execute()
    
    # Get the total number of results
    total = response.get('resultSizeEstimate', 0)
    return total

def get_all_unread_emails(service, user_id='me', batch_size=25) -> List[Dict[str, Any]]:
    """Fetch all unread emails from Gmail with progress indicator, excluding spam and archived emails."""
    try:
        # First, get the total count of unread emails
        total_count = get_unread_email_count(service, user_id)
        print(f"Total unread emails (excluding spam and archived): {total_count}")
        
        # Get list of unread messages
        all_messages = []
        next_page_token = None
        
        # Keep fetching pages until we have all messages
        page_count = 0
        while True:
            page_count += 1
            print(f"Fetching page {page_count} of unread emails...")
            response = service.users().messages().list(
                userId=user_id, 
                q='is:unread in:inbox -in:spam',
                pageToken=next_page_token,
                maxResults=100  # Maximum allowed by Gmail API
            ).execute()
            
            messages = response.get('messages', [])
            if not messages:
                break
                
            all_messages.extend(messages)
            
            print(f"Fetched message IDs: {len(all_messages)}/{total_count}")
            
            # Check if there are more pages
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
            # Safety check - if we've fetched more than expected, stop
            if len(all_messages) >= total_count * 1.1:  # Allow 10% margin for error
                print("Reached expected number of unread emails. Stopping fetch.")
                break
        
        if not all_messages:
            print("No unread messages found.")
            return []
            
        # Fetch full message details for each unread message
        emails = []
        total_messages = len(all_messages)
        
        print(f"Processing {total_messages} unread emails...")
        
        for i, message in enumerate(all_messages):
            if i % 10 == 0:
                print(f"Processing email {i+1}/{total_messages}...")
            
            msg = service.users().messages().get(
                userId=user_id, id=message['id'], format='metadata',
                metadataHeaders=['From', 'Subject', 'Date']
            ).execute()
            
            # Extract email data
            email_data = {}
            email_data['id'] = msg['id']
            email_data['threadId'] = msg['threadId']
            
            # Get headers
            headers = msg['payload']['headers']
            for header in headers:
                if header['name'] == 'From':
                    email_data['from'] = header['value']
                if header['name'] == 'Subject':
                    email_data['subject'] = header['value']
                if header['name'] == 'Date':
                    email_data['date'] = header['value']
            
            # Check if the email has a body by looking at the snippet
            # If snippet is not empty, the email has some content
            has_body = bool(msg.get('snippet', '').strip())
            email_data['has_body'] = has_body
            
            # Skip body content for faster processing
            email_data['body'] = ''
            
            emails.append(email_data)
            
            # Add a small delay to avoid rate limiting
            if (i + 1) % batch_size == 0 and i < total_messages - 1:
                print(f"Pausing briefly to avoid rate limits...")
                time.sleep(1)
        
        return emails
    
    except Exception as error:
        print(f'An error occurred: {error}')
        return []

def suggest_categories(emails: List[Dict[str, Any]], num_clusters=5):
    """Suggest categories for emails using clustering."""
    if not emails:
        return []
    
    # Use subject for categorization since we're skipping body content
    documents = []
    for email in emails:
        subject = email.get('subject', '')
        documents.append(subject)
    
    # Extract features with TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        min_df=1,  # Changed from 2 to 1 to work with fewer documents
        max_df=0.9  # Changed from 0.8 to 0.9 to be more inclusive
    )
    
    # If we have too few documents, adjust the clustering
    if len(documents) < num_clusters:
        num_clusters = max(2, len(documents) // 2)
    
    try:
        X = vectorizer.fit_transform(documents)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)
        
        # Get cluster assignments
        clusters = kmeans.labels_
        
        # Get important features for each cluster
        feature_names = vectorizer.get_feature_names_out()
        cluster_centers = kmeans.cluster_centers_
        
        # Create a category name for each cluster based on important words
        categories = []
        for i in range(num_clusters):
            # Get indices of top 3 words for this cluster
            top_indices = cluster_centers[i].argsort()[-3:][::-1]
            top_words = [feature_names[idx] for idx in top_indices if idx < len(feature_names)]
            if not top_words:  # If no words were found
                category_name = f"Category {i+1}"
            else:
                category_name = f"Category {i+1}: {', '.join(top_words)}"
            categories.append(category_name)
        
        # Assign category to each email
        for i, email in enumerate(emails):
            email['category'] = categories[clusters[i]]
        
        return emails
    
    except Exception as e:
        print(f"Error in categorization: {e}")
        # If clustering fails, assign a default category
        for email in emails:
            email['category'] = "Uncategorized"
        return emails

def save_categories_to_file(categories, filename="email_categories.csv"):
    """Save categorized emails to a CSV file."""
    try:
        with open(filename, 'w', encoding='utf-8', newline='') as csvfile:
            # Create CSV writer
            fieldnames = ['Category', 'From', 'Subject', 'Date', 'Has Body']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header with note about excluded emails
            writer.writeheader()
            
            # Write data
            for category, emails in categories.items():
                for email in emails:
                    writer.writerow({
                        'Category': category,
                        'From': email.get('from', 'Unknown'),
                        'Subject': email.get('subject', 'No subject'),
                        'Date': email.get('date', 'Unknown'),
                        'Has Body': email.get('has_body', False)
                    })
                    
        print(f"Categories saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving to file: {e}")
        return False

def main():
    """Main function to fetch and categorize emails."""
    print("Authenticating with Gmail...")
    service = authenticate_gmail()
    
    print("\nFetching all unread emails (excluding spam and archived)...")
    emails = get_all_unread_emails(service)
    
    if not emails:
        print("No unread emails to analyze.")
        return
    
    print(f"\nFound {len(emails)} unread emails (excluding spam and archived).")
    
    print("\nCategorizing emails...")
    categorized_emails = suggest_categories(emails)
    
    # Group emails by category
    categories = {}
    for email in categorized_emails:
        category = email['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(email)
    
    # Display results
    print("\n===== Email Categories =====")
    for category, emails in categories.items():
        print(f"\n{category} ({len(emails)} emails):")
        for i, email in enumerate(emails):
            if i < 3:  # Only show first 3 emails per category in console
                print(f"  - From: {email.get('from', 'Unknown')}")
                print(f"    Subject: {email.get('subject', 'No subject')}")
                print(f"    Date: {email.get('date', 'Unknown')}")
                print()
        if len(emails) > 3:
            print(f"  ... and {len(emails) - 3} more emails in this category.")
    
    # Save results to file
    save_categories_to_file(categories)
    print(f"\nAll categorized emails have been saved to email_categories.csv")

if __name__ == "__main__":
    main() 