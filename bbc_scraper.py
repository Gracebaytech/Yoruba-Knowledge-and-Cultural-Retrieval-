import requests
from bs4 import BeautifulSoup
import os
import unicodedata
import time
import pandas as pd

# Function to normalize and preserve Yoruba diacritics
def preserve_diacritics(text):
    return unicodedata.normalize('NFC', text)

# Function to clean text  while keeping diacritics
def clean_text(text):
    text = text.replace('\n', ' ').strip()
    return preserve_diacritics(text)

def scrape_bbc_yoruba():
    root_url = 'https://www.bbc.com/yoruba'
    response = requests.get(root_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    art = soup.select('li[class*="bbc"]')
    links = []
    for article in art:
        link = article.find('a')['href']
        if '/articles/' in link:
            links.append(link)

    titles = []
    dates = []
    urls = []
    contents = []
    for url in links:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Get title
        title = clean_text(soup.find('h1').text)
        titles.append(title)
        # Get date
        date = clean_text(soup.find('time')['datetime'])
        dates.append(date)
        # Get url
        urls.append(url)
        # Get content
        article = soup.find_all('div', dir='ltr', class_="bbc-19j92fr ebmt73l0")
        content = ''
        for paragraph in article:
            content += clean_text(paragraph.text)
        contents.append(content)
        time.sleep(2)
        print(f"Scraped successfully: {title}")

    # Convert to dataframe
    df = pd.DataFrame({'title': titles, 'date': dates, 'url': urls, 'content': contents})
    
    # Ask user for folder and file name
    folder_path = input("Enter the folder path to save the file: ").strip()
    file_name = input("Enter the file name (without extension): ").strip()
    file_path = os.path.join(folder_path, f"{file_name}.csv")
    
    # Save the file
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    scrape_bbc_yoruba()
    