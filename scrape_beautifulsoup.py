import requests
from bs4 import BeautifulSoup
import os
import time




url = "https://about.usps.com/who/legal/foia/owned-facilities.htm"

# I saw some issues when trying to access the URL
# This is in the case USPS requires additional headers to access content 
# I think USPS might have some anti-scraping measures in place so this is useful
# to help bypass that by mimicing a browser.


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
response = requests.get(url, headers=headers)#, allow_redirects=False)


#response = requests.get(url, allow_redirects=False)
print(response.status_code)

#print(response.history)  # Shows the chain of redirects
#print(response.url)      # Shows the final URL after redirects


soup = BeautifulSoup(response.content, 'html.parser')
# This sends a GET request to the url and fetches HTML content
# soup parses HTML using BeautifulSoup which makes it easier to
# navigate, search, and modify

csv_links = [] # empty, will be filled with complete links to csv

for link in soup.find_all('a', href=True): # This finds all <a>
# tags which represents hyperlink. The href argument ensures
# only tags with href attribute (specified link URL) are included
    href = link['href'] # all href or all links 

    if href.endswith('.csv'):
        full_url = requests.compat.urljoin(url,href)
        # This forms a complete URL to handle relative links so 
        # all links are absolute
        csv_links.append(full_url)


print(csv_links)
os.makedirs('webscraping_eliot/csv_results', exist_ok=True)

#os.makedirs('csv_output', exist_ok = True)

for csv_url in csv_links:
    time.sleep(2)
    csv_response = requests.get(csv_url, headers=headers)

    # get the last 2 letters from the csv (like ne.csv or la.csv)
    original = csv_url.split('/')[-1] # gets last part of url
    # https://about.usps.com/who/legal/foia/documents/owned-facilities/vt.csv
    # becomes original -> vt.csv
    file_prefix = original.split('.')[0] # this gives us vt

    file_name = f"webscraping_eliot/csv_results/file_{file_prefix}.csv"

    with open(file_name, 'wb') as file:
        file.write(csv_response.content)

     # with statement useful for files, network connections, or things that require
     # proper setup and cleanup after use
     # with ensures file is properly closed once a block of code is executed
     # this way we don't need to manually call file.close()
        

    print(f"Downloaded {file_name}")

print("All downloaded")




"""
for idx, csv_url in enumerate(csv_links):
    wait
    csv_response = requests.get(csv_url)
    # Sends a GET request to csv URL to fetch raw content
    file_name = f"webscraping_eliot/csv_results/file_{idx + 1}.csv"
    with open(file_name, 'wb') as file:
        file.write(csv_response.content)
    print(f"downloaded: {file_name}")

print("Downloaded successfully")
"""