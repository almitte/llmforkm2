import os
from dotenv import load_dotenv
from atlassian import Confluence
from bs4 import BeautifulSoup
    
# loop um mehr als nur 100 pages auf einmal zu bekommen
def get_all_pages(confluence, space):
    start = 0
    limit = 100
    all_pages = []
    while True:
        pages = confluence.get_all_pages_from_space(space, start, limit, status=None, expand="body.storage", content_type='page')
        all_pages = all_pages + pages
        if len(pages) < limit: 
            break
        start = start + limit
    return all_pages


def get_data_confluence():
    # Configuration
    load_dotenv()
    username = os.getenv("confluence_username")
    url=os.getenv("confluence_url")
    CONFLUENCE_API_KEY=os.getenv("CONFLUENCE_API")
    # Initialize Confluence
    space_key = os.getenv("confluence_spacekey")
    confluence = Confluence(
        url=url,
        username=username,
        password=CONFLUENCE_API_KEY
        )
    # web scraper mir Rest-API
    pages = get_all_pages(confluence, space_key)
    bodies = ""
    for page in pages:
        bodies = bodies + "\n\n\n#################################\n\n\n"
        bodies = bodies + page["title"] +  ":\n\n" + page["body"]["storage"]["value"]
        
    # html Zeug entfernen    
    soup = BeautifulSoup(bodies, 'html.parser')

    # Extract text from the parsed HTML
    text = soup.get_text(separator="")
        
    # Saving the text to a .txt file
    # better as json with metadata like page name, page id and maybe subpage id
    file_path = "confluence_data.txt"  # Define the path and file name
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)
