import os
from dotenv import load_dotenv
from atlassian import Confluence
from bs4 import BeautifulSoup
import json
    
# loop um mehr als nur 100 pages auf einmal zu bekommen
def get_all_pages(confluence: Confluence, space_key: str, start: int = 0, limit: int = 100):
    """
    Returns all pages from a confluence space

    Args:
        confluence (Confluence Object): Confluence object with url, username and API key
        space_key (String): Key of Confluence space
        start (Integer): Start Index which Confluence pages should be returned
        limit (Integer): Limit how many pages should 

    Returns:
        all_pages (List of JSON Data): List of all pages with properties and body

    """

    all_pages = []
    while True:
        pages = confluence.get_all_pages_from_space(space_key, start, limit, status=None, expand="body.storage", content_type='page')
        all_pages = all_pages + pages
        if len(pages) < limit: 
            break
        start = start + limit
    return all_pages


def get_data_confluence():
    """
    Calls Confluence API to put Confluence Pages in txt files

    """
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
    pages = get_all_pages(confluence, space_key, start = 0, limit = 100)
    pages_dic = {"pages": []}
    for page in pages:
        body = page["title"] +  ":\n\n" + page["body"]["storage"]["value"]
        soup = BeautifulSoup(body, 'html.parser')
        text = soup.get_text(separator="")
        pages_dic["pages"].append({"text": text, "p_id": page["id"], "p_title": page["title"], "p_parent": "not supported"})
        
    file_path = "Data/confluence_data.json"
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(pages_dic, file, indent=4)