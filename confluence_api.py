import os
from dotenv import load_dotenv
from atlassian import Confluence
from bs4 import BeautifulSoup
import json
from datetime import datetime
    
# loop for getting all pages, because Confluence package has maximum of 100 pages that it returns
def get_all_pages(confluence: Confluence, space_key: str, start: int = 0, limit: int = 100) -> list[dict]:
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
    Calls Confluence API to put Confluence Pages in json file

    """
    # configuration
    load_dotenv()
    username = os.getenv("confluence_username")
    url=os.getenv("confluence_url")
    CONFLUENCE_API_KEY=os.getenv("CONFLUENCE_API")
    # initialize confluence
    space_key = os.getenv("confluence_spacekey")
    confluence = Confluence(
        url=url,
        username=username,
        password=CONFLUENCE_API_KEY
        )
    # scrape all pages of knowledge base 
    pages_api = get_all_pages(confluence, space_key, start = 0, limit = 100)
    # iterate over all pages and only get content and relevant metapadata (title and page id)
    pages = {"pages":[]}
    for page_api in pages_api:
        # get the title and the content of the page 
        body = page_api["body"]["storage"]["value"]
        # remove html stuff 
        soup = BeautifulSoup(body, 'html.parser')
        text = soup.get_text(separator="")

        # Get Page properties
        p_id = page_api["id"]
        p_properties = confluence.get_page_properties(page_id = p_id)      
     
        # Last Edited
        last_edited = p_properties['results'][0]['version']['when']
        dt = datetime.strptime(last_edited, '%Y-%m-%dT%H:%M:%S.%fZ')
        formatted_timestamp = str(dt.strftime('%d.%m.%Y (%H:%M)'))
        type(formatted_timestamp)
        # Get parent id
        parents = confluence.get_page_ancestors(p_id)
        if parents:      
            parent_p_id = parents[0]['id']
        else:
            parent_p_id = "no parent"
    
        page = {"text": text, "p_id": page_api["id"], "p_title": page_api["title"], "p_parent": parent_p_id, "last_edited": formatted_timestamp}
        pages["pages"].append(page)
    
    # store as json file     
    file_path = "Data/confluence_data.json"
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(pages, file, indent=4)
        
# get json file
if __name__ == "__main__":
    get_data_confluence()    