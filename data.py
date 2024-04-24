import http
import requests
import json

 # load all new knowledge
def update_data(knowledgeBaseId):
    
    # put it in browser or postman to check if it works
    response = requests.get(f"http://www.confluence.de/rest/servicedeskapi/knowledgebase/{knowledgeBaseId}/articles")
    #response.json()
    
    return None