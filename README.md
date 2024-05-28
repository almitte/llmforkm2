# llmforkm2

Repo for hosting code for developing a solution to a knowledge management problem with LLMs.

## Setup

1. Create a virtual environment and install the required packages:

```zsh
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

2. Create a `.env` file with the following variables:

```zsh
OPENAI_API_KEY = [ENTER YOUR OPENAI API KEY HERE]
PINECONE_API_KEY = [ENTER YOUR PINECONE API KEY HERE]
PINECONE_API_ENV = [ENTER YOUR PINECONE API ENVIRONMENT HERE]
LANGCHAIN_API_KEY = [ENTER YOUR LANGCHAIN API KEY HERE]
CONFLUENCE_API_KEY = [ENTER YOUR CONFLUENCE API KEY HERE]
confluence_username = [ENTER YOUR USERNAME HERE]
confluence_url = [ENTER YOUR SPACE URL HERE]
confluence_spacekey = [ENTER YOUR SPACE KEY HERE]
```

3. Run the website locally in your browser 

```zsh
streamlit run ui.py
```

Mac: Control + C to end
