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


OPENAI_API_KEY = "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"
PINECONE_API_KEY = "7013bfb3-4ebc-4181-a894-c9721d1bc1bb"
PINECONE_API_ENV = "us-east-1"
LANGCHAIN_API_KEY = "ls__b502fece98e349a18a0a5178ebeaedbc"
CONFLUENCE_API = "ATATT3xFfGF0zh7EBrasRHZQEerjfJTdqleap-5uJqM9qFTdP8V8sZY91qm3hcu1kdKC0GnIHET_TPlFSLcx-nJ1OA3hvwJBSkhtn_veaqpJDxqItb_NXdAjfR4OIKh31a3VEYqpdTeYGeHYZx1MaBn4KAHSJWlHyba0i4pCTuLSEctUH82_cAk=A878B409"
confluence_username = "max1912sch@gmail.com"
confluence_url = "https://llmgruppenarbeit.atlassian.net/"
confluence_spacekey = "KB"