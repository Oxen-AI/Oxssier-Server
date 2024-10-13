from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
import os
import json
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd
import argparse
import time
from urllib.parse import unquote
from oxen import RemoteRepo
from oxen import Repo
from oxen.auth import config_auth

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)


namespace='ox'
repo_name = 'Investors'

def get_firecrawl_response(company_url):
    api_key = os.environ.get("FIRECRAWL_API_KEY")

    url = 'https://api.firecrawl.dev/v1/scrape'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    data = {
        "url": f"{company_url}",
        "formats": ["extract"],
        "extract": {
            "schema": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string"
                    },
                    "company_description": {
                        "type": "string"
                    }
                
                },
                "required": [
                    "company_name",
                    "company_description"      
                ]
            }
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    response_json = response.json()
    logging.debug("Firecrawl response: ", response_json)
    extracted_content = response_json['data']['extract']
    company_name = extracted_content['company_name']
    company_description = extracted_content['company_description']
    return company_name, company_description


def get_portfolio_links(vc_url):
    vc_page_response = requests.get(vc_url)
    soup = BeautifulSoup(vc_page_response.content, 'html.parser')

    company_urls = []

    # Filter the tags with url
    for a_tag in soup.find_all('a', href=True):  
        company_urls.append(a_tag['href']) 

    text_for_prompt = ', '.join(company_urls)


    # Use gpt-4o to find the company links
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"This is all the links in a venture capital portfolio website, help me find the links that belong to their portfolio company.\n {text_for_prompt} \n Answer with all the links only, links should be delimited with a new line character \n."}]
    )

    content = chat_completion.choices[0].message.content
    company_links_list = [line.strip() for line in content.split('\n')]
    # filter out any strings that don't start with http
    company_links_list = [link for link in company_links_list if link.startswith('http')]
    
    # convert http:// to https://
    company_links_list = [link.replace('http://', 'https://') for link in company_links_list]
    
    logging.debug(company_links_list)
    return company_links_list

def portfolio_file_from_name(vc_name):
    name = vc_name.split(' ')
    filename = '_'.join(name)
    return os.path.join(repo_name, f"{filename}_Portfolio.jsonl")

def crawl_company_links(company_links_list, n=5):
    company_urls = []
    company_names = []
    company_descriptions = []
    
    # if n is -1, crawl all the companies
    if n == -1:
        n = len(company_links_list)
        
    logging.debug(f"Crawling {n} companies")
    
    for company_link in company_links_list[:n]:
        logging.debug(f"Crawling: {company_link}")
        try:
            company_name, company_description = get_firecrawl_response(company_link)
            logging.debug(f"Company name: {company_name}")
            logging.debug(f"Company description: {company_description}")
            company_names.append(company_name)
            company_descriptions.append(company_description)
            company_urls.append(company_link)
        except Exception as e:
            logging.debug(f"Error crawling: {e}")
            company_names.append("Error")
            company_descriptions.append(f"Error crawling {company_link}: {e}")
            company_urls.append(company_link)
    return company_urls, company_names, company_descriptions

def get_or_crawl_companies(vc_url, vc_name, num_companies, force=False):
    portfolio_file = portfolio_file_from_name(vc_name)
    logging.debug(f"Portfolio file: {portfolio_file}")
    if not force and os.path.exists(portfolio_file):
        logging.debug(f"Data for {vc_name} already exists. Use --force to overwrite.")
        # read the file
        df = pd.read_json(portfolio_file, lines=True)
        return df
    
    company_links_list = get_portfolio_links(vc_url)
    company_urls, company_names, company_descriptions = crawl_company_links(company_links_list, n=num_companies)
    df = pd.DataFrame({
        'url': company_urls,
        'company_name': company_names,
        'company_description': company_descriptions
    })

    df.to_json(portfolio_file, orient='records', lines=True)
    return df

def push_to_oxen(vc_name):
    # Add a check for existing data before crawling
    portfolio_file = portfolio_file_from_name(vc_name)

    # upload to oxen
    oxen_api_key = os.environ.get("OXENAI_API_KEY")
    config_auth(oxen_api_key)
    
    repo = Repo(f'{repo_name}')
    repo.add(portfolio_file)
    remote_repo = RemoteRepo(f'{namespace}/{repo_name}', host="hub.oxen.ai")
    repo.set_remote("origin", remote_repo.url)
    repo.commit(f"Adding {vc_name} portfolio company data")
    repo.push()
    
def kick_off_evaluation(vc_name):
    oxen_api_key = os.environ.get("OXENAI_API_KEY")
    namespace = 'ox'  # Make sure this matches your namespace
    repo_name = 'Investors'  # This should match your repo_name variable
    resource = portfolio_file_from_name(vc_name)  # Use the CSV file as the resource
    
    # Replace the reponame in the path with 'main'
    resource = resource.replace(repo_name, 'main')

    url = f"https://hub.oxen.ai/api/repos/{namespace}/{repo_name}/evaluations/{resource}"
    logging.debug(f"Evaluation URL: {url}")

    headers = {
        "Authorization": f"Bearer {oxen_api_key}",
        "Content-Type": "application/json"
    }

    oxen_ai_description = """
Oxen.ai is a platform for versioning, storing, and evaluating data.
"""

    prompt = f"""
You are an expert at evaluating venture capital portfolios. You are considering the following portfolio:

{oxen_ai_description}

Here is another portfolio company:

{{company_description}}

Are these two companies competitive with each other? Respond with one word only, all lowercase: "true" or "false".
"""

    data = {
        "name": f"{vc_name} Portfolio Evaluation",
        "prompt": prompt,
        "type": "text",
        "model": "gpt-4o-mini",
        "is_sample": False,
        "target_column": "industry_prediction",
        "target_branch": f"api-results-branch-{vc_name}",
        "auto_commit": True,
        "commit_message": f"Evaluation results for {vc_name} portfolio"
    }

    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    logging.debug(f"Evaluation response: {response_json}")
    
    if response.status_code == 200:
        logging.debug(f"Evaluation for {vc_name} portfolio kicked off successfully.")
    else:
        logging.debug(f"Failed to kick off evaluation for {vc_name} portfolio. Status code: {response.status_code}")
        logging.debug(f"Response: {response.text}")

def crawl_vc_portfolio(vc_url, vc_name, num_companies, force=False):
    if force:
        df = get_or_crawl_companies(vc_url, vc_name, num_companies, force)
        logging.debug(df)
        push_to_oxen(vc_name)
        logging.debug("Waiting 5 seconds for data to be indexed")
        time.sleep(5)
        kick_off_evaluation(vc_name)
    else:
        df = get_or_crawl_companies(vc_url, vc_name, num_companies, force)
        logging.debug(df)


# In-memory storage for demonstration purposes
data_store = []

def list_results(vc_name):
    portfolio_file = portfolio_file_from_name(vc_name)
    file_path = portfolio_file.replace(repo_name, f'api-results-branch-{vc_name}')
    url = f"https://hub.oxen.ai/api/repos/{namespace}/{repo_name}/file/{file_path}"
    logging.debug(f"File path: {file_path}")
    logging.debug(f"URL: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.text
        
        logging.debug(data)
        
        # Parse the line-delimited JSON
        results = []
        for line in data.splitlines():
            company_data = json.loads(line)
            logging.debug(company_data)
            results.append({
                "url": company_data.get("url", ""),
                "name": company_data.get("company_name", ""),
                "description": company_data.get("company_description", ""),
                "competitive": company_data.get("industry_prediction", False)
            })
        
        return results
    except requests.RequestException as e:
        logging.error(f"Error fetching results for {vc_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching results")

def is_competitive(company_a, company_b):
    return False

def crawl_portfolio_website(row):
    name = row['name']
    url = row['url']
    logging.debug(url)
    return [
        {
            "name": name,
            "description": "Description of Company A"
        }
    ]

def crawl_vc_website(url):
    return [
        {
            "name": "Company A",
            "url": "https://www.company-a.com"
        },
        {
            "name": "Company B",
            "url": "https://www.company-b.com"
        }
    ]

class CrawlRequest(BaseModel):
    url: str
    prompt: str
    name: str
    numCompanies: int

class CompanyData(BaseModel):
    url: str
    name: str
    description: str
    competitive: bool

@app.post('/api/crawl', status_code=201)
async def add_data(vc_crawl_request: CrawlRequest):
    try:
        vc_crawl_request.name = vc_crawl_request.name.replace(" ", "_")
        logging.debug(f"Received data: {vc_crawl_request}")
        
        url = vc_crawl_request.url
        data = crawl_vc_website(url)
        for row in data:
            data_store.append(crawl_portfolio_website(row))

        data_store.append(vc_crawl_request.dict())
        
        crawl_vc_portfolio(url, vc_crawl_request.name, vc_crawl_request.numCompanies, force=True)
        logging.debug("Waiting 5 seconds for data to be indexed")
        time.sleep(5)
        
        return {"message": "Data added successfully", "data": vc_crawl_request}
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get('/api/results/{vc_name}', response_model=List[CompanyData])
async def get_data(vc_name: str):
    # url decode vc_name
    vc_name = unquote(vc_name)
    logging.debug(f"Getting data for {vc_name}")
    vc_name = vc_name.replace(" ", "_")
    response = list_results(vc_name)
    return response

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
