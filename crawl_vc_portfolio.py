import os
import json
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd
import argparse
import time

from oxen import RemoteRepo
from oxen import Repo
from oxen.auth import config_auth

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
    print("Firecrawl response: ", response_json)
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
    
    print(company_links_list)
    return company_links_list

def portfolio_file_from_name(vc_name):
    name = vc_name.split(' ')
    filename = '_'.join(name)
    return os.path.join(repo_name, f"{filename}_Portfolio.parquet")

def crawl_company_links(company_links_list, n=5):
    company_urls = []
    company_names = []
    company_descriptions = []
    
    # if n is -1, crawl all the companies
    if n == -1:
        n = len(company_links_list)
        
    print("Crawling ", n, " companies")
    
    for company_link in company_links_list[:n]:
        print(f"Crawling: {company_link}")
        try:
            company_name, company_description = get_firecrawl_response(company_link)
            print(f"Company name: {company_name}")
            print(f"Company description: {company_description}")
            company_names.append(company_name)
            company_descriptions.append(company_description)
            company_urls.append(company_link)
        except Exception as e:
            print(f"Error crawling: {e}")
            company_names.append("Error")
            company_descriptions.append(f"Error crawling {company_link}: {e}")
            company_urls.append(company_link)
    return company_urls, company_names, company_descriptions

def get_or_crawl_companies(vc_url, vc_name, num_companies, force=False):
    portfolio_file = portfolio_file_from_name(vc_name)
    print(f"Portfolio file: {portfolio_file}")
    if not force and os.path.exists(portfolio_file):
        print(f"Data for {vc_name} already exists. Use --force to overwrite.")
        # read the file
        df = pd.read_csv(portfolio_file)
        return df
    
    company_links_list = get_portfolio_links(vc_url)
    company_urls, company_names, company_descriptions = crawl_company_links(company_links_list, n=num_companies)
    df = pd.DataFrame({
        'url': company_urls,
        'company_name': company_names,
        'company_description': company_descriptions
    })


    df.to_parquet(portfolio_file, index=False)
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
    print("Evaluation URL: ", url)

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
    print("Evaluation response: ", response_json)
    
    if response.status_code == 200:
        print(f"Evaluation for {vc_name} portfolio kicked off successfully.")
    else:
        print(f"Failed to kick off evaluation for {vc_name} portfolio. Status code: {response.status_code}")
        print(f"Response: {response.text}")

def crawl_vc_portfolio(vc_url, vc_name, num_companies, force=False):
    df = get_or_crawl_companies(vc_url, vc_name, num_companies, force)
    print(df)
    push_to_oxen(vc_name)
    print("Waiting 5 seconds for data to be indexed")
    time.sleep(5)
    kick_off_evaluation(vc_name)

def main():
    parser = argparse.ArgumentParser(description="Crawl VC portfolio companies")
    parser.add_argument("vc_url", help="URL of the VC portfolio page")
    parser.add_argument("vc_name", help="Name of the VC firm")
    parser.add_argument("-n", "--num_companies", type=int, default=5, help="Number of companies to crawl (default: 5)")
    parser.add_argument("-f", "--force", action="store_true", help="Force crawl even if data exists")
    args = parser.parse_args()

    crawl_vc_portfolio(args.vc_url, args.vc_name, args.num_companies, args.force)

if __name__ == "__main__":
    main()
