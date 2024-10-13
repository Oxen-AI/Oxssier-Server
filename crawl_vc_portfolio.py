import os
import json
import requests

from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd

from oxen import RemoteRepo
from oxen import LocalRepo
from oxen.auth import config_auth

namespace='ox'
repo_name = 'Investor'

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
    extracted_content = response_json['data']['extract']
    company_name = extracted_content['company_name']
    company_description = extracted_content['company_description']
    return company_name, company_description


def crawl_vc_portfolio(vc_url, vc_name):
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
        messages=[{"role": "user", "content": f"This is all the links in a venture capital portfolio website, help me find the links that belong to their portfolio company.\n {text_for_prompt} \n Anwer with all the links only, splitted with \n."}]
    )

    content = chat_completion.choices[0].message.content
    company_links_list = [line.strip() for line in content.split('\n')]
    print(company_links_list)

    company_names = []
    company_descriptions = []
    for company_link in company_links_list:
        company_name, company_description = get_firecrawl_response(company_link)
        company_names.append(company_name)
        company_descriptions.append(company_description)

    df = pd.DataFrame({
        'url': company_links_list,
        'company_name': company_names,
        'company_description': company_descriptions
    })

    name = vc_name.split(' ')
    filename = '_'.join(name)
    df.to_csv(f"{filename}_Portfolio.csv")
    
    # upload to oxen
    oxen_api_key = os.environ.get("OXENAI_API_KEY")
    config_auth(oxen_api_key)
    
    repo = LocalRepo(f'{repo_name}')
    repo.add(f"{filename}_Portfolio.csv")
    remote_repo = RemoteRepo(f'{namespace}/{repo_name}', host="hub.oxen.ai")
    repo.set_remote("origin", remote_repo.url)
    repo.commit(f"Adding {vc_name} portfolio company data")
    repo.push()
