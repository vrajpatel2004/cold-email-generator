from langchain_groq import ChatGroq 
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd 
import uuid
import chromadb
import os

def create_model(groq_api):
    llm = ChatGroq(
    temperature = 0.0,
    groq_api_key = groq_api,
    model="llama-3.3-70b-versatile",
    )      
    return llm

def fetch_data_from_url(url_link):
    loader = WebBaseLoader(url_link)
    page_data = loader.load().pop().page_content
    #print(page_data)
    return page_data

def access_perticular_data(llm, page_data):
    prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        the screaped text is from the career page of a website.
        your job is to extract the job postings and return them in JSON format containing
        following keys: 'role', 'experience', 'skills' and 'description'.
        only return the valid JSON.
        ### VALID JSON (NO PRAMBLE):
        """
    )
    chain_extract = prompt_extract | llm  # give data to the llm  
    #print(chain_extract)
    res = chain_extract.invoke(input={'page_data':page_data})
    return res.content

def convert_into_json(res):
    json_parser = JsonOutputParser()
    json_res = json_parser.parse(res)
    return json_res

def create_database(df) :
    client =chromadb.PersistentClient('vectorstore')
    collection = client.get_or_create_collection(name="portfolio")

    if not collection.count():
        for _, row in df.iterrows():
            collection.add(documents=row["Techstack"],
                           metadatas={"links": row["Links"]},
                           ids=[str(uuid.uuid4())])
            
    return collection

def find_job_profile(job, collection) :
    links = collection.query(
    # query_texts=["Experience in python", "Expertise in React Native"],n_results =10).get('metadatas', [])  # search custom job description  
    query_texts=job['skills'], n_results=2).get('metadatas', [])
    return links

def create_email(llm, job, links):
    prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools. 
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
        process optimization, cost reduction, and heightened overall efficiency. 
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
        Remember you are Mohan, BDE at AtliQ. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )

    chain_email = prompt_email | llm
    res = chain_email.invoke({"job_description": str(job), "link_list": links})
    return res.content


def main(groq_api, my_portfolio, job_url):
    # create llm model
    llm_model = create_model(groq_api)
    
    # all data stored in html content from url 
    url_all_content= fetch_data_from_url(job_url)
    #print(url_all_content)
    
    #extract job posting data from all content
    data = access_perticular_data(llm_model, url_all_content)
    #print(data)
    
    #convert data into json
    json_data = convert_into_json(data)
    #print(json_data)
    
    job = json_data
    job['skills']
    #print(job['skills'])
    
    # convert csv into dataframe
    df =pd.read_csv(my_portfolio)
    #print(df)
    
    # create database and stored my protfolio detail in database
    collection = create_database(df)
    
    # find job description in database  
    links = find_job_profile(job, collection)
    #print(links)
    
    email = create_email(llm_model, job, links)
    print(email)


if __name__ == "__main__":
    
    
    groq_api     = "your_api_key"      
    my_portfolio = "./my_portfolio.csv"
    job_url      = "https://jobs.nike.com/job/R-43293?from=job%20search%20funnel"
           
    main(groq_api, my_portfolio, job_url)
