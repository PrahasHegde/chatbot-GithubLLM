import streamlit as st
from transformers import pipeline, set_token
import os

# Set your Hugging Face token
set_token("your_hugging_face_token")


# Set the HF_HOME environment variable to the directory where Hugging Face credentials are stored
os.environ["HF_HOME"] = "path_to_hugging_face_credentials_directory"


# Initialize Hugging Face pipeline for question answering
qa_pipeline = pipeline("question-answering", model="gpt-3.5-turbo", device=0)

# Streamlit UI
st.title("GitHub Repository Question Answering Chatbot")
repo_url = st.text_input("Enter GitHub repository URL:")

import requests

def get_repo_info(repo_url):
    # Extract owner and repo name from the URL
    parts = repo_url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]

    # GitHub API endpoint for repository information
    url = f"https://api.github.com/repos/{owner}/{repo}"

    # Make a GET request to the GitHub API
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the JSON response
        repo_info = response.json()
        return {
            "name": repo_info["name"],
            "description": repo_info["description"],
            "html_url": repo_info["html_url"]
        }
    else:
        return None


if st.button("Ask"):
    if repo_url:
        # Use the GitHub API to get repository information
        # (you will need to implement this part)
        repo_info = get_repo_info(repo_url)
        if repo_info:
            st.write(f"Repository Name: {repo_info['name']}")
            st.write(f"Description: {repo_info['description']}")
            st.write(f"URL: {repo_info['html_url']}")
            
            # Ask a question about the repository
            question = st.text_input("Ask a question about the repository:")
            if question:
                answer = qa_pipeline({
                    "context": f"Repository: {repo_info['name']}\nDescription: {repo_info['description']}",
                    "question": question
                })
                st.write("Answer:", answer['answer'])
        else:
            st.write("Error: Unable to retrieve repository information. Please check the repository URL.")
    else:
        st.write("Please enter a GitHub repository URL.")



##################################################################
#Run with streamlit run app3.py