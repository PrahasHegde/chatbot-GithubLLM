import openai
import requests

# Set up OpenAI API access
openai.api_key = "###enter api key"

# Set up GitHub API access
headers = {
    "Authorization": "#enter github token"
}

# Function to fetch repository information
def fetch_repo_info(repo_url):
    repo_api_url = f"https://api.github.com/repos/{repo_url}"
    response = requests.get(repo_api_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Main loop
while True:
    # Get user input
    user_input = input("User: ")

    # Fetch repository information based on the user's query
    repo_url = "#enter repo url"
    repo_info = fetch_repo_info(repo_url)

    # Generate a prompt based on the user's query and repository information
    prompt = f"User: {user_input}\nRepository: {repo_info}\n"

    # Get chatbot response from GPT-3
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=50
    )

    # Display chatbot response
    print("Chatbot:", response["choices"][0]["text"].strip())
