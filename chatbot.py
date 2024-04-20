import streamlit as st
import langchain
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

st.title("Chat-with-your-codebase")

# Get user inputs
user_repo = st.text_input("Github Link to your public codebase", "######enter repo")
openai_api_key = dotenv.load_dotenv('OPENAI_API_KEY')

# Initialize Langchain models
retriever = langchain.Retriever()
generator = langchain.Generator(api_key=openai_api_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the GitHub Repo
if st.button("Load Repository"):
    # Placeholder code for cloning and loading repository
    st.success("Repository cloned and loaded successfully.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.echo(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.text_input("Type your question here:")

# Process user input if a question is provided
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.echo("user"):
        st.markdown(prompt)
    # Retrieve relevant documents from repository
    relevant_documents = retriever.retrieve_documents(user_repo)
    # Generate response based on the prompt and relevant documents
    response = generator.generate_response(prompt, relevant_documents)
    # Display chatbot response in chat message container
    with st.echo("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


#################################################################################
#Run with  streamlit run chatbot.py       