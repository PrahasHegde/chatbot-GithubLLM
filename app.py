import os
import git
from langchain_text_splitters import Language


class MockChroma:
    @classmethod
    def from_texts(cls, texts, embeddings):
        return cls()

    def as_retriever(self, search_type, search_kwargs):
        return self



# Clone the repository
repo_url = "https://github.com/PrahasHegde/Return-to-Zero-digital-signal.git"
repo_name = "Return-to-Zero-digital-signal"
repo_path = os.path.join(os.getcwd(), repo_name)
if not os.path.exists(repo_path):
    git.Repo.clone_from(repo_url, repo_path)

# Load the Python code
documents = []
for root, dirs, files in os.walk(repo_path):
    for file in files:
        if file.endswith(".py"):
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                documents.append(f.read())

print(f"Number of documents loaded: {len(documents)}")

# Split python files
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)

# Ensure documents is not empty before splitting
if documents:
    # Join the list of documents into a single string
    text = "\n".join(documents)
    texts = python_splitter.split_text(text)
    print(f"Number of texts: {len(texts)}")

    # Continue with the rest of your code...
else:
    print("No texts found. Exiting.")

