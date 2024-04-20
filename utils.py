import os
import shutil
import git
import dotenv
import deeplake
from queue import Queue
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI

class Embedder:
    def __init__(self, repo_url, openai_api_key):
        self.repo_url = repo_url
        self.openai_api_key = openai_api_key
        self.git_link = repo_url
        last_name = self.git_link.split('/')[-1]
        self.clone_path = last_name.split('.')[0]
        self.deeplake_path = f"hub://priyadwivedi/{self.clone_path}"
        self.model = ChatOpenAI(model_name="gpt-3.5-turbo-0125")  # switch to 'gpt-4'
        self.hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.openai = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.MyQueue = Queue(maxsize=2)
        self.token = dotenv.load_dotenv('DEEPLAKE_TOKEN')
        self.retriever = None



    def clone_repo(self):
        try:
            if not os.path.exists(self.clone_path):
                git.Repo.clone_from(self.git_link, self.clone_path)
            return True
        except Exception as e:
            print(f"Failed to clone repository: {e}")
            return False

    def extract_all_files(self):
        root_dir = self.clone_path
        self.docs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in ['.py', '.ipynb', '.md']:
                    try:
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        self.docs.extend(loader.load_and_split())
                    except Exception as e:
                        print(f"Failed to load file {file}: {e}")

    def chunk_files(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = text_splitter.split_documents(self.docs)
        self.num_texts = len(self.texts)

    def embed_deeplake(self):
        db = DeepLake(dataset_path=self.deeplake_path, embedding_function=self.hf)
        db.add_documents(self.texts)
        # Remove data from the cloned path
        self.delete_directory(self.clone_path)
        return db

    def delete_directory(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)

    def load_db(self):
        exists = deeplake.exists(self.deeplake_path)
        if exists:
            ## Just load the DB
            self.db = DeepLake(
            dataset_path=self.deeplake_path,
            read_only=True,
            embedding_function=self.hf,
             )
        else:
            ## Create and load
            self.extract_all_files()
            self.chunk_files()
            self.db = self.embed_deeplake()

        self.retriever = self.db.as_retriever()
        self.retriever.search_kwargs['distance_metric'] = 'cos'
        self.retriever.search_kwargs['fetch_k'] = 100
        self.retriever.search_kwargs['k'] = 3

    def retrieve_results(self, query):
        if self.retriever is None:
            # Initialize the retriever if it's not already initialized
            self.load_db()

        chat_history = list(self.MyQueue.queue)
        qa = ConversationalRetrievalChain.from_llm(self.model, chain_type="stuff", retriever=self.retriever, condense_question_llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo'))
        result = qa({"question": query, "chat_history": chat_history})
        self.add_to_queue((query, result["answer"]))
        return result['answer']

       

