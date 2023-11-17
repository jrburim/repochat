import os
import requests
import io
from zipfile import ZipFile
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from deeplake.core.dataset import Dataset

def download_and_extract_repo(url):
    # Extrai o nome do repositório da URL
    repo_name = url.split("/")[-1].replace(".git", "")
    
    # Faz o download do arquivo zip do repositório
    zip_url = f"{url}/archive/refs/heads/main.zip"
    response = requests.get(zip_url)
    
    # Cria uma pasta com o nome do repositório e extrai o conteúdo do zip nela
    os.makedirs(repo_name, exist_ok=True)
    with ZipFile(io.BytesIO(response.content)) as zip_file:
        zip_file.extractall(repo_name)
    
    extracted_dir: None;

    # le dentro da pasta do repositório, não entrar em sub pastas
    for dir_branch in os.listdir(repo_name):
        if os.path.isdir(os.path.join(repo_name, dir_branch)):
            extracted_dir = os.path.join(repo_name, dir_branch)
            break

    if extracted_dir != None:
        # move o conteúdo da pasta extraída para a pasta do repositório
        for dir in os.listdir(extracted_dir):
            os.rename(os.path.join(extracted_dir, dir), os.path.join(repo_name, dir))
        # deleta a pasta extraída
        os.rmdir(extracted_dir)

    return repo_name

def db_add_repo_files(db, repo) -> Dataset:
    """
    Walk our target codebase and load all our files
    for chunking and then text embedding
    """    
    documents = []
    for dirpath, dirnames, filenames in os.walk(repo):
        for file in filenames:
            try: 
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                documents.extend(loader.load_and_split())
            except UnicodeDecodeError:
                loader = TextLoader(os.path.join(dirpath, file), encoding='ISO-8859-1')
                documents.extend(loader.load_and_split())    
            except Exception as e: 
                print(e)

    for doc in documents:
      doc.metadata['repo'] = file  
    
    # chunk our files
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)

    # generate text embeddings for our target codebase
    # store in local database ./deeplake/
    # db = DeepLake(embedding_function=EMBEDDINGS)
    db.add_documents(chunks)
    return db

def get_retriever(db, repo):
    return db.as_retriever( search_kwargs={'distance_metric': 'cos', 'fetch_k': 100, 'maximal_marginal_relevance': True, 'k': 10, 'filter': {'metadata': { 'repo':repo }}})

def check_repo_in_db(db, repo):
    return len(db.vectorstore.search(filter={'metadata': { 'repo':repo }})['id']) > 0


