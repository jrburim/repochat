import os
import pickle

# pip install --upgrade langchain deeplake openai tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from deeplake.core.dataset import Dataset
import inquirer
import shutil

from functions import custo_embeddings_repo, db_add_repo_files, download_and_extract_repo

# Cant continue w/o API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    print("Chave da API não encontrada")
    exit(-1)

# We'll be using OpenAI's embeddings
EMBEDDINGS = OpenAIEmbeddings(disallowed_special=())

print("Inicializando banco de dados...")

arquivo_lock = "deeplake/dataset_lock.lock"
if os.path.exists(arquivo_lock):
    os.remove(arquivo_lock)

db = DeepLake(embedding_function=EMBEDDINGS)

print("Banco de dados inicializado! ")

REPO_LIST_FILE = "repos_list.pkl"
repos_list = [ ]

if os.path.exists(REPO_LIST_FILE):
    with open(REPO_LIST_FILE, 'rb') as file:
        repos_list = pickle.load(file)

# Pergunta ao usuário qual repositório ele deseja trabalhar
questions = [
    inquirer.List('repo',
                  message="Qual repositório você deseja trabalhar?",
                  choices=repos_list + ["Outro..."],
              ),
]

repoName = None
answers = inquirer.prompt(questions)
repoName = answers['repo']

if repoName == "Outro...":
    questions = [
        inquirer.Text('repoURL',
                      message="Digite a URL do repositório"),
    ]

    answers_other = inquirer.prompt(questions)

    # Adiciona o repositório ao banco de dados
    repoURL = answers_other['repoURL']

    assert repoURL is not None, "URL do repositório vazia, abortando..."
    repoName, destination_folder = download_and_extract_repo(repoURL)

    # Calcula o custo de adicionar o repositório ao banco de dados
    total_tokens, custoUSD = custo_embeddings_repo(destination_folder)

    # Exibe o custo em USD
    print(f"Número total de tokens: {total_tokens}")
    print(f"Custo em USD: {custoUSD:.2f}")

    # Confirma a geração dos embeddings
    questions = [
        inquirer.Confirm('confirmacao',
                         message="Deseja gerar os embeddings?",
                         default=True),
    ]

    answers_other = inquirer.prompt(questions)
    confirmacao = answers_other['confirmacao']

    if confirmacao:
        # Gera os embeddings
        #db.add_documents(repoName, all_docs)
        db_add_repo_files(db, repoName, destination_folder)
        
        # Adicionando na lista de repositórios e salvando no disco
        repos_list.append(repoName)
        with open(REPO_LIST_FILE, 'wb') as file:
            pickle.dump(repos_list, file)

        # Apagando a pasta do repositório 
        shutil.rmtree(destination_folder)    

        print("Embeddings gerados com sucesso!")
    else:
        print("Geração dos embeddings cancelada.")
        exit(0)

# Seleciona o repositório escolhido pelo usuário
retriever = db.as_retriever( search_kwargs={'distance_metric': 'cos', 'fetch_k': 100, 'maximal_marginal_relevance': True, 'k': 10, 'filter': {'metadata': { 'repo': repoName }}})
chat_history = []
model = ChatOpenAI(model='gpt-3.5-turbo')
qa_chain = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

print("Inicializando chatbot...")
while True:
    question = input("Digite sua pergunta: ")
    if question == "exit"  or question == "sair" or question == "":
        break
    result = qa_chain({"question": question, "chat_history": chat_history})
    #print all keys from result
    print(result.keys())
    chat_history.append((question, result['answer']))    
    print(f" >>>>> : {result['answer']} \n")