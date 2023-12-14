# Importando as bibliotecas necessárias
import streamlit as st
from streamlit_chat import message
import os
import pickle
import shutil
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from functions import custo_embeddings_repo, db_add_repo_files, download_and_extract_repo

# Função para inicializar o banco de dados
def init_db():
    # Definindo as embeddings que serão usadas (neste caso, as embeddings da OpenAI)
    EMBEDDINGS = OpenAIEmbeddings(disallowed_special=())
    # Inicializando o banco de dados
    db = DeepLake(embedding_function=EMBEDDINGS)
    return db

# Função para carregar a lista de repositórios
def load_repos_list():
    # Verifica se o arquivo com a lista de repositórios existe
    if os.path.exists(REPO_LIST_FILE):
        # Se existir, carrega a lista de repositórios do arquivo
        with open(REPO_LIST_FILE, 'rb') as file:
            repo_list = pickle.load(file)
            # Remove duplicatas da lista de repositórios
            repo_list = list(dict.fromkeys(repo_list))
            return repo_list
    # Se o arquivo não existir, retorna uma lista vazia
    return []

# Configuração de variáveis globais e estado de sessão
#OPENAI_API_KEY = #st.secrets["OPENAI_API_KEY"]
# Obtém a chave da API da OpenAI do ambiente
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Define o nome do arquivo onde a lista de repositórios será armazenada
REPO_LIST_FILE = "repos_list.pkl"

# Se o banco de dados ainda não foi inicializado, inicializa
if 'db' not in st.session_state:
    st.session_state.db = init_db()

# Se a lista de repositórios ainda não foi carregada, carrega
if 'repos_list' not in st.session_state:
    st.session_state.repos_list = load_repos_list()

# Função para inicializar o histórico do chat
def init_chat_history():    
    print("Inicializando chat_history...")
    # Inicializa o histórico do chat como uma lista vazia
    st.session_state.chat_history = []

# Se o histórico do chat ainda não foi inicializado, inicializa
if 'chat_history' not in st.session_state:
    init_chat_history()

# Inicia um container para a interface do usuário
with st.container():    
    # Define o título da página
    st.title("Chat com Repositórios GitHub")

    # Seleção de Repositório
    # Cria uma caixa de seleção para o usuário escolher um repositório
    selected_repo = st.selectbox("Escolha um repositório", options=st.session_state.repos_list + ["Adicionar novo..."])

    # Se o usuário escolher "Adicionar novo...", pede para ele digitar a URL do repositório
    if selected_repo == "Adicionar novo...":
        repo_url = st.text_input("Digite a URL do repositório")
        # Se o usuário clicar no botão "Processar Repositório", faz o download do repositório e calcula o custo para processá-lo
        if st.button("Processar Repositório"):
            repo_name, destination_folder = download_and_extract_repo(repo_url)
            total_tokens, custoUSD = custo_embeddings_repo(destination_folder)
            # Exibe o total de tokens e o custo para o usuário
            st.write(f"Total de tokens: {total_tokens}")
            st.write(f"Custo: {custoUSD:.2f} USD")
            
            # Guarda as informações no estado da sessão
            st.session_state['processar_repositorio'] = True
            st.session_state['repo_name'] = repo_name
            st.session_state['destination_folder'] = destination_folder

    # Se o usuário confirmou o processamento do repositório
    if 'processar_repositorio' in st.session_state:
        # Pergunta ao usuário se ele deseja continuar
        st.write("Deseja continuar?")
        # Se o usuário clicar no botão "Não", cancela o processamento do repositório
        if st.button("Não"):
            print("Repositório não adicionado")
            shutil.rmtree(st.session_state['destination_folder'])            
            del st.session_state['processar_repositorio']            
            st.experimental_rerun()
            
        # Se o usuário clicar no botão "Sim", adiciona o repositório ao banco de dados
        if st.button("Sim"):
            print("Adicionando repositório...")
            db_add_repo_files(st.session_state.db, st.session_state['repo_name'], st.session_state['destination_folder'])
            st.session_state.repos_list.append(st.session_state['repo_name'])
            with open(REPO_LIST_FILE, 'wb') as file:
                pickle.dump(st.session_state.repos_list, file)
            shutil.rmtree(st.session_state['destination_folder'])            
            del st.session_state['processar_repositorio']
            print("Repositório adicionado com sucesso!")
            st.experimental_rerun()

# Inicia um container para o chat
with st.container():
    # Exibe as mensagens do histórico do chat
    for msg in st.session_state.chat_history:
        print(msg)
        message(msg['message'], is_user=msg['is_user'])    

# Se o chat ainda não foi inicializado ou se o repositório selecionado mudou, inicializa o chat
if 'qa_chain' not in st.session_state or selected_repo != st.session_state.get('last_repo', None):
    retriever = st.session_state.db.as_retriever(search_kwargs={'distance_metric': 'cos', 'fetch_k': 100, 'maximal_marginal_relevance': True, 'k': 10, 'filter': {'metadata': { 'repo': selected_repo }}})    
    #GPT 4 Turbo:
    model = ChatOpenAI(model='gpt-4-1106-preview', api_key=OPENAI_API_KEY)
    #GPT 3.5 Turbo:
    #model = ChatOpenAI(model='gpt-3.5-turbo', api_key=OPENAI_API_KEY)
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
    st.session_state.last_repo = selected_repo
    init_chat_history()

# Cria uma caixa de texto para o usuário digitar sua pergunta
user_input = st.chat_input('Digite sua pergunta', key="chat_input")
if user_input:    
    # Converte o histórico do chat para o formato LangChain
    langchain_history = [(msg["message"], "user" if msg["is_user"] else "system") for msg in st.session_state.chat_history]

    # Processa a pergunta do usuário
    result = st.session_state.qa_chain({"question": user_input, "chat_history": langchain_history})

    # Adiciona a pergunta e a resposta ao histórico do chat
    st.session_state.chat_history.append({"message": user_input, "is_user": True})
    st.session_state.chat_history.append({"message": result['answer'], "is_user": False})

    # Atualiza a página
    st.experimental_rerun()