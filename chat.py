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
    EMBEDDINGS = OpenAIEmbeddings(disallowed_special=())
    db = DeepLake(embedding_function=EMBEDDINGS)
    return db

# Função para carregar a lista de repositórios
def load_repos_list():
    if os.path.exists(REPO_LIST_FILE):
        with open(REPO_LIST_FILE, 'rb') as file:
            return pickle.load(file)
    return []

# Configuração de variáveis globais e estado de sessão
#OPENAI_API_KEY = #st.secrets["OPENAI_API_KEY"]
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REPO_LIST_FILE = "repos_list.pkl"

if 'db' not in st.session_state:
    st.session_state.db = init_db()

if 'repos_list' not in st.session_state:
    st.session_state.repos_list = load_repos_list()

def init_chat_history():    
    print("Inicializando chat_history...")
    st.session_state.chat_history = []

    # Mensagens de teste
    # st.session_state.chat_history.append({"text": "Olá! Como posso ajudar?", "is_user": False})
    # st.session_state.chat_history.append({"text": "Qual é a URL do repositório?", "is_user": True})
    # st.session_state.chat_history.append({"text": "https://github.com/meu-repositorio", "is_user": False})
    # st.session_state.chat_history.append({"text": "Processar Repositório", "is_user": True})
    # st.session_state.chat_history.append({"text": "Sim, desejo continuar.", "is_user": False})
    # st.session_state.chat_history.append({"text": "Qual é a minha pergunta?", "is_user": True})
    # st.session_state.chat_history.append({"text": "Qual é a resposta para essa pergunta?", "is_user": False})

if 'chat_history' not in st.session_state:
    init_chat_history()

with st.container():    
    st.title("🔎 LangChain - Chat com Repositórios GitHub")

    # Seleção de Repositório
    selected_repo = st.selectbox("Escolha um repositório", options=st.session_state.repos_list + ["Adicionar novo..."])

    # Adicionar novo Repositório
    if selected_repo == "Adicionar novo...":
        repo_url = st.text_input("Digite a URL do repositório")
        if st.button("Processar Repositório"):
            repo_name, destination_folder = download_and_extract_repo(repo_url)
            total_tokens, custoUSD = custo_embeddings_repo(destination_folder)
            if st.confirm(f"Custo: {custoUSD:.2f} USD. Deseja continuar?"):
                db_add_repo_files(st.session_state.db, repo_name, destination_folder)
                st.session_state.repos_list.append(repo_name)
                with open(REPO_LIST_FILE, 'wb') as file:
                    pickle.dump(st.session_state.repos_list, file)
                shutil.rmtree(destination_folder)
                st.success("Repositório adicionado com sucesso!")    


# Chat
with st.container():
    for msg in st.session_state.chat_history:
        print(msg)
        message(msg['message'], is_user=msg['is_user'])    

# Inicializar Chat
if 'qa_chain' not in st.session_state or selected_repo != st.session_state.get('last_repo', None):
    retriever = st.session_state.db.as_retriever(search_kwargs={'distance_metric': 'cos', 'fetch_k': 100, 'maximal_marginal_relevance': True, 'k': 10, 'filter': {'metadata': { 'repo': selected_repo }}})
    model = ChatOpenAI(model='gpt-3.5-turbo', api_key=OPENAI_API_KEY)
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
    st.session_state.last_repo = selected_repo
    init_chat_history()

# Input do usuário
with st.container():
    # user_input = st.text_input("Digite sua pergunta", key="chat_input")
    # if st.button("Enviar Pergunta") and user_input:
    #     # Adicionar pergunta do usuário ao chat_history
    #     #st.session_state.chat_history.append({"text": user_input, "is_user": True})
    #     #st.session_state.chat_history.append((user_input, result['answer']))

    #     # Obter resposta e adicionar ao chat_history
    #     langchain_history = [(msg["message"], msg["is_user"]) for msg in st.session_state.chat_history]        
    #     result = st.session_state.qa_chain({"question": user_input, "chat_history": langchain_history})
    #     #st.session_state.chat_history.append({"text": result['answer'], "is_user": False})
    #     st.session_state.chat_history.append({"message": user_input, "is_user": True})
    #     st.session_state.chat_history.append({"message": result['answer'], "is_user": False})
        
    #     st.session_state["chat_input"] = ""
    user_input = st.text_input("Digite sua pergunta", key="user_input")

    # Botão invisível para resetar o formulário
    #reset_button = st.button("Reset", key="reset_button", on_click=lambda: st.session_state.update(user_input=""), args=())

    if st.button("Enviar Pergunta"):
        if user_input:
            # Conversão do chat_history para o formato LangChain
            langchain_history = [(msg["message"], "user" if msg["is_user"] else "system") for msg in st.session_state.chat_history]

            # Processamento da pergunta
            result = st.session_state.qa_chain({"question": user_input, "chat_history": langchain_history})

            # Adiciona pergunta e resposta ao chat_history (formato st-chat)
            st.session_state.chat_history.append({"message": user_input, "is_user": True})
            st.session_state.chat_history.append({"message": result['answer'], "is_user": False})

            # Aciona o botão de reset para limpar o input
            #st.session_state.reset_button = True    