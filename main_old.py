import os

# pip install --upgrade langchain deeplake openai tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from deeplake.core.dataset import Dataset
from cost import custo_embeddings_repo

# Cant continue w/o API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    print("missing api key")
    exit(-1)

# We'll be using OpenAI's embeddings
EMBEDDINGS = OpenAIEmbeddings(disallowed_special=())
TARGET_REPO = 'tiktoken-main'



# def load_deeplake():


#     docs = loader.load_and_split()
#     #db.add_document('This is a test', metadata=[])    
#     return db

# Exemplo de criar um banco de dados com documentos de várias fontes
# print("Loading repo files...")
# db = DeepLake(embedding_function=EMBEDDINGS)

# all_docs = []

# def add_doc(file):   
#     loader = TextLoader(f'/Users/admin/senac/proj2/{file}.md')
#     docs = loader.load_and_split()
#     for doc in docs:
#         doc.metadata['repo'] = file
#     all_docs.extend(docs)    
# add_doc('test1')
# add_doc('test2')        
# print(all_docs)
# print("Loaded")
# db.add_documents(all_docs)

#db = DeepLake(read_only=True, embedding_function=EMBEDDINGS)
#documentos = db.similarity_search("Qual o nome do artista?", filter={'metadata': { 'repo':'test2' }})
#print(db.get())
#print("Loaded", documentos)

## Verificando se tem o repositório no banco de dados
# len(db.vectorstore.search(filter={'metadata': { 'repo':'test1' }})['id']) -> retorna a qtd de documentos

########################
# print("Loading repo files...")
# db = DeepLake(read_only=True, embedding_function=EMBEDDINGS)
# print("Loaded")
# retriever = db.as_retriever( search_kwargs={'distance_metric': 'cos', 'fetch_k': 100, 'maximal_marginal_relevance': True, 'k': 10, 'filter': {'metadata': { 'repo':'test2' }}})
# # retriever.search_kwargs['distance_metric'] = 'cos'
# # retriever.search_kwargs['fetch_k'] = 100
# # retriever.search_kwargs['maximal_marginal_relevance'] = True
# # retriever.search_kwargs['k'] = 10
# # retriever.search_kwargs['filter']: {'metadata': { 'repo':'test1' }}
# chat_history = []
# model = ChatOpenAI(model='gpt-3.5-turbo')
# qa_chain = ConversationalRetrievalChain.from_llm(model,retriever=retriever)
# result = qa_chain({"question":  "Qual o nome do artista?", "chat_history": chat_history})
# print(result['answer'])

# def construct_chain() -> ConversationalRetrievalChain:
#     """
#     Build a retriever for our dataset, then build and return a 
#     ConversationalChain that we can interact with
#     """
    # db = load_repo_files()
    # retriever = db.as_retriever()
    # retriever.search_kwargs['distance_metric'] = 'cos'
    # retriever.search_kwargs['fetch_k'] = 100
    # retriever.search_kwargs['maximal_marginal_relevance'] = True
    # retriever.search_kwargs['k'] = 10

#     model = ChatOpenAI(model='gpt-3.5-turbo')
#     qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)
#     return qa

# def main():
#     chat_history = []
#     questions = [
#     "What are the top 3 programming languages present in this application?",
#     "What are the key components of this Android app?",
#     "List any functions or classes that are related to authentication or authorization",
#     "Are there any authentication or authorization issues in this code?",
#     "Are there any input validation or output encoding issues in the code?",
#     "Are there any insecure cryptographic implementations in the code?",
#     "Are there any known CVEs associated with the libraries used?",
#     ] 

#     qa_chain = construct_chain()
#     for question in questions:  
#         result = qa_chain({"question": question, "chat_history": chat_history})
#         chat_history.append((question, result['answer']))
#         print(f"-> **Question**: {question} \n")
#         print(f"**Answer**: {result['answer']} \n")

#db = load_deeplake()


#custo = custo_embeddings_repo(TARGET_REPO)
#print(custo)
