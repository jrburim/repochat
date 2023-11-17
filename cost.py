import tiktoken

def calcular_total_tokens(nome_arquivo):
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
            conteudo = arquivo.read()
    except UnicodeDecodeError:
        with open(nome_arquivo, 'r', encoding='ISO-8859-1') as arquivo:  # Tente uma codificação diferente
            conteudo = arquivo.read()
    tokens = list(encoding.encode(conteudo, disallowed_special=()))
    total_tokens = len(tokens)
    return total_tokens

#total = calcular_total_tokens('meu_arquivo.txt')

import os
import fnmatch

def calcular_total_tokens_diretorio(diretorio, extensoes_dev=None):   
    total_tokens = 0
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            if extensoes_dev is None:
                total_tokens += calcular_total_tokens(os.path.join(root, file))
            elif any(fnmatch.fnmatch(file, '*.' + ext) for ext in extensoes_dev):                
                total_tokens += calcular_total_tokens(os.path.join(root, file))
    return total_tokens

def custo(total_tokens):
    #$0.0001 / 1K tokens
    return (total_tokens / 1000) * 0.0001

def custo_embeddings_repo(diretorio):
    extensoes_dev = ["py", "js", "ts", "html", "css", "scss", "json", "xml", "yml", "md", 
                "java", "cpp", "h", "c", "php", "rb", "go", "swift", "kt", "sql",
                "cs", "sh", "pyc", "rs", "tsx", "jsx", "sass", "less", "vue", "rbw",
                "pl", "ps1", "bat", "cmd"]
    total_tokens = calcular_total_tokens_diretorio(diretorio, extensoes_dev=extensoes_dev)
    return { "total_tokens": total_tokens, "custo": custo(total_tokens) }