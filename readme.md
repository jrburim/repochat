# Sobre o Projeto

SENAC - Projetos 2 - Chat com repositórios do GitHub

Este repositório é uma aplicação de chatbot que utiliza a LangChain (com o LLM da OpenAI) para responder perguntas baseadas em repositórios do GitHub. O chatbot é capaz de processar repositórios, adicionar novos repositórios e responder perguntas com base no conteúdo do repositório selecionado.

O arquivo principal do projeto é o `chat.py`, que contém a lógica principal do chatbot e a interface do usuário.

## Autores

- Alexandre Moraes de Souza Lima
- Edgard de Souza Lemos Junior
- Luiz Antonio Burim Junior
- Mauricio dos Santos Menandro

## chat.py

O `chat.py` é o arquivo principal do projeto. Ele utiliza a biblioteca Streamlit para criar uma interface de usuário interativa para o chatbot. O arquivo contém várias funções e blocos de código que realizam tarefas específicas:

- Inicialização do banco de dados
- Carregamento da lista de repositórios
- Configuração de variáveis globais e estado de sessão
- Seleção de repositório
- Processamento de repositórios
- Inicialização do histórico de chat
- Adição de repositórios ao banco de dados
- Inicialização do chatbot
- Processamento de perguntas do usuário

Depois de instalar as dependências, você pode iniciar o aplicativo com o seguinte comando:

```bash
streamlit run chat.py
```

Isso iniciará o servidor Streamlit e abrirá o aplicativo em seu navegador padrão.

## Variáveis de Ambiente

Este projeto utiliza a chave da API da OpenAI, que é lida como uma variável de ambiente. Certifique-se de definir a variável de ambiente `OPENAI_API_KEY` com sua chave da API da OpenAI antes de iniciar o aplicativo.

## Dependências

Este projeto depende de várias bibliotecas Python, que estão listadas no arquivo `requirements.txt`. Você pode instalar todas as dependências com o seguinte comando:

```bash
pip install -r requirements.txt
```

## Contribuindo

Contribuições são sempre bem-vindas! Sinta-se à vontade para abrir um problema ou enviar um pull request.

## Licença

Este projeto está licenciado sob os termos da licença MIT.
