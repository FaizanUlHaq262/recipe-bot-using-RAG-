from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import warnings
warnings.filterwarnings('ignore')


#initialize the database and model
def readChromadb(folderPath: str, model: str, collection_name: str):
    embedder = OllamaEmbeddings(model=model, show_progress=True)
    vectorDB = Chroma(persist_directory=folderPath, collection_name=collection_name, embedding_function=embedder)
    return vectorDB

#process the query using the RAG process and get results
def makeOllamaQuery(vectorDB, model: str, numResults: int, promptQuery, ragPromptTemplate, userInp: str):
    llm = ChatOllama(model=model)
    retriever = MultiQueryRetriever.from_llm(vectorDB.as_retriever(num_results=numResults), llm=llm, prompt=promptQuery)
    prompt = ChatPromptTemplate.from_template(ragPromptTemplate)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    results = chain.invoke(userInp)
    return results

#global variables for database connection and templates
phidbb3 = readChromadb(folderPath='phidbb3', model='phi3', collection_name='recipe')
promptRecipeQuery = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI chef assistant trained on a RAG process. Your task is to take in the users recipe name and then generate
    the ingredients need and directions to cook that recipe. Only include the title,ingredients and directions you find in the vector database. 
    Use the Follow Format to print the results
    Recipe Name: , Ingredients: , Directions:
    User question: {question}""",
)

promptRecommendQuery = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI chef assistant trained on a RAG process. Your task is to take in the users ingredients and then generate
    the Name of recipe with directions and ingredients needed to cook that recipe. Only include the title,ingredients and directions you find in the vector database. 
    Use the Follow Format to print the results
    Recipe Name: , Ingredients: , Directions:
    User question: {question}""",
)

promptHowQuery = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI chef assistant trained on a RAG process. Your task is to take in the users question about how something is supposed to be done and then generate
     how its supposed to be done with clear instructions .
    User question: {question}""",
)

template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

model = 'phi3'
numResults = 5 