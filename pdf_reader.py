from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
)
# Load the PDF or ddocument file:
loader = TextLoader(r'C:\Users\Shiva\OneDrive\Desktop\langchain_models\Langchain_Output_Parsers\Langchain_Runnable\sample.txt')
documents = loader.load()

# Spliting the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert the text into embeddings and store in FAISS.
vectorstore = FAISS.from_documents(docs, embeddings)

# Create a retriever from the vectorstore
retriever = vectorstore.as_retriever()

query = input("Enter your query here: ")

retrieved_docs = retriever.get_relevant_documents(query)

# Combine the retrieved documents into a single context string
retrieved_text  =  '\n'.join([doc.page_content for doc in retrieved_docs])
 
# Initialize the chat model
chat_model = ChatHuggingFace(llm=llm)

# Manually Pass Retrieved Context to the LLM
prompt = f"You are a helpful AI assistant. Use the following context to answer the question: {query}\n\n{retrieved_text}"
answer = chat_model.invoke(prompt)


# Print the answer
print('Answer:', answer)
