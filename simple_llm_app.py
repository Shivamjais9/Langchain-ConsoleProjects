from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# HuggingFace chat model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
)
parser = StrOutputParser()
chat_model = ChatHuggingFace(llm=llm)

# Prompt
prompt = PromptTemplate(
    template="Suggest a 5 jokes based on the following topic: {topic}",
    input_variables=["topic"],
)

# Runnable chain 
chain = prompt | chat_model | parser

topic = input("Enter a topic for the joke: ")

result = chain.invoke({"topic": topic})

print("Generated Joke:", result)
