from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Chat_template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer suppport agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])
chat_history = []
# Load chat history
with open('Langchain_Prompt/chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# Create prompt
prompt =chat_template.invoke({'chat_history': chat_history, 'query':'Where is my refund?'})

print(prompt)