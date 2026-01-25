from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='Fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='Fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='Fact_3', description='Fact 3 about the topic')
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me three interesting facts about {topic}.\n{format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)
prompt = template.format(topic='black hole')
result = model.invoke(prompt)
final_result = parser.parse(result)
print(final_result)

# In the latest LangChain, StructuredOutputParser was removed. RIP