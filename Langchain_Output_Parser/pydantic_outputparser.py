from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt=18, lt=20, description="Age of the person (must be 19)")
    city: str = Field(description="City of the person")
    mobile: str = Field(pattern=r"^\d{10}$", description="10 digit mobile number")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me name, age, city and mobile number of a 19-year-old girl from Bangalore.\n{format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)
prompt = template.invoke({'place':'Indian'})
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)