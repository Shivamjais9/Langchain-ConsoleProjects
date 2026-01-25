from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# ---------------- LLM ----------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)

chat_model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# ---------------- Sentiment Schema ----------------
class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        description="Give the sentiment of feedback text"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# ---------------- Classifier Prompt ----------------
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text as "
        "Positive, Negative or Neutral.\n"
        "{feedback}\n"
        "{format_instruction}"
    ),
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model | parser2

# ---------------- Response Prompts ----------------
prompt_positive = PromptTemplate(
    template="Write an appropriate response to this Positive feedback:\n{feedback}",
    input_variables=["feedback"],
)

prompt_negative = PromptTemplate(
    template="Write an appropriate response to this Negative feedback:\n{feedback}",
    input_variables=["feedback"],
)

prompt_neutral = PromptTemplate(
    template="Write an appropriate response to this Neutral feedback:\n{feedback}",
    input_variables=["feedback"],
)

# ---------------- Attach feedback back ----------------
attach_feedback = RunnableLambda(
    lambda x: {
        "sentiment": x.sentiment,
        "feedback": feedback_text
    }
)

# ---------------- Branch Chain ----------------
branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "Positive", prompt_positive | model | parser),
    (lambda x: x["sentiment"] == "Negative", prompt_negative | model | parser),
    (lambda x: x["sentiment"] == "Neutral",  prompt_neutral  | model | parser),
    RunnableLambda(lambda _: "Could not find sentiment"),
)

# ---------------- Final Chain ----------------
def run_chain(feedback_text: str):
    sentiment = classifier_chain.invoke({"feedback": feedback_text})
    data = {
        "sentiment": sentiment.sentiment,
        "feedback": feedback_text
    }
    return branch_chain.invoke(data)

# ---------------- Test ----------------
result = run_chain("The result was acceptable and matched what was expected")
print(result)
