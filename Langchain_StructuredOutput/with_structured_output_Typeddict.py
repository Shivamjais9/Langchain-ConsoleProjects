from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import json

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=150
)

model = ChatHuggingFace(llm=llm)

prompt = """
Return ONLY valid JSON.
Keys:
- summary: string
- sentiment: positive | negative | neutral

Text:
The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove.
Also, the UI looks outdated compared to other brands.
"""

response = model.invoke(prompt)

# HF returns text → parse manually
result = json.loads(response.content)

print(result)
print(result['summary'])
print(result['sentiment'])