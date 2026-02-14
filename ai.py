from dotenv import load_dotenv
import chromadb, os
from chromadb.utils import embedding_functions
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("API_KEY")
client = OpenAI(api_key=API_KEY)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name="text-embedding-3-small"
)

chromadb_client = chromadb.PersistentClient(path='./file_data')
collection = chromadb_client.get_or_create_collection(
    name="pdf_data",
    embedding_function=openai_ef # type: ignore
)

def chat_with_file(prompt : str):
    try:
        results = collection.query(
            query_texts=[prompt],
            n_results=5
        )
        context = "\n\n".join(results["documents"][0]) # type: ignore
        prompt = f"""
        You are an expert document analyst. Use the provided context to answer the question.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {prompt}
        
        INSTRUCTIONS:
        - If the answer isn't in the context, look for clues or summarize what IS there.
        - Mention if the document seems to be about a specific topic (like Rocket Science or Physics).
        - If you truly cannot find it, explain WHAT kind of information is in the context instead.
        - If answer seems universal or out of context, answer it normally without using the context.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
    

def simple_chat(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content