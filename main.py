import os
import ast
import openai
import pandas as pd
import tiktoken
from scipy import spatial

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

CODE_DIR_PATH = "/home/rasdani/git/mp-eyetracking/src"


def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below snippets of code. If the snippets do not provide enough context, write "I need more context."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_snippet = f'\n\nSnippet of code:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_snippet + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_snippet
    return message + question


def ask(
    query: str,
    df: pd.DataFrame,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the snippets of code and assist in code generation."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

def embed_code():
    """Embeds all code in the code directory and saves it to a csv."""
    file_paths = [ f"{CODE_DIR_PATH}/{file_path}" for file_path in os.listdir(CODE_DIR_PATH) if file_path.endswith(".py")]
    df = pd.DataFrame(columns=["text", "embedding"])
    for file_path in file_paths:
        with open(file_path, "r") as f:
            file_text = f.read()
        print("Embedding ", file_path)
        file_embedding_response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=file_text,
        )
        file_embedding = file_embedding_response["data"][0]["embedding"]
        df = df.append({"text": file_text, "embedding": file_embedding}, ignore_index=True)
    df.to_csv("embeddings.csv")
    
def csv_to_embeddings(csv_path: str) -> pd.DataFrame:
    """Converts a csv of text and embeddings saved as strings to a dataframe."""
    df = pd.read_csv(csv_path)
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    return df

# embed_code()
df = csv_to_embeddings("embeddings.csv")

question = "What are the priors in the threshold model?"
answer = ask(question, df)
print(answer)