

#!pip install -qU pix2text langchain jq tiktoken langchain_community langchain_chroma langchain-huggingface huggingface-hub sentence_transformers
#!pip uninstall onnxruntime -y
#!pip install -qU onnxruntime-gpu

from pix2text import Pix2Text
from pix2text.latex_ocr import *
import os


# Set environment variables to prevent thread affinity issues
os.environ["OMP_NUM_THREADS"] = "1"  # Set to limit threads
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure only one GPU is visible
os.environ["OMP_PROC_BIND"] = "false"  # Prevents issues with OpenMP thread binding



img_fp = '/scratch/courses0101/mdhandapani/.pdf'
p2t = Pix2Text.from_config()

"""## Page by page

"""

#page by page
from langchain_core.documents import Document
pages=[]
for i in range(3,34):
  doc = p2t.recognize_pdf(img_fp,table_as_image=False,page_numbers=[i])
  pages.append(Document(page_content=doc.to_markdown('Page'), metadata={'page':i}))

pages

"""# Langchain"""

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings, HuggingFaceHubEmbeddings
from langchain_huggingface import HuggingFacePipeline

import bs4, getpass, os, tiktoken
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate

# Load, chunk and index the contents of the blog.
#loader = TextLoader("/content/output/output.md")
#docs = loader.load()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
#num_tokens_from_string(question, "cl100k_base")
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)
#similarity = cosine_similarity(query_result, document_result)

model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
question = "What Ideal gas law?"
#document = pages

document = pages



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(document)
vectorstore = Chroma.from_documents(documents=splits, embedding=hf, persist_directory="/scratch/courses0101/mdhandapani/vector1")