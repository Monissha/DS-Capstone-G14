from pix2text import Pix2Text
from pix2text.latex_ocr import *
import os
import torch

# Set environment variables to prevent thread affinity issues
os.environ["OMP_NUM_THREADS"] = "1"  # Set to limit threads
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure only one GPU is visible
os.environ["OMP_PROC_BIND"] = "false"  # Prevent issues with OpenMP thread binding

img_fp = '/scratch/courses0101/mdhandapani/mccabe7(revised).pdf'
p2t = Pix2Text.from_config()

# Page by page
from langchain_core.documents import Document
pages = []
for i in range(0, 1139):
    doc = p2t.recognize_pdf(img_fp, table_as_image=False, page_numbers=[i])
    pages.append(Document(page_content=doc.to_markdown('Page'), metadata={'page': i}))

# Langchain imports
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the HuggingFace model
model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)

# Check the number of splits
print(f"Number of document splits: {len(splits)}")

# Ensure that IDs are generated correctly
ids = [f"doc_{i}" for i in range(len(splits))]
print(f"Generated IDs: {ids[:10]}...")  # Check the first 10 IDs to ensure they are generated correctly

# Create the vectorstore and pass the IDs
vectorstore = Chroma.from_documents(
    documents=splits, embedding=hf, ids=ids, persist_directory="/scratch/courses0101/mdhandapani/vector1"
)
