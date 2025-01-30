from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import Dataset
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# LOAD AND PREPROCESS DOCUMENTS
# Load the PDF file
loader = PyPDFLoader("guide-to-the-general-data-protection-regulation-gdpr-1-0.pdf")
documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Extract text content from the documents
text_list = [text.page_content for text in texts]

# CREATE EMBEDDINGS
# Load a pre-trained embeddings model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(text_list)  # Generate embeddings for the text chunks

# BUILD A DATASET WITH REQUIRED COLUMNS
# Create a dataset with 'title', 'text', and 'embeddings' columns
dataset_dict = {
    "title": [f"Document {i+1}" for i in range(len(text_list))],  # Add dummy titles
    "text": text_list,  # Text content
    "embeddings": embeddings.tolist(),  # Embeddings as lists
}

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict(dataset_dict)

# BUILD A FAISS INDEX
dimension = embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatIP(dimension)  # Inner product index for similarity search
index.add(np.array(embeddings))  # Add embeddings to the index

# Save the dataset and FAISS index to disk
dataset_path = "custom_dataset"
index_path = "custom_index"

dataset.save_to_disk(dataset_path)  # Save the dataset
faiss.write_index(index, index_path)  # Save the FAISS index

# IMPLEMENT THE RAG PIPELINE
# Load the RAG tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

# Initialize the retriever with the custom dataset and index
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",
    passages_path=dataset_path,  # Path to the saved dataset
    index_path=index_path,  # Path to the saved FAISS index
)

# Load the RAG model
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

# Query the RAG model
query = "What is GDPR?"
input_dict = tokenizer(query, return_tensors="pt")  # Tokenize the query

# Debugging: Check the shape and type of input_ids
input_ids = input_dict["input_ids"]
print("Input IDs shape:", input_ids.shape)  # Should be (1, sequence_length)
print("Input IDs:", input_ids)

# Ensure the model is in evaluation mode
model.eval()

# Generate the answer
with torch.no_grad():  # Disable gradient calculation for inference
    try:
        generated_ids = model.generate(input_ids=input_ids)
        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Answer:", answer)
    except Exception as e:
        print("Error during generation:", e)