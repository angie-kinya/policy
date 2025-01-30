from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

# Load the RAG tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

# Initialize the retriever with a pre-built index (e.g., "wiki_dpr")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="wiki_dpr",  # Use a pre-built index
    passages_path=None,  # Not needed for pre-built indices
    index_path=None,  # Not needed for pre-built indices
)

# Load the RAG model
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

# Query the RAG model
query = "What are the key requirements for data breach notification under the GDPR?"
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