import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from keybert import KeyBERT

# Create the storage folder
if not os.path.exists("models"):
    os.makedirs("models")

print("Step 1: Summarizer (Already done, but checking...)")
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
model.save_pretrained("./models/t5-summarizer")
tokenizer.save_pretrained("./models/t5-summarizer")

print("Step 2: Downloading the Keyword extractor...")
# This is the fix: we save the underlying embedding model
kw_model = KeyBERT(model='all-MiniLM-L6-v2')
kw_model.model.embedding_model.save("./models/keybert")

print("--- FINISHED SUCCESSFULLY! ---")