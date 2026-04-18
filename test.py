from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load from local cache ONLY
tokenizer = T5Tokenizer.from_pretrained(
    "google/flan-t5-base",
    local_files_only=True
)

model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-base",
    local_files_only=True
)

text = """
Artificial Intelligence is transforming industries by automating tasks,
improving efficiency, and enabling data-driven decision-making.
Machine learning models analyze large amounts of data and generate insights.
"""

input_text = "Summarize: " + text

inputs = tokenizer(
    input_text,
    return_tensors="pt",
    max_length=512,
    truncation=True
)

outputs = model.generate(
    inputs["input_ids"],
    max_length=60,
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Summary:\n")
print(summary)
