from transformers import T5Tokenizer, T5ForConditionalGeneration

# ----------------------------
# STEP 1: Put your snapshot path here
# ----------------------------

model_path = "/Users/koditejasvi/.cache/huggingface/hub/models--google--flan-t5-base/snapshots/37f520fa929c961707657b28798b30c003dd100b"

# IMPORTANT:
# Replace the long number above with YOUR snapshot folder name if different.

# ----------------------------
# STEP 2: Load model OFFLINE
# ----------------------------

tokenizer = T5Tokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

model = T5ForConditionalGeneration.from_pretrained(
    model_path,
    local_files_only=True
)

# ----------------------------
# STEP 3: Input text
# ----------------------------

text = """
Artificial Intelligence is transforming industries by automating tasks,
improving efficiency, and enabling data-driven decision-making.
Machine learning models analyze large amounts of data and generate insights.
"""

input_text = "Summarize: " + text

# ----------------------------
# STEP 4: Tokenize
# ----------------------------

inputs = tokenizer(
    input_text,
    return_tensors="pt",
    max_length=512,
    truncation=True
)

# ----------------------------
# STEP 5: Generate summary
# ----------------------------

outputs = model.generate(
    inputs["input_ids"],
    max_length=60,
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Summary:\n")
print(summary)

