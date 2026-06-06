from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch


def run_sentiment():
    classifier = pipeline("sentiment-analysis")
    text = input("Enter text for sentiment analysis: ").strip()
    result = classifier(text)
    print(result)


def run_bert():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    sentence = input("Enter a sentence for BERT: ").strip()
    inputs = tokenizer(sentence, return_tensors="pt")

    print("Tokenizer Output")
    print(f"input_ids : {inputs['input_ids']}")
    print(f"tokens : {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    print(f"shape : {inputs['input_ids'].shape}")

    with torch.no_grad():
        outputs = model(**inputs)

    print("\n=== BERT Raw Output ===")
    print(f"last_hidden_state shape : {outputs.last_hidden_state.shape}")
    print(f"CLS vector shape        : {outputs.last_hidden_state[0, 0, :].shape}")
    print(f"CLS vector (first 8)    : {outputs.last_hidden_state[0, 0, :8].numpy().round(3)}")


def run_gpt2():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    prompt = input("Enter a prompt for GPT-2 to continue: ").strip()
    inputs = tokenizer(prompt, return_tensors="pt")

    print("Tokenizer Output")
    print(f"tokens : {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    print(f"shape : {inputs['input_ids'].shape}")

    output_ids = model.generate(inputs["input_ids"], max_new_tokens=20, do_sample=False)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== GPT-2 Generated Text ===")
    print(generated)

    with torch.no_grad():
        outputs = model(**inputs)

    print("\n=== GPT-2 Raw Output ===")
    print(f"logits shape : {outputs.logits.shape}")
    print(f"last position logits (first 8) : {outputs.logits[0, -1, :8].numpy().round(3)}")


OPTIONS = {
    "1": ("Sentiment Analysis (pipeline)", run_sentiment),
    "2": ("BERT — encoder embeddings", run_bert),
    "3": ("GPT-2 — causal text generation", run_gpt2),
}

print("Choose a demo to run:")
for key, (label, _) in OPTIONS.items():
    print(f"  {key}. {label}")

choice = input("\nEnter 1, 2, or 3: ").strip()

if choice in OPTIONS:
    label, fn = OPTIONS[choice]
    print(f"\n--- Running: {label} ---\n")
    fn()
else:
    print("Invalid choice.")
