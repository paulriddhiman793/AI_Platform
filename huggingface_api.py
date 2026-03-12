"""
Hugging Face Inference API - Example Usage
------------------------------------------
Uses huggingface_hub InferenceClient (recommended as of 2025).
The old api-inference.huggingface.co endpoint is fully deprecated.
hf-inference handles CPU-friendly NLP tasks (BERT-class models).
LLM text generation uses InferenceClient with provider="auto".

Requirements:
    pip install huggingface_hub python-dotenv

Setup:
    Add your HF token to a .env file in the same directory:
        HUGGINGFACE_API_KEY=hf_your_token_here

Get your free API token at: https://huggingface.co/settings/tokens
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()

HF_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_API_TOKEN:
    raise EnvironmentError("HUGGINGFACE_API_KEY not found in .env file.")

# Standard InferenceClient for classic NLP tasks (BERT-class models)
client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)

# Provider is specified directly in the model string as "model_id:provider"
llm_client = InferenceClient(api_key=HF_API_TOKEN)


# ── Task functions ────────────────────────────────────────────────────────────

def fix_code(broken_code: str, language: str = "python",
             model: str = "meta-llama/Llama-3.3-70B-Instruct:groq") -> str:
    """Fix syntax errors in code using Llama-3.3-70B via Groq (free). Provider is embedded in the model string as "model:provider".
    Returns the corrected code with a brief summary of changes made."""
    completion = llm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are an expert {language} developer and code reviewer. "
                    f"Fix ALL syntax errors, typos, and bugs in the {language} code provided. "
                    "Rules:\n"
                    "1. Return the fully corrected code inside a ```code block```.\n"
                    "2. After the code block, add a short '### Changes' section listing what you fixed.\n"
                    "3. Do NOT change logic or functionality — only fix errors.\n"
                    "4. If the code is already correct, say so."
                )
            },
            {
                "role": "user",
                "content": f"Fix the following {language} code:\n\n```{language}\n{broken_code}\n```"
            }
        ],
        max_tokens=1000,
    )
    return completion.choices[0].message.content


def text_classification(text: str, model: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english") -> list:
    """Sentiment / text classification."""
    result = client.text_classification(text, model=model)
    return result


def zero_shot_classification(text: str, candidate_labels: list[str],
                              model: str = "facebook/bart-large-mnli") -> list:
    """Zero-shot classification into provided labels."""
    result = client.zero_shot_classification(text, candidate_labels=candidate_labels, model=model)
    # Returns list of ZeroShotClassificationOutputElement(label, score)
    return sorted(result, key=lambda x: x.score, reverse=True)


def summarization(text: str, model: str = "facebook/bart-large-cnn") -> str:
    """Summarize a long piece of text."""
    result = client.summarization(text, model=model)
    return result.summary_text


def question_answering(question: str, context: str,
                        model: str = "deepset/roberta-base-squad2") -> dict:
    """Extract an answer from context given a question."""
    result = client.question_answering(
        question=question, context=context, model=model
    )
    return {"answer": result.answer, "score": result.score}


def translation(text: str, model: str = "Helsinki-NLP/opus-mt-en-fr") -> str:
    """Translate text (default: English → French)."""
    result = client.translation(text, model=model)
    return result.translation_text


def fill_mask(text: str, model: str = "google-bert/bert-base-uncased") -> list:
    """Predict the word(s) that fill a [MASK] token."""
    result = client.fill_mask(text, model=model)
    return result  # list of FillMaskOutputElement


# ── Main demo ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Hugging Face Inference API Demo")
    print("=" * 60)

    # 1. Code Correction
    print("\n[1] Code Correction  (Llama-3.3-70B via Groq)")
    broken = """
def caluculate_sum(a, b)
    reslt = a ++ b
    print("Sum is: " + reslt)
    return reslt
"""
    print(f"    Broken code:{broken}")
    try:
        fixed = fix_code(broken, language="python")
        print(f"    → Output:\n")
        for line in fixed.strip().splitlines():
            print(f"       {line}")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    # 2. Sentiment Classification
    print("\n[2] Sentiment Classification")
    sample = "I absolutely loved this movie, it was fantastic!"
    print(f"    Text: \"{sample}\"")
    try:
        labels = text_classification(sample)
        for item in labels:
            print(f"    → {item.label}: {item.score:.4f}")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    # 3. Zero-Shot Classification
    print("\n[3] Zero-Shot Classification")
    news = "The central bank raised interest rates by 0.5% today."
    labels_list = ["finance", "sports", "politics", "technology"]
    print(f"    Text: \"{news}\"")
    print(f"    Labels: {labels_list}")
    try:
        result = zero_shot_classification(news, labels_list)
        for item in result:
            print(f"    → {item.label}: {item.score:.4f}")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    # 4. Summarization
    print("\n[4] Summarization")
    long_text = (
        "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical "
        "rainforest in the Amazon biome that covers most of the Amazon basin of South America. "
        "This basin encompasses 7,000,000 km² (2,700,000 sq mi), of which "
        "5,500,000 km² (2,100,000 sq mi) are covered by the rainforest. This region includes "
        "territory belonging to nine nations and 3,344 formally acknowledged indigenous territories."
    )
    print(f"    Input ({len(long_text)} chars) → summarizing...")
    try:
        summary = summarization(long_text)
        print(f"    → {summary}")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    # 5. Question Answering
    print("\n[5] Question Answering")
    context = "Python was created by Guido van Rossum and first released in 1991."
    question = "Who created Python?"
    print(f"    Context: \"{context}\"")
    print(f"    Question: \"{question}\"")
    try:
        answer = question_answering(question, context)
        print(f"    → Answer: {answer['answer']}  (score: {answer['score']:.4f})")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    # 6. Translation
    print("\n[6] Translation (EN → FR)")
    sentence = "Hello, how are you today?"
    print(f"    Input: \"{sentence}\"")
    try:
        translated = translation(sentence)
        print(f"    → {translated}")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    # 7. Fill Mask
    print("\n[7] Fill-Mask")
    masked = "Paris is the [MASK] of France."
    print(f"    Input: \"{masked}\"")
    try:
        predictions = fill_mask(masked)
        for pred in predictions[:3]:
            print(f"    → \"{pred.sequence}\"  (score: {pred.score:.4f})")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()