import torch
from fastapi import FastAPI, HTTPException, Body
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = "CraneAILabs/ganda-gemma-1b"

app = FastAPI(title="Luganda ASR Post-Correction API")

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    if device == "cuda":
        model = model.to("cuda")
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
    )
    return generator

generator = load_model_and_tokenizer()

# Improved prompt engineering for post-ASR correction
CORRECTION_PROMPT = (
    "You are a Luganda language expert and your job is to correct errors in automatic speech recognition (ASR) transcripts. "
    "Given a possibly noisy Luganda transcript, return the corrected Luganda text only. "
    "Do not translate, do not explain, do not add extra text. "
    "If the input is already correct, return it unchanged.\n"
    "Transcript: {transcript}\nCorrected:"
)

@app.post("/correct_transcript")
def correct_transcript(payload: dict = Body(...)):
    transcript = (payload.get("transcript") or "").strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="'transcript' is required")
    prompt = CORRECTION_PROMPT.format(transcript=transcript)
    out = generator(prompt, max_length=128, temperature=0.2, do_sample=False)
    corrected = out[0]["generated_text"].split("Corrected:")[-1].strip()
    # Optionally, compute a simple confidence score (e.g., similarity)
    confidence = 1.0 if corrected == transcript else 0.7  # Placeholder logic
    return {"corrected_text": corrected, "confidence": confidence}
