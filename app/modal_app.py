import modal
import torch
import time
import logging
from fastapi import FastAPI, HTTPException, Body, Request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

# -------------------------
# Config
# -------------------------
MODEL_ID = "CraneAILabs/ganda-gemma-1b"
SUNFLOWER_MODEL_ID = "Sunbird/Sunflower-32B-4bit-fp4-bnb"
GPU = os.environ.get("GPU", "A100")
SCALEDOWN = 120
HF_SECRET_NAME = os.environ.get("HF_SECRET_NAME", "huggingface")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("luganda-api")

# -------------------------
# Modal Image
# -------------------------
image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "uvicorn"
    )
)

app = modal.App("luganda-asr-corrector")
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)


# -------------------------
# Helpers
# -------------------------
def validate_text(text, name):
    if text is None:
        raise HTTPException(422, f"{name} required")

    text = text.strip()
    if len(text) < 2:
        raise HTTPException(422, f"{name} too short")

    return text


def clean_output(text):
    if not text:
        return ""
    text = text.strip()
    if text.lower() in ["none", "null"]:
        return ""
    return text


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr_row[j - 1] + 1
            delete = prev_row[j] + 1
            replace = prev_row[j - 1] + (ca != cb)
            curr_row.append(min(ins, delete, replace))
        prev_row = curr_row
    return prev_row[-1]


# -------------------------
# Modal Class (FIXED)
# -------------------------
@app.cls(gpu=GPU, scaledown_window=SCALEDOWN, image=image)
class LugandaGemma:

    @modal.enter()
    def load(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        if self.device == "cuda":
            self.model = self.model.to("cuda")
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

    def _generate(self, prompt: str):
        return self.generator(
            prompt,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

    def _correct(self, text: str):
        prompt = f"""
You are a professional Luganda language editor.

Your task:
- Fix all ASR (speech-to-text) errors in the sentence below.
- Always rewrite the sentence, even if it looks correct.
- Improve spelling, grammar, and naturalness.
- If the sentence is not perfect Luganda, rewrite it in perfect Luganda.
- If it is already perfect, rewrite it identically.
- NEVER return the exact same input unless it is already perfect Luganda.

Sentence: {text}
Corrected:
"""
        out = self.generator(
            prompt,
            max_new_tokens=128,
            temperature=0.5,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]
        corrected = out.split("Corrected:")[-1].strip()
        return clean_output(corrected) or text

    @modal.method()
    def correct_transcript(self, transcript: str):
        corrected = self._correct(transcript)
        return {
            "corrected_text": corrected,
            "confidence": 1.0 if corrected == transcript else 0.7
        }

    @modal.method()
    def batch_correct(self, transcripts: list[str]):
        results = []
        for t in transcripts:
            corrected = self._correct(t)
            results.append({
                "corrected_text": corrected,
                "confidence": 1.0 if corrected == t else 0.7
            })
        return results

    @modal.method()
    def translate(self, text: str):
        prompt = f"Translate to Luganda:\n{text}\nTranslation:"
        result = self._generate(prompt)

        translation = result.split("Translation:")[-1].strip()
        translation = clean_output(translation) or "Nkyusizza tekisobose"

        return {"translation": translation}

    @modal.method()
    def converse(self, prompt: str):
        result = self._generate(prompt)
        response = clean_output(result) or "Nsonyiwa, sisobola kuddamu bulungi."
        return {"response": response}


@app.cls(gpu=GPU, scaledown_window=SCALEDOWN, image=image, secrets=[hf_secret])
class Sunflower32B:

    @modal.enter()
    def load(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "Missing Hugging Face token. Configure Modal secret with HF_TOKEN for gated model access."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            SUNFLOWER_MODEL_ID,
            trust_remote_code=True,
            token=hf_token,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            SUNFLOWER_MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )

    def _chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 256,
        temperature: float | None = 0.3,
        do_sample: bool = True,
    ):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "no_repeat_ngram_size": 5,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature is not None and do_sample:
            generation_kwargs["temperature"] = temperature

        outputs = self.model.generate(
            **inputs,
            **generation_kwargs,
        )
        completion_tokens = outputs[0][len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        return clean_output(response)

    @modal.method()
    def chat(self, prompt: str, system_message: str | None = None):
        sys_msg = system_message or (
            "You are Sunflower, a multilingual assistant for Ugandan languages made by Sunbird AI. "
            "Give concise and helpful answers."
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]
        response = self._chat(messages, max_new_tokens=512, temperature=0.4)
        return {"response": response or "Nsonyiwa, sisobola kuddamu bulungi."}

    @modal.method()
    def translate(self, text: str, target_language: str = "Luganda"):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional translator for Ugandan languages. "
                    "Translate faithfully and preserve meaning. "
                    "Return only the translated text with no explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Translate to {target_language}:\n{text}",
            },
        ]
        translation = self._chat(messages, max_new_tokens=512, temperature=0.2)
        return {"translation": translation or "Nkyusizza tekisobose"}

    @modal.method()
    def luganda_asr_correction(self, transcript: str):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Luganda ASR post-correction engine. "
                    "Your task is to correct spelling, punctuation, and grammar errors in Luganda transcripts only. "
                    "Hard rules: do not translate, do not summarize, do not paraphrase, do not change intent, "
                    "do not add or remove factual content. "
                    "Only make minimal edits where the transcript is clearly incorrect. "
                    "If a segment is already correct, keep it unchanged. "
                    "Output only the corrected Luganda transcript and nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Correct this Luganda ASR transcript using the rules above.\n"
                    f"Transcript: {transcript}"
                ),
            },
        ]
        corrected = self._chat(
            messages,
            max_new_tokens=512,
            temperature=None,
            do_sample=False,
        )
        corrected = corrected or transcript
        distance = levenshtein_distance(transcript, corrected)
        normalized_distance = distance / max(len(transcript), 1)
        return {
            "original_text": transcript,
            "corrected_text": corrected,
            "edit_distance": distance,
            "normalized_edit_distance": round(normalized_distance, 4),
        }


# -------------------------
# FastAPI
# -------------------------
fastapi_app = FastAPI(title="Luganda Gemma API")
model = LugandaGemma()
sunflower_model = Sunflower32B()


@fastapi_app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    logger.info(f"{request.method} {request.url.path} {response.status_code} {duration:.3f}s")
    response.headers["X-Process-Time"] = str(duration)
    return response


@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "models": {
            "current": MODEL_ID,
            "sunflower": SUNFLOWER_MODEL_ID,
        },
    }


@fastapi_app.post("/correct_transcript")
def correct(payload: dict = Body(...)):
    text = validate_text(payload.get("transcript"), "transcript")
    return model.correct_transcript.remote(text)


@fastapi_app.post("/batch_correct")
def batch(payload: dict = Body(...)):
    texts = payload.get("transcripts")

    if not isinstance(texts, list) or len(texts) == 0:
        raise HTTPException(422, "'transcripts' must be a list")

    cleaned = [validate_text(t, "transcript") for t in texts]
    return model.batch_correct.remote(cleaned)


@fastapi_app.post("/translate")
def translate(payload: dict = Body(...)):
    text = validate_text(payload.get("text"), "text")
    return model.translate.remote(text)


@fastapi_app.post("/converse")
def converse(payload: dict = Body(...)):
    prompt = validate_text(payload.get("prompt"), "prompt")
    return model.converse.remote(prompt)


@fastapi_app.post("/sunflower/chat")
def sunflower_chat(payload: dict = Body(...)):
    prompt = validate_text(payload.get("prompt"), "prompt")
    system_message = payload.get("system_message")
    return sunflower_model.chat.remote(prompt, system_message)


@fastapi_app.post("/sunflower/translate")
def sunflower_translate(payload: dict = Body(...)):
    text = validate_text(payload.get("text"), "text")
    target_language = (payload.get("target_language") or "Luganda").strip()
    return sunflower_model.translate.remote(text, target_language)


@fastapi_app.post("/sunflower/luganda_asr_correction")
def sunflower_luganda_asr_correction(payload: dict = Body(...)):
    transcript = validate_text(payload.get("transcript"), "transcript")
    return sunflower_model.luganda_asr_correction.remote(transcript)


# -------------------------
# Entrypoint
# -------------------------
@app.function(image=image, gpu=GPU, timeout=900)
@modal.asgi_app()
def fastapi_entrypoint():
    return fastapi_app