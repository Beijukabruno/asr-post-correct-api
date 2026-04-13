# Luganda Gemma API

A FastAPI service for Luganda ASR post-correction, English-to-Luganda translation, and Luganda conversation using the Ganda Gemma 1B model. Deployable on Modal with GPU support.

This API now also includes a second, comparison-ready model based on Sunbird Sunflower 32B 4-bit.

## Endpoints

- `POST /correct_transcript` — Corrects noisy Luganda ASR transcripts.
- `POST /translate` — Translates English text to Luganda.
- `POST /converse` — General Luganda conversation.
- `POST /sunflower/chat` — Chat with Sunflower 32B.
- `POST /sunflower/translate` — Translation with Sunflower 32B.
- `POST /sunflower/luganda_asr_correction` — Luganda ASR post-correction with strict correction-only prompting.

## Example Usage

### Correct Transcript

```json
{
  "transcript": "noisy Luganda ASR text here"
}
```

### Translate

```json
{
  "text": "Welcome to our school"
}
```

### Converse

```json
{
  "prompt": "Oli otya! Osobola okuntuyamba leero?"
}
```

### Sunflower Chat

```json
{
  "prompt": "Give me a short summary of this text in Luganda"
}
```

### Sunflower Translate

```json
{
  "text": "Sunbird AI builds practical AI systems for African languages.",
  "target_language": "Luganda"
}
```

### Sunflower Luganda ASR Correction

```json
{
  "transcript": "nze ngenda kusoma ebitabo ebyo enkya"
}
```

The Sunflower correction endpoint uses strict instructions to preserve meaning:

- No translation
- No summarization
- No paraphrasing
- Minimal edits only where text is clearly incorrect
- Leave already-correct segments unchanged
- Deterministic decoding for correction (`do_sample=false`) to reduce randomness

Example response shape:

```json
{
  "original_text": "nzee nggendaa kuuu somaa ebitabooo",
  "corrected_text": "Nze ngenda kusoma ebitabo.",
  "edit_distance": 13,
  "normalized_edit_distance": 0.4194,
}
```

## Quick cURL Tests

Set your deployed URL once:

```bash
BASE_URL="https://beijuka-cdli-hackathon-uganda--luganda-asr-corrector-fas-f890f3.modal.run"
```

1. Health check

```bash
curl -sS "$BASE_URL/health"
```

2. Current model ASR correction (`/correct_transcript`)

```bash
curl -sS -X POST "$BASE_URL/correct_transcript" \
  -H "Content-Type: application/json" \
  -d '{"transcript":"nze ndi musomesa wa luganda era njagala okuyamba abayizi"}'
```

3. Sunflower ASR correction (`/sunflower/luganda_asr_correction`)

```bash
curl -sS -X POST "$BASE_URL/sunflower/luganda_asr_correction" \
  -H "Content-Type: application/json" \
  -d '{"transcript":"nzee nggendaa kuuu somaa ebitabooo ebyo enkyyaa kubanga nninaa essuubii"}'
```

4. Sunflower translation (`/sunflower/translate`)

```bash
curl -sS -X POST "$BASE_URL/sunflower/translate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Sunbird AI builds practical AI systems for African languages.","target_language":"Luganda"}'
```

5. Sunflower chat (`/sunflower/chat`)

```bash
curl -sS -X POST "$BASE_URL/sunflower/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Mpa okunnyonnyola okumpi ku ngeri y okunoonyerezaamu ebyawandiikibwa mu Luganda."}'
```

## Limitations

- Language Output: Responds only in Luganda
- Context Length: Optimized for shorter conversational inputs
- Cultural Context: May not capture all nuances of Luganda culture
- Regional Variations: Trained on standard Luganda, may not reflect all dialects

## Technical Details

- Base Model: Google Gemma 3 1B Instruct
- Fine-tuning Method: Supervised fine-tuning on English-Luganda pairs
- Context Length: 2048 tokens
- Precision: 16-bit floating point
- Framework: Transformers (PyTorch)

### Comparison Model

- Base Model: Sunbird Sunflower 32B 4-bit FP4 (`Sunbird/Sunflower-32B-4bit-fp4-bnb`)
- Loading Mode: 4-bit quantized with BitsAndBytes
- Device Placement: `device_map="auto"`

## Deployment

- Choose GPU: `A100` (40GB) or `L40S`. For most NLP tasks, A100 is preferred for speed and memory.
- Sunflower is a gated model. Create or update a Modal secret named `huggingface` with your HF token in `HF_TOKEN`.
- You can also override the secret name using env var `HF_SECRET_NAME`.
- Deploy using Modal CLI:

```bash
modal secret create huggingface HF_TOKEN=<your_hf_token>
```

```bash
modal deploy app/modal_app.py
```

## Applications

- Offline English-Luganda translation
- Language learning and practice
- Culturally aware Luganda content generation
- Educational tools
- NLP research for Luganda
- Luganda content creation for media

---