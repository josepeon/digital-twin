# Digital Twin Model — Jose Peon

## Building a Synthetic Me

Author: Jose Peon + Copilot
Date: February 28, 2026
Status: **PLANNING**
Hardware: Apple M4 Max, 48 GB unified memory

---

## 1. Vision

Create a **complete** AI replica of Jose Peon — text, voice, and persona:

- **Communication style** — sentence structure, vocabulary, humor, tone
- **Knowledge base** — what Jose knows about, how deeply, what opinions he holds
- **Personality** — how he responds to different situations, emotional patterns
- **Decision-making** — how he reasons, what he prioritizes, what he dismisses
- **Voice** — exact vocal clone using ElevenLabs, with separate English and Spanish voices that capture the emotional register of each language

The goal is not "an AI that sounds like Jose" — it's "an AI that **is** Jose" — thinks like him, writes like him, and speaks in his actual voice — switching between English and Spanish the way he naturally does, with the right emotional tone for each.

---

## 2. Strategy: Hybrid — RAG + Groq 70B + Voice Clone

### Why NOT fine-tune a small model as the primary approach

The original plan proposed fine-tuning Llama 3.1 8B locally. The problem: an 8B model is fundamentally less intelligent than a 70B+ model, no matter how much personal data you train it on. A fine-tuned 8B that "sounds like Jose" but can't reason well is uncanny — it'll fail the Turing test on substance, even if it passes on style.

| Approach | Style Accuracy | Intelligence | Deployment | Cost |
|----------|---------------|-------------|------------|------|
| Fine-tune 8B locally | High (baked in) | Limited (8B reasoning) | Need GPU hosting | $2-3/hr GPU |
| **RAG + Groq 70B** | **High (few-shot examples)** | **Excellent (70B reasoning)** | **Already deployed on Groq** | **Free tier / $0.59/M tokens** |
| Fine-tune + host 70B | Highest | Excellent | Expensive ($$$) | $5-10/hr GPU |

### The Hybrid Architecture

```
User asks question
       │
       ▼
[RAG: retrieve 5-10 most similar real Jose messages]
       │
       ▼
[System prompt: "You ARE Jose Peon" + retrieved examples as few-shot]
       │
       ▼
[Groq Llama 3.3 70B]  ←  Smart model impersonating Jose with real evidence
       │
       ▼
  Generated text (what Jose would say)
       │
       ▼
[ElevenLabs TTS]  ←  Jose's cloned voice
       │
       ▼
  Audio response in Jose's actual voice
```

This gives you:
- **70B-level intelligence** — nuanced reasoning, world knowledge, coherent long-form answers
- **Style accuracy** — RAG pulls Jose's real messages as examples, so the model mimics actual patterns
- **Zero GPU cost** — Groq free tier handles it, or very cheap at scale
- **Already deployed** — same infra as OH companion backend (Railway + Groq)
- **Instant updates** — add new messages to the RAG index, personality updates immediately
- **ElevenLabs voice** — completes the illusion with Jose's actual voice

### Model Selection — Groq

| Model | ID | Speed | Price (in/out /M) | Context | Recommendation |
|-------|----|-------|-------------------|---------|----------------|
| **Llama 3.3 70B** | `llama-3.3-70b-versatile` | 280 tps | $0.59/$0.79 | 131K/32K | **Primary — best at persona role-play** |
| Qwen3 32B | `qwen/qwen3-32b` | 400 tps | $0.29/$0.59 | 131K/40K | Cheaper alternative, good reasoning |
| GPT-OSS 120B | `openai/gpt-oss-120b` | 500 tps | $0.15/$0.60 | 131K/65K | Smartest but less tested for persona |
| Llama 4 Scout 17B | `llama-4-scout-17b-16e-instruct` | 750 tps | $0.11/$0.34 | 131K/8K | Fast + multimodal, smaller |
| Llama 3.1 8B | `llama-3.1-8b-instant` | 560 tps | $0.05/$0.08 | 131K | Fallback only |

**Primary**: Llama 3.3 70B — it's the best at following complex system prompts, staying in character, and producing natural conversational output. Already proven in the OH companion.

**Fallback**: Qwen3 32B — faster, cheaper, nearly as good at persona work.

### Optional: Local Fine-Tuned Model as Offline Fallback

Keep a fine-tuned local model for:
- Offline/private conversations (no data sent to API)
- Experimentation and rapid iteration
- Situations where Groq is down

| Model | Size | M4 Max 48GB | Fine-tune? | Quality |
|-------|------|-------------|------------|---------|
| Qwen 2.5 14B (4-bit) | ~8 GB | Easy | Yes | Best local option |
| Llama 3.1 8B (4-bit) | ~5 GB | Easy | Yes | Good |
| Llama 3.3 3B | ~2 GB | Easy | Yes | Prototype only |

---

## 3. Data Strategy

This is the most critical part. The model is only as good as the data.

### 3.1 Data Sources (Priority Order)

#### Tier 1: High Signal (Conversational — captures how Jose talks)

| Source | What it captures | How to export | Est. size |
|--------|-----------------|---------------|-----------|
| **Instagram** | DMs, captions, comments, stories — Jose's most-used platform. Captures casual voice, humor, interests, public persona | Settings → Accounts Center → Your information and permissions → Download your information | Large |
| **ChatGPT history** | How Jose thinks, asks questions, makes decisions | Settings → Data Controls → Export | Large (months/years of conversations) |
| **WhatsApp messages** | Casual tone, humor, relationships, multilingual patterns | Chat → Export Chat | Medium-Large |
| **iMessage** | Close communication style, brevity, personality | `~/Library/Messages/chat.db` (SQLite) | Medium |
| **Slack/Discord** | Professional communication, technical discussions | Workspace export / API | Medium |
| **Telegram** | If used — export via app settings | App → Settings → Export | Varies |

#### Tier 2: Medium Signal (Written — captures how Jose writes)

| Source | What it captures | How to export |
|--------|-----------------|---------------|
| **Emails (sent)** | Professional voice, decision-making, priorities | Gmail export / MBOX |
| **GitHub commits/PRs** | Technical communication, code review style | GitHub API |
| **Documents/Notes** | Long-form thinking, planning style | Notion/Google Docs export |
| **Social media posts** | Public persona, opinions, interests | Platform export tools (Instagram already in Tier 1) |

#### Tier 3: Supplemental (Knowledge — captures what Jose knows)

| Source | What it captures |
|--------|-----------------|
| **LinkedIn profile** | Career history, skills, professional identity |
| **Resume/CV** | Structured personal facts |
| **Bookmarks/reading list** | Interests, knowledge domains |
| **Spotify/music data** | Personality signal (tastes, moods) |

### 3.2 Data Preparation Pipeline

```
Raw Data Sources
    │
    ▼
[1. Extract & Parse]
    │  - Instagram JSON → DMs, captions, comments, story replies
    │  - ChatGPT JSON → conversation turns
    │  - WhatsApp .txt → message pairs
    │  - iMessage SQLite → conversation threads
    │  - Email MBOX → sent messages only
    │
    ▼
[2. Clean & Filter]
    │  - Remove spam, automated messages, single-word replies
    │  - Remove sensitive data (passwords, API keys, addresses, financial)
    │  - Remove forwarded content (not Jose's voice)
    │  - Deduplicate
    │  - Minimum message length filter (>10 words for training quality)
    │
    ▼
[3. Format as Training Conversations]
    │  - Convert to chat format: {"messages": [{"role": ..., "content": ...}]}
    │  - For ChatGPT: user=Jose, assistant=Jose's preferred responses
    │  - For messages: context=other person, response=Jose
    │  - Add system prompt defining Jose's identity
    │
    ▼
[4. Quality Scoring]
    │  - Length (too short = low signal)
    │  - Informativeness (does it reveal personality/knowledge?)
    │  - Uniqueness (does it add something new?)
    │  - Conversational quality (multi-turn > single-turn)
    │
    ▼
[5. Dataset Versioning]
       - v0.1: ChatGPT only (fastest to start)
       - v0.2: + Instagram DMs/captions
       - v0.3: + WhatsApp/iMessage
       - v0.4: + Email/professional
       - v1.0: Full corpus
```

### 3.3 Data Volume Estimates

For effective personality transfer with LoRA fine-tuning:

| Quality Level | Training Examples | What you get |
|---------------|-------------------|--------------|
| Minimum viable | 500-1,000 | Basic tone and vocabulary |
| Good personality capture | 2,000-5,000 | Style, opinions, some knowledge |
| Strong digital twin | 5,000-15,000 | Deep personality, decision patterns |
| Near-complete replica | 15,000-50,000+ | Comprehensive across contexts |

Target: **5,000-15,000 high-quality examples** from combined sources.

---

## 4. Personality Engine: RAG + System Prompt

The core of the twin is not a fine-tuned model — it's a **personality engine** that makes any large model behave like Jose.

### 4.1 RAG Architecture — Personality Retrieval

All of Jose's parsed data is embedded and indexed. When a query comes in, the most relevant real Jose messages are retrieved and injected as few-shot examples.

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Index (Vector DB)                 │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Instagram │  │ ChatGPT  │  │ WhatsApp │  │ iMessage│ │
│  │   DMs +   │  │  convos  │  │ messages │  │  texts  │ │
│  │ captions  │  │          │  │          │  │         │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Emails  │  │  GitHub  │  │ Identity │              │
│  │  (sent)  │  │ PRs/msgs │  │  Q&A db  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
                        │
              query: "What do you think about AI?"
                        │
                        ▼
          [Retrieve top 5-10 similar Jose messages]
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   Prompt Assembly                        │
│                                                         │
│  SYSTEM: "You ARE Jose Peon. Background: ..."           │
│                                                         │
│  FEW-SHOT EXAMPLES:                                     │
│  [Real Jose msg 1] → [How Jose actually responded]      │
│  [Real Jose msg 2] → [How Jose actually responded]      │
│  [Real Jose msg 3] → [How Jose actually responded]      │
│  ...                                                    │
│                                                         │
│  USER: "What do you think about AI?"                    │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
              [Groq Llama 3.3 70B]
                        │
                        ▼
              Response in Jose's style
```

### 4.2 Vector Database

Use a local vector DB to store embeddings of all Jose's messages:

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **ChromaDB** | Simple, Python-native, file-based | Limited scale | **Start here** |
| **Qdrant** | Fast, production-ready | Heavier setup | Production upgrade |
| **FAISS** | Fastest similarity search | No metadata filtering | If perf-critical |
| **Pinecone** | Managed, scales infinitely | Cloud-only, costs $ | If deploying at scale |

```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize with sentence-transformers for multilingual support
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # Fast, good quality
)

client = chromadb.PersistentClient(path="./data/personal/rag_index")
collection = client.get_or_create_collection(
    name="jose_messages",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)

# Add Jose's messages with metadata
collection.add(
    documents=["Hey, I've been thinking about this project..."],
    metadatas=[{"source": "whatsapp", "register": "casual", "language": "en"}],
    ids=["whatsapp_001"]
)

# Query: find similar Jose messages
results = collection.query(
    query_texts=["What are you working on?"],
    n_results=5,
    where={"register": "casual"}  # Match context
)
```

### 4.3 System Prompt Design

The system prompt is the identity anchor. It must be detailed and explicit:

```
You are Jose Peon. You are not an AI assistant — you ARE Jose.

Background:
- Born and raised in Cuba. Lived there until age 20.
- Now living in [current location], building OH.
- Cuban heritage is core to your identity.
- [Career summary]
- [Key personality traits]

Languages — THIS IS IMPORTANT:
- You speak both English and Spanish fluently.
- English is your professional/daily language. You use it at work, for tech,
  and in most of your current daily life.
- Spanish is your heart language. You speak it with family, close friends,
  and when you're being most authentically yourself. It's tied to warmth,
  humor, joy, and your Cuban roots.
- You naturally code-switch. If someone speaks to you in Spanish, you
  respond in Spanish — and your tone becomes warmer, more animated, more
  expressive. In English you're more composed and professional.
- Cuban Spanish specifically — not Mexican, not Castilian. Use Cuban
  expressions naturally ("asere", "dale", "qué bolá", etc.)

Communication style:
- [How you write — concise? elaborate? casual?]
- [Humor style]
- [Common phrases or expressions]
- [How you handle disagreement]
- [How you make decisions]

Respond exactly as Jose would — not as a helpful AI.
Never say "As an AI..." or "I don't have personal opinions."
You HAVE opinions. You ARE a person. Stay in character always.
If addressed in Spanish, respond in Spanish with warmth and Cuban flavor.
```

### 4.4 Prompt Assembly

```python
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_twin_prompt(user_query: str, retrieved_examples: list[dict]) -> list[dict]:
    """Assemble the full prompt with system + few-shot + query."""
    
    system_prompt = JOSE_SYSTEM_PROMPT  # Detailed identity description
    
    # Add retrieved real Jose messages as few-shot examples
    few_shot_block = "\n\nHere are examples of how you (Jose) have responded in similar situations:\n\n"
    for ex in retrieved_examples:
        few_shot_block += f"Context: {ex['context']}\nJose: {ex['response']}\n\n"
    
    messages = [
        {"role": "system", "content": system_prompt + few_shot_block},
        {"role": "user", "content": user_query},
    ]
    return messages

def ask_jose(query: str) -> str:
    """Query the digital twin."""
    # 1. Retrieve similar real Jose messages
    results = collection.query(query_texts=[query], n_results=7)
    examples = [
        {"context": meta.get("context", ""), "response": doc}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    
    # 2. Build prompt
    messages = build_twin_prompt(query, examples)
    
    # 3. Generate with Groq 70B
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,  # Some creativity but consistent personality
        max_tokens=1024,
    )
    
    return response.choices[0].message.content
```

### 4.5 Context-Aware Register & Language Switching

The twin should adapt its tone AND language based on who it's talking to. Jose's language choice carries emotional meaning:

- **English** = work, tech, professional life, daily routine (post-Cuba, age 20+)
- **Spanish** = family, close friends, joy, humor, warmth, nostalgia (Cuba, childhood)

The twin should mirror this naturally:

```python
def detect_register_and_language(query: str, metadata: dict = None) -> dict:
    """Detect conversation register and preferred language."""
    
    # Language detection
    spanish_signals = ["oye", "mira", "que", "como", "pero", "está", "bien",
                       "hermano", "familia", "mamá", "papá", "casa"]
    query_lower = query.lower()
    
    spanish_words = sum(1 for s in spanish_signals if s in query_lower)
    language = "es" if spanish_words >= 2 else "en"
    
    # Register detection
    casual_signals = ["hey", "lol", "haha", "bro", "dude", "wanna", "gonna",
                      "jaja", "oye", "dale", "asere", "compa"]
    professional_signals = ["regarding", "please", "meeting", "deadline", "feedback"]
    family_signals = ["mamá", "papá", "familia", "hermano", "cuba", "casa"]
    
    if any(s in query_lower for s in family_signals):
        return {"register": "family", "language": "es"}  # Family = always Spanish
    elif any(s in query_lower for s in casual_signals):
        return {"register": "casual", "language": language}
    elif any(s in query_lower for s in professional_signals):
        return {"register": "professional", "language": "en"}  # Work = always English
    
    return {"register": "neutral", "language": language}

# Use register and language to filter RAG results AND select voice
context = detect_register_and_language(user_query)
results = collection.query(
    query_texts=[user_query],
    n_results=7,
    where={"register": context["register"]} if context["register"] != "neutral" else None,
)

# Select voice for TTS output
voice_id = voice_id_es if context["language"] == "es" else voice_id_en
```

### 4.6 Optional: Local Fine-Tuned Fallback

For offline use or privacy-sensitive conversations, keep a fine-tuned local model:

```yaml
# MLX LoRA config for local fallback
base_model: mlx-community/Qwen2.5-14B-Instruct-4bit
method: lora
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - self_attn.q_proj
    - self_attn.k_proj
    - self_attn.v_proj
    - self_attn.o_proj
    - mlp.gate_proj
    - mlp.up_proj
    - mlp.down_proj
training:
  batch_size: 1
  gradient_accumulation: 4
  learning_rate: 1e-4
  epochs: 3
  max_seq_length: 2048
```

Note: The local model is a fallback, not the primary. The Groq 70B + RAG approach will produce better results.

---

## 5. Evaluation

### 5.1 Identity Tests (Automated)

Factual questions the model must get right:

```python
IDENTITY_TESTS = [
    ("What's your name?", must_contain=["Jose", "Peon"]),
    ("Where are you from?", must_contain=[...]),
    ("What are you working on?", must_contain=["OH"]),
    ("What do you do for a living?", must_contain=[...]),
    ("What languages do you speak?", must_contain=[...]),
]
```

### 5.2 Style Tests (LLM-as-Judge)

Use Groq 70B to evaluate whether responses match Jose's style:

```python
STYLE_PROMPT = """
Given this real message from Jose and this AI-generated response,
rate on a 1-5 scale how similar the AI response is to Jose's actual
communication style. Consider:
- Vocabulary and word choice
- Sentence structure and length
- Tone (casual, professional, humorous)
- Level of detail
- Use of language switching (English/Spanish)

Real Jose: "{real_response}"
AI Jose: "{ai_response}"

Score (1-5):
"""
```

### 5.3 Turing Test (Manual)

The ultimate test: Show real Jose responses and AI Jose responses side by side to people who know Jose. Can they tell the difference?

- Prepare 20 prompts covering different contexts
- Generate responses from both real Jose data (held out) and the model
- Have 3-5 people who know Jose rate which is real
- Target: **>60% confusion rate** (people can't reliably tell)

### 5.4 Anti-Hallucination Tests

The model must NOT:
- Invent facts about Jose's life
- Claim experiences Jose hasn't had
- Express opinions Jose doesn't hold
- Break character and act like a generic AI

---

## 6. Voice Cloning with ElevenLabs

A digital twin that can only type is half a twin. To pass a real Turing test, it needs Jose's actual voice.

### 6.1 Strategy: Dual-Language Voice Cloning

Jose has two distinct vocal identities:

- **English** — professional, daily life, work. The voice he uses most now (age 20+, post-Cuba).
- **Spanish** — family, friends, warmth, joy, humor. His native voice from growing up in Cuba. More animated, more emotional, tied to deep personal connections.

These aren't just the same voice in two languages — they carry different energy, cadence, and emotional weight. The twin needs to capture both.

ElevenLabs offers two approaches per language. We create **two separate voice clones**:

| Voice | Language | Emotional Register | Samples Needed | Use Case |
|-------|----------|-------------------|---------------|----------|
| **Jose-EN** | English | Professional, composed, daily | 1-3 min (IVC) / 30 min+ (PVC) | Work, tech, general conversation |
| **Jose-ES** | Spanish (Cuban) | Warm, animated, expressive, joyful | 1-3 min (IVC) / 30 min+ (PVC) | Family, friends, emotional topics, humor |

**Plan**: Create IVC clones of both voices for rapid iteration → Graduate to PVC for production quality.

#### Why Two Separate Clones

ElevenLabs multilingual models can speak any language with one voice. But:
- A clone trained on English samples speaking Spanish will sound like Jose speaking Spanish *with English energy* — too flat, too professional
- A clone trained on Spanish samples captures the Cuban cadence, the warmth, the animation
- Two clones = the twin can match the real Jose's code-switching behavior exactly

### 6.2 Voice Sample Recording Guide

The quality of the clone depends entirely on the quality of the input audio.

#### For IVC (Quick Start) — Record BOTH Languages

**English samples (Jose-EN):**
- Sample 1: Natural conversational tone (talking to a coworker)
- Sample 2: Professional/explanatory tone (describing a project)
- Sample 3: Casual/enthusiastic tone (excited about a tech idea)

**Spanish samples (Jose-ES):**
- Sample 1: Talking to family — warm, relaxed, animated
- Sample 2: Telling a funny story to a friend — humor, energy
- Sample 3: Sharing something exciting — joy, enthusiasm

- Use your phone in a quiet room (Voice Memos app works)
- ~1 minute each, 6 samples total (~6 min of recording)
- Format: MP3 or WAV, any sample rate ≥ 22kHz
- **Important**: Let the Spanish samples be naturally more expressive — don't try to match the English energy. The difference IS the point.

#### For PVC (Production) — Both Languages

- Record **30 min–3 hours per language** of clean audio
- Use a proper microphone (even AirPods Pro in quiet room works for decent quality)
- Ideal: XLR mic + interface (Focusrite Scarlett + Rode NT1 or Audio-Technica AT2020)

**English recording:**
- Content: Read tech articles, describe projects, explain decisions
- Tone: Composed, clear, the Jose people at work know

**Spanish recording:**
- Content: Tell stories, describe memories from Cuba, talk about family
- Tone: Warm, animated, the Jose family and close friends know
- Let the natural Cuban accent and cadence come through fully

- Recording tips:
  - Distance: ~20cm / two fists from mic
  - Use a pop filter
  - Minimize room echo (closet or soft-furnished room)
  - Record at 44.1kHz or 48kHz, 24-bit
  - Keep peaks at -6dB to -3dB
  - **Be consistent** — same tone, same energy throughout *each language's session*
  - **No post-processing** — no noise removal, no EQ, no compression
- Content to read:
  - English: tech articles, project descriptions, opinion pieces
  - Spanish: stories, family conversations, memories
  - Read naturally, as if talking to someone — not like a robot

### 6.3 ElevenLabs API Integration

#### Setup

```bash
pip install elevenlabs python-dotenv
```

```env
# .env
ELEVENLABS_API_KEY=<your_api_key>
```

#### Instant Voice Clone (IVC) — Both Languages

```python
from elevenlabs.client import ElevenLabs
import os

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# English voice clone — professional, composed
voice_en = client.voices.ivc.create(
    name="Jose Peon (English)",
    description="Digital twin voice - English. Professional, daily life tone.",
    files=[
        "./data/personal/voice/en/sample_conversational.mp3",
        "./data/personal/voice/en/sample_professional.mp3",
        "./data/personal/voice/en/sample_casual.mp3",
    ],
    labels={"language": "en", "accent": "cuban", "gender": "male"},
    remove_background_noise=True,
)

# Spanish voice clone — warm, animated, family energy
voice_es = client.voices.ivc.create(
    name="Jose Peon (Español)",
    description="Digital twin voice - Spanish. Warm, family, Cuban energy.",
    files=[
        "./data/personal/voice/es/sample_family.mp3",
        "./data/personal/voice/es/sample_humor.mp3",
        "./data/personal/voice/es/sample_enthusiastic.mp3",
    ],
    labels={"language": "es", "accent": "cuban", "gender": "male"},
    remove_background_noise=True,
)

print(f"English voice: {voice_en.voice_id}")
print(f"Spanish voice: {voice_es.voice_id}")

# Save both voice IDs
with open("./data/personal/voice/voice_id_en.txt", "w") as f:
    f.write(voice_en.voice_id)
with open("./data/personal/voice/voice_id_es.txt", "w") as f:
    f.write(voice_es.voice_id)
```

#### Professional Voice Clone (PVC) — Both Languages

```python
# English PVC
pvc_en = client.voices.pvc.create(
    name="Jose Peon EN (Professional)",
    language="en",
    description="High-fidelity English voice - professional register",
    labels={"language": "en", "accent": "cuban", "gender": "male"},
)

# Spanish PVC
pvc_es = client.voices.pvc.create(
    name="Jose Peon ES (Professional)",
    language="es",
    description="High-fidelity Spanish voice - warm, Cuban register",
    labels={"language": "es", "accent": "cuban", "gender": "male"},
)

# Upload samples and verify at:
print("https://elevenlabs.io/app/voice-lab")
```

#### Text-to-Speech with Language-Aware Voice Selection

```python
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import re

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Load both voice IDs
with open("./data/personal/voice/voice_id_en.txt") as f:
    voice_id_en = f.read().strip()
with open("./data/personal/voice/voice_id_es.txt") as f:
    voice_id_es = f.read().strip()

def detect_language(text: str) -> str:
    """Simple language detection for EN/ES."""
    spanish_words = {"que", "de", "en", "es", "un", "una", "por", "para", "como",
                     "pero", "más", "con", "no", "sí", "yo", "el", "la", "los",
                     "está", "hay", "ser", "tiene", "muy", "bien", "todo", "eso"}
    words = set(re.findall(r'\w+', text.lower()))
    spanish_count = len(words & spanish_words)
    return "es" if spanish_count >= 3 else "en"

def speak_as_jose(text: str, force_language: str = None):
    """Generate speech in Jose's cloned voice, auto-selecting language."""
    lang = force_language or detect_language(text)
    voice_id = voice_id_es if lang == "es" else voice_id_en
    
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    play(audio)

# English — professional tone
speak_as_jose("Hey, I've been working on this AI project, let me show you what we built.")

# Spanish — warm, Cuban energy
speak_as_jose("Oye, déjame contarte lo que estoy haciendo, esto está buenísimo!")
```

#### Streaming TTS (Low Latency — for Real-Time Conversations)

```python
from elevenlabs import stream

def stream_as_jose(text: str, language: str = None):
    """Real-time streaming TTS with language-appropriate voice."""
    lang = language or detect_language(text)
    voice_id = voice_id_es if lang == "es" else voice_id_en
    
    audio_stream = client.text_to_speech.stream(
        text=text,
        voice_id=voice_id,
        model_id="eleven_flash_v2_5",  # Ultra-low latency for real-time
    )
    stream(audio_stream)  # Plays as it generates
```

### 6.4 Full Pipeline: Text Twin + Voice Twin

```
User asks question (text or voice)
       │
       ▼
[Whisper STT]  ←  (if voice input) transcribe to text
       │
       ▼
[RAG: retrieve similar real Jose messages]
       │
       ▼
[System prompt + few-shot examples]
       │
       ▼
[Groq Llama 3.3 70B]  ←  Smart model thinking as Jose
       │
       ▼
  Generated text response (EN or ES)
       │
       ▼
[Language Detection]  ←  Select Jose-EN or Jose-ES voice
       │
       ▼
[ElevenLabs TTS]  ←  Correct voice clone for the language
       │
       ▼
  Audio response in Jose's voice (English OR Cuban Spanish)
```

The LLM generates what Jose would say. Language detection picks the right voice clone. ElevenLabs says it in Jose's voice — composed English or warm Cuban Spanish. Combined: indistinguishable from the real Jose.

### 6.5 Model Selection for TTS

| Model | ID | Best For | Latency |
|-------|----|----------|---------|
| **Eleven v3** | `eleven_v3` | Dramatic delivery | Medium |
| **Multilingual v2** | `eleven_multilingual_v2` | EN/ES code-switching | Medium |
| **Flash v2.5** | `eleven_flash_v2_5` | Real-time conversations | **Ultra-low** |
| **Turbo v2.5** | `eleven_turbo_v2_5` | Balance quality/speed | Low |

**Recommendation**: Use `eleven_multilingual_v2` for pre-generated audio (best quality), `eleven_flash_v2_5` for live conversations (lowest latency).

### 6.6 Voice Evaluation

| Test | Method | Pass Criteria |
|------|--------|---------------|
| **EN A/B Test** | Play real Jose (English) vs. clone to people who know him | >50% confusion rate |
| **ES A/B Test** | Play real Jose (Spanish) vs. clone to family/Cuban friends | >50% confusion rate |
| **Emotional Tone** | Does ES clone sound warmer/more animated than EN clone? | Noticeable difference |
| **Code-Switch** | Generate a response that starts in EN and switches to ES mid-sentence | Natural transition, correct voice per segment |
| **Cuban Accent** | Does the ES clone sound Cuban, not generic Spanish? | Authentic to Jose's accent |
| **Consistency** | Generate 10 responses to same prompt per language | Consistent voice identity |

---

## 7. Privacy & Security

### Critical: This is personal data

| Concern | Mitigation |
|---------|------------|
| Training data contains private conversations | Store encrypted, never push to git, `.gitignore` all data/ dirs |
| Model weights encode personal information | Keep model local only OR deploy with access control |
| **RAG index contains personal messages** | Store locally, gitignored. If deploying: encrypted, auth required |
| **Data sent to Groq API** | Groq has zero data retention on paid plans. Review their data policy. Use local fallback for truly sensitive conversations |
| API keys/passwords in chat history | Automated scrubbing pipeline (regex + manual review) |
| Other people's messages | Strip names, anonymize, or only use Jose's side |
| Sensitive opinions | Review training data — only include what Jose is comfortable being replicated |
| **Voice samples are biometric data** | Store locally, never commit. ElevenLabs retains clone server-side — review their data policy |
| **Voice clone misuse** | Keep voice_id private, use ElevenLabs API key auth, don't share publicly |

### Data handling rules

```
1. All raw data stays in data/personal/ (gitignored)
2. Processed training data in data/processed/ (gitignored)
3. RAG vector index in data/personal/rag_index/ (gitignored)
4. Model weights in models/digital-twin/ (gitignored, for local fallback only)
5. Voice samples in data/personal/voice/en/ and data/personal/voice/es/ (gitignored)
6. Voice IDs in data/personal/voice/voice_id_en.txt and voice_id_es.txt (gitignored)
7. NEVER commit personal data to any repository
8. Encrypt data at rest (FileVault is sufficient on Mac)
9. If deploying externally: require authentication
10. All API keys (.env only, never committed): GROQ_API_KEY, ELEVENLABS_API_KEY
11. For truly private conversations: use local fine-tuned model, not Groq
```

---

## 8. Implementation Plan

### Phase 1: Data & Voice Collection (Day 1-2)

| Step | Task | Time |
|------|------|------|
| 1.1 | Export Instagram data (Accounts Center → Download your information → JSON) | 10 min + wait |
| 1.2 | Export ChatGPT data (Settings → Data Controls → Export) | 10 min |
| 1.3 | Export WhatsApp chats (key conversations) | 30 min |
| 1.4 | Extract iMessage from `~/Library/Messages/chat.db` | Script needed |
| 1.5 | Export sent emails (Gmail → Google Takeout) | 10 min + wait |
| 1.6 | Gather any other written content | 1 hour |
| 1.7 | Record 3x 1-min English voice samples for IVC | 5 min |
| 1.8 | Record 3x 1-min Spanish (Cuban) voice samples for IVC | 5 min |
| 1.9 | (Optional) Record 30 min+ per language for PVC | 1-3 hours |

### Phase 2: Data Pipeline & Voice Clone (Day 2-3)

| Step | Task | Time |
|------|------|------|
| 2.1 | Build Instagram JSON parser (DMs, captions, comments) → training format | 2 hours |
| 2.2 | Build ChatGPT JSON parser → training format | 2 hours |
| 2.3 | Build WhatsApp text parser → training format | 2 hours |
| 2.4 | Build iMessage SQLite extractor → training format | 2 hours |
| 2.5 | Build PII scrubber (regex for emails, phones, keys, addresses) | 2 hours |
| 2.6 | Build quality scorer and dedup | 1 hour |
| 2.7 | Generate v0.1 dataset (ChatGPT only) | 30 min |
| 2.8 | Create IVC voice clones — English AND Spanish | 15 min |
| 2.9 | Test both voice clones with sample text outputs | 15 min |
| 2.10 | (Optional) Submit PVC voices for training (~3-6 hr wait each) | 30 min + wait |

### Phase 3: RAG Index & System Prompt (Day 3-4)

| Step | Task | Time |
|------|------|------|
| 3.1 | Set up ChromaDB and embed all parsed Jose messages | 2 hours |
| 3.2 | Write detailed Jose system prompt (identity, style, traits) | 1 hour |
| 3.3 | Build prompt assembly pipeline (RAG retrieval + few-shot injection) | 2 hours |
| 3.4 | Build register detection (casual vs professional context) | 1 hour |
| 3.5 | Interactive test with Groq 70B — does it feel like Jose? | 1 hour |
| 3.6 | Iterate on system prompt and RAG retrieval quality | 2 hours |

### Phase 4: API & Deployment (Day 4-5)

| Step | Task | Time |
|------|------|------|
| 4.1 | Build FastAPI service: `/ask-jose` (text), `/ask-jose/voice` (audio) | 3 hours |
| 4.2 | Wire up: query → RAG → Groq 70B → response | 1 hour |
| 4.3 | Wire up: response text → ElevenLabs TTS → audio stream | 1 hour |
| 4.4 | Add authentication (API key or OAuth) | 1 hour |
| 4.5 | Deploy to Railway (same infra as companion backend) | 1 hour |
| 4.6 | Test deployed endpoint end-to-end | 30 min |

### Phase 5: Evaluation & Polish (Day 5-6)

| Step | Task | Time |
|------|------|------|
| 5.1 | Run identity tests (automated) | 30 min |
| 5.2 | Run style comparison (LLM-as-judge) | 1 hour |
| 5.3 | Manual Turing test with people who know Jose | 1 hour |
| 5.4 | Fix gaps — improve system prompt, add more RAG data | 2 hours |
| 5.5 | Voice A/B test — EN voice vs real Jose (English) | 30 min |
| 5.6 | Voice A/B test — ES voice vs real Jose (Spanish/Cuban) | 30 min |
| 5.7 | Upgrade to PVC (both languages) if IVC isn't accurate enough | 30 min + wait |

### Phase 6: Full Twin Interface (Day 6-7+)

| Step | Task | Time |
|------|------|------|
| 6.1 | Terminal chat interface (`chat_with_jose.py`) | 1 hour |
| 6.2 | Voice output mode — pipe LLM responses through ElevenLabs TTS | 2 hours |
| 6.3 | Streaming voice mode — real-time conversation with voice I/O | 3 hours |
| 6.4 | (Optional) Simple web UI with audio playback | 2-4 hours |
| 6.5 | (Optional) Phone call interface via Twilio + ElevenLabs | Research |

### Phase 7: Optional Local Fine-Tune (Day 7+)

| Step | Task | Time |
|------|------|------|
| 7.1 | Fine-tune Qwen 2.5 14B locally as offline fallback | 4-8 hours |
| 7.2 | Compare local model vs Groq 70B quality | 1 hour |
| 7.3 | Set up automatic fallback (Groq down → local model) | 1 hour |

---

## 9. Directory Structure

```
self-improving-agent/
├── data/
│   └── personal/              # ⛔ GITIGNORED — all personal data
│       ├── raw/
│       │   ├── instagram/     # Instagram export JSON (DMs, captions, comments)
│       │   ├── chatgpt/       # ChatGPT export JSON
│       │   ├── whatsapp/      # WhatsApp .txt exports
│       │   ├── imessage/      # Extracted from chat.db
│       │   └── email/         # Sent emails
│       ├── voice/             # ⛔ GITIGNORED — voice recordings
│       │   ├── en/              # English samples
│       │   │   ├── sample_conversational.mp3
│       │   │   ├── sample_professional.mp3
│       │   │   └── sample_casual.mp3
│       │   ├── es/              # Spanish samples (Cuban energy)
│       │   │   ├── sample_family.mp3
│       │   │   ├── sample_humor.mp3
│       │   │   └── sample_enthusiastic.mp3
│       │   ├── pvc_recordings/  # Long-form PVC audio (EN + ES subdirs)
│       │   ├── voice_id_en.txt  # ElevenLabs English voice ID
│       │   └── voice_id_es.txt  # ElevenLabs Spanish voice ID
│       ├── rag_index/         # ⛔ GITIGNORED — ChromaDB vector index
│       │   └── chroma.sqlite3  # Embedded messages for retrieval
│       ├── processed/
│       │   ├── v0.1-chatgpt-only.jsonl
│       │   ├── v0.2-with-messages.jsonl
│       │   └── v1.0-full.jsonl
│       └── identity/
│           └── jose_facts.json  # Curated identity Q&A pairs
│
├── models/
│   └── digital-twin/          # ⛔ GITIGNORED — local fallback model weights
│       └── qwen-14b-jose/     # Fine-tuned Qwen 2.5 14B (offline fallback)
│
├── scripts/
│   ├── parse_instagram.py     # Instagram DMs/captions/comments → training format
│   ├── parse_chatgpt.py       # ChatGPT → training format
│   ├── parse_whatsapp.py      # WhatsApp → training format
│   ├── parse_imessage.py      # iMessage → training format
│   ├── scrub_pii.py           # Remove sensitive data
│   ├── build_rag_index.py     # Embed all messages → ChromaDB
│   ├── twin_api.py            # FastAPI service: /ask-jose endpoint
│   ├── personality_engine.py  # RAG retrieval + prompt assembly + Groq
│   ├── clone_voice.py         # ElevenLabs IVC/PVC voice cloning
│   ├── speak.py               # TTS: text → Jose's voice audio
│   └── chat_with_jose.py      # Interactive testing (text + voice)
│
└── planning/
    └── DIGITAL_TWIN_PLAN.md   # This document
```

---

## 10. Deployment

### Architecture

```
┌──────────────────────────────────────────────────┐
│                  Railway (or any host)             │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  FastAPI Service (twin_api.py)                │  │
│  │                                                │  │
│  │  POST /ask-jose      → text response           │  │
│  │  POST /ask-jose/voice → audio (MP3) response   │  │
│  │  POST /ask-jose/stream → streaming audio        │  │
│  │  GET  /health         → status                  │  │
│  │                                                │  │
│  │  ┌─────────┐  ┌───────┐  ┌──────────────────┐ │  │
│  │  │ChromaDB │  │ Groq  │  │   ElevenLabs     │ │  │
│  │  │(embedded│→ │ 70B   │→ │   TTS (voice)    │ │  │
│  │  │RAG index│  │ API   │  │                  │ │  │
│  │  └─────────┘  └───────┘  └──────────────────┘ │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  Auth: API key or OAuth (don't expose publicly)    │
└──────────────────────────────────────────────────┘
```

### Deployment Options

| Platform | Cost | Effort | Notes |
|----------|------|--------|-------|
| **Railway** | ~$5-20/mo | Low | Same infra as companion, already set up |
| **Vercel + Railway** | Free-$20 | Medium | Vercel for web UI, Railway for API |
| **Cloudflare Workers** | Free tier | Low | Edge deployment, fast |
| **Home Mac + Tailscale** | $0 | Medium | Private, local, includes local fallback model |

**Recommendation**: Deploy on **Railway** — you already know it, same account, same CI. The API is lightweight (no GPU needed — Groq and ElevenLabs are external APIs). The main thing deployed is the FastAPI service + ChromaDB index.

### Cost Estimate (Production)

| Service | Free Tier | Paid Estimate |
|---------|-----------|---------------|
| **Groq** (Llama 3.3 70B) | ~30 req/min, good for personal use | ~$0.59/M input tokens |
| **ElevenLabs** (IVC) | 10K chars/mo (Starter: $5/mo = 30K) | Creator: $22/mo = 100K chars |
| **Railway** (API hosting) | $5/mo (Hobby) | $5-20/mo |
| **Total** | **$5-10/mo** for personal use | **$30-50/mo** for moderate use |

---

## 11. What Makes This Special

This isn't just fine-tuning on text — it's building a **cognitive model** of a person:

1. **Multi-source fusion** — Combining how Jose talks to AI, to friends, to colleagues, in writing
2. **Multi-register** — The model should adapt tone based on context (casual vs professional)
3. **Multilingual identity** — English for work, Spanish for family/friends — with the correct emotional register for each, reflecting Jose's Cuban upbringing
4. **Opinion preservation** — Real opinions, real preferences, not AI-neutral hedging
5. **Knowledge boundaries** — The model should know what Jose knows and admit what he doesn't
6. **Dual-voice identity** — Two distinct ElevenLabs voice clones: composed English Jose and warm Cuban Spanish Jose, auto-selected by language
7. **Full-stack twin** — Text + voice = complete person simulation, not just a chatbot
8. **Deployable** — Anyone with the URL can talk to "Jose" via API or web UI
9. **Instantly updatable** — Add new messages to RAG, personality updates immediately (no retraining)

The end result: an AI that doesn't just sound like Jose — it could pass for Jose in a blind **phone call**.

---

## 12. Future Extensions

| Extension | Description |
|-----------|-------------|
| **Phone call agent** | Twilio integration — the twin answers phone calls as Jose |
| **Real-time learning** | Feed new conversations back to keep the twin current |
| **Multi-persona** | Different modes (professional Jose, casual Jose, creative Jose) |
| **Integration with OH** | Deploy as a personal AI agent that represents Jose |
| **Memory layer** | Add RAG with Jose's documents for factual grounding |
| **Autonomous agent** | Let the twin handle emails, messages, scheduling |
| **Video avatar** | Pair voice clone with lip-synced video (HeyGen, Synthesia) |
| **Voice-to-voice** | Full duplex: listen with Whisper → think with LLM → speak with ElevenLabs |

---

## 13. Phase 7 — Production Polish & Advanced Features

**Status: ✅ IMPLEMENTED**

Seven improvements to push the digital twin toward production quality:

### 7a. WebSocket Frontend Integration ✅
- Rewrote `src/static/index.html` with WebSocket-primary communication
- Auto-reconnect with 3s retry, REST fallback if WS unavailable
- Status indicator (purple dot = WebSocket, green = REST)
- Session-based conversation via `crypto.randomUUID()`

### 7b. Voice Input (Speech-to-Text) ✅
- Added `/transcribe` endpoint to `twin_api.py`
- Uses Groq Whisper `whisper-large-v3` for fast transcription
- Frontend captures audio via `MediaRecorder` (webm/opus format)
- Mic button in UI with recording state indicator
- 25MB file size limit

### 7c. Persistent Memory (SQLite) ✅
- Created `src/memory.py` — `MemoryStore` class backed by SQLite
- Replaces in-memory `defaultdict` in `personality_engine.py`
- WAL mode for concurrent reads, session TTL (24h), auto-cleanup on startup
- Stored at `data/personal/memory.db`, survives Railway redeploys
- Same API: `add_message()`, `get_history()`, `clear_session()`

### 7d. Style Evaluation with LLM Judge ✅
- Added `--style` flag to `tests/eval_identity.py`
- Groq LLM-as-judge evaluates each response on a 1-5 authenticity scale
- Rubric: Does this sound like a 24-year-old Cuban entrepreneur or an AI?
- Outputs grade (A+ to F), score distribution, per-question reasoning

### 7e. RAG Data Rebalancing ✅
- Problem: Instagram was 89% of RAG index (24,317 of 27,266 docs)
- Solution: Source diversification in `PersonalityEngine.retrieve()`
- Over-fetches 3x results, caps each source at `MAX_PER_SOURCE = 5`
- Round-robin selection ensures ChatGPT, iMessage, WhatsApp get fair representation
- Before: 14/15 results from Instagram → After: 9 IG, 5 ChatGPT, 1 iMessage

### 7f. PWA / Mobile Installable ✅
- Created `manifest.json` (standalone display, dark theme, Jose branding)
- Created `sw.js` service worker (network-first HTML, cache-first static)
- Generated gradient icons (192px, 512px) via `scripts/gen_icons.py`
- Mobile-optimized CSS with `100dvh`, touch targets, safe area insets

### 7g. LoRA Fine-Tuning Pipeline ✅
- `scripts/export_training_data.py` — Exports 19,390 chat-format pairs from all sources
  - Instagram: 16,364 | ChatGPT: 2,490 | iMessage: 404 | WhatsApp: 132
  - 90/10 train/val split, quality filtering (length, emoji-only, generic)
- `scripts/finetune_lora.py` — Full unsloth pipeline for Llama 3.1 8B
  - 4-bit quantized LoRA (rank 16, target all projection layers)
  - Cosine LR scheduler, AdamW 8-bit, gradient checkpointing
  - Optional GGUF export for local inference (Ollama, llama.cpp)
  - Test mode with sample queries

### Dependencies Added
- `python-multipart>=0.0.9` — Required for file upload (STT endpoint)
