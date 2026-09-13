# 👩‍🏫 AI Teacher • Personal AI Tutor & Memory Agent

A standalone, intelligent AI Teacher application engineered with **Python**, **Streamlit**, **OpenAI API**, and **Zep Cloud** (`zep-cloud` SDK).

The AI Teacher is designed to explain complex Artificial Intelligence, Machine Learning, Deep Learning, and Generative AI concepts with patience, clarity, and real-world analogies. With persistent memory powered by Zep Cloud, it remembers your experience level, learning preferences, questions, and weak points across conversation sessions.

---

## 🌟 Key Features

1. **Session Onboarding Screen**: A sleek, dark-purple aesthetic landing screen that initializes the learner profile before launching the chat room.
2. **Pedagogical Persona**: Acts as an articulate, warm female AI Teacher with 20 core teaching principles, adapting explanations to the student's background.
3. **Structured Conceptual Breakdowns**: Formulates explanations using:
   - 💡 **Definition**
   - 🧠 **Intuitive Explanation**
   - 🌟 **Analogy**
   - ⚙️ **Technical Details**
   - 🚀 **Practical Example**
   - 🌐 **Real-World Use Cases**
   - 🎯 **Check Your Understanding**
4. **Persistent Long-Term Memory (Zep Cloud)**:
   - Tracks learner identity (`Zep User`) independently from individual conversations (`Zep Thread`).
   - Retrieves temporal context blocks (`<USER_SUMMARY>`, `<FACTS>`) for smart personalization.
   - Preserves memory across `+ New Chat` thread resets.
5. **Prompt Security**: Treats external memory as untrusted user-level context to prevent instruction hijacking.
6. **Graceful Degradation**: Clear in-app guidance if API keys are missing or offline.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.13+ | Core application & business logic |
| **Frontend / UI** | Streamlit | Chat messages, input, and onboarding interface |
| **LLM** | OpenAI API (`gpt-4o` / `gpt-4o-mini`) | Generative responses & pedagogical reasoning |
| **Long-Term Memory** | Zep Cloud (`zep-cloud` SDK) | User profiles, threads, and temporal context graphs |
| **Environment** | python-dotenv | Secure secret and API key handling |

---

## 📐 Architecture & Flow

```
USER
 │
 ▼
Streamlit UI (Session Screen / Chat Interface)
 │
 ▼
Python App Controller & Memory Coordinator
 │
 ├──► Zep Cloud (zep-cloud SDK)
 │     ├── 1. Ingest User Query (RFC3339 timestamped)
 │     └── 2. Retrieve Relevant Context (<USER_SUMMARY>, <FACTS>)
 │
 ├──► OpenAI API (openai SDK)
 │     └── 3. Send Teacher System Prompt + Untrusted Context + Chat History
 │
 ├──► Zep Cloud
 │     └── 4. Store Assistant Response in Thread Memory
 │
 ▼
Streamlit UI (Rendered to Learner)
```

---

## 📂 Project Structure

```
d:/Antigravity Projects/AI agent/
│
├── app.py                      # Main Streamlit application entry point
│
├── services/
│   ├── __init__.py
│   ├── openai_service.py       # OpenAI client with streaming & error handling
│   ├── zep_service.py          # Zep Cloud SDK client (User, Thread, Messages, Context)
│   └── memory_service.py       # Memory & pedagogical workflow coordinator
│
├── prompts/
│   ├── __init__.py
│   └── teacher_prompt.py       # AI Teacher pedagogical instructions & prompt formatting
│
├── utils/
│   ├── __init__.py
│   ├── session_manager.py      # Streamlit session_state lifecycle management
│   └── helpers.py              # Identifier generation & sanitization utilities
│
├── components/
│   ├── __init__.py
│   ├── session_screen.py       # Modern dark-purple session initialization card
│   ├── chat_interface.py       # Chat feed, message bubbles, and sidebar controls
│   └── avatar.py               # Lightweight SVG/CSS animated avatar states
│
├── assets/
│   └── style.css               # Dark-purple AI aesthetic design system
│
├── .env                        # Local secrets (ignored by Git)
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation & guide
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI API Key (Required for AI Teacher responses)
OPENAI_API_KEY=your_openai_api_key_here

# Zep Cloud API Key (Required for persistent memory)
# Obtain from https://app.getzep.com/
ZEP_API_KEY=your_zep_api_key_here

# OpenAI Model configuration (e.g. gpt-4o, gpt-4o-mini)
OPENAI_MODEL=gpt-4o
```

> **Security Note**: Never commit your `.env` file to version control. Keys are loaded securely on the backend and are never displayed in the Streamlit UI.

---

## 🚀 Installation & Running

### 1. Clone or Open the Workspace
Navigate to the project root directory:
```bash
cd "d:\Antigravity Projects\AI agent"
```

### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Or on Command Prompt:
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 🧪 Testing the Memory Workflow

You can verify that Zep persistent memory works across sessions and conversation threads:

### Step 1: Thread 1 (Learner Profile & Background)
1. Launch the app and enter your name (e.g. `Yashraj`).
2. Click **Initialize New Session**.
3. Send the message:
   ```
   "My name is Yashraj. I am a beginner in AI and prefer learning through practical, real-world examples."
   ```
4. The AI Teacher will greet you and record this interaction into Zep.

### Step 2: Thread 2 (Cross-Thread Memory Recall)
1. Click the **➕ New Chat** button in the sidebar.
   - Notice that your chat history resets to a clean greeting.
   - Your `Learner ID` remains identical (`learner_yashraj`), but a new `Thread ID` is generated.
2. Ask:
   ```
   "Explain embeddings."
   ```
3. Zep retrieves your learner profile from Thread 1. The AI Teacher tailors its explanation specifically for a beginner, highlighting practical examples rather than heavy mathematical jargon.
4. Expand **🔍 Session & Memory Details** in the sidebar to inspect the retrieved context block directly.

---

## 🔮 Roadmap / Future Features

- [ ] **Voice Pipeline**: Fast request/response speech-to-text (Whisper) and text-to-speech (TTS).
- [ ] **Animated 3D/Video Avatar**: Lifelike visual expressions and lip-sync.
- [ ] **Agent Tools**: Web search, Python code execution runner, and math solver.
- [ ] **Curated RAG Knowledge Base**: Ingest specialized AI/ML textbooks and lecture notes.
- [ ] **Interactive Quizzes & Learning Plans**: Automated weekly syllabus generation and progress tracking.
