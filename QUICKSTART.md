# Quick Start (2 Minutes)

Get the app running in under 2 minutes.

## Prerequisites
- Python 3.8+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here

# 3. Run
python main.py
```

## Test It

1. Open http://localhost:8000
2. Click "Add New Patient"
3. Enter a name (e.g., "John Smith")
4. Click the patient name
5. Click "Create New SOAP Note"
6. Click the microphone button and speak:

> "Patient came in with lower back pain, rated 6 out of 10. Worked on the lumbar paraspinals and gluteus medius. Pain reduced to 3 out of 10 after treatment."

7. Watch the magic happen!

## What Just Happened?

1. **Whisper** transcribed your voice accurately (including medical terms)
2. **GPT-4** structured it into professional SOAP format:
   - **S**ubjective: What the patient reported
   - **O**bjective: What you observed/did
   - **A**ssessment: Clinical interpretation
   - **P**lan: Follow-up recommendations

## API Endpoints

Test the API directly:

```bash
# Get all patients
curl http://localhost:8000/api/patients/

# Add a patient
curl -X POST http://localhost:8000/api/patients/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Jane Doe"}'

# Get patient's notes
curl http://localhost:8000/api/soap-notes/patient/1
```

## Project Structure

```
fast_api_tester/
├── main.py              # FastAPI app & endpoints
├── database.py          # SQLAlchemy models
├── schemas.py           # Pydantic models
├── static/
│   └── index.html      # Frontend (HTML/CSS/JS)
├── requirements.txt     # Python dependencies
├── .env                # Your API key (not committed)
└── soap_notes.db       # SQLite database (auto-created)
```

## Tech Stack

- **Backend**: FastAPI (async Python web framework)
- **Database**: SQLite + SQLAlchemy ORM
- **AI**: OpenAI (Whisper for transcription, GPT-4 for SOAP)
- **Frontend**: Vanilla JavaScript (no framework needed)

## Cost

~$0.04 per note (~$14/month for 400 notes)

## Next Steps

See [README.md](README.md) for full documentation
See [DEMO_SETUP.md](DEMO_SETUP.md) for demo presentation guide

## Troubleshooting

**"Failed to transcribe"**
- Check your OpenAI API key in `.env`
- Verify you have API credits
- Make sure microphone permissions are granted

**"Module not found"**
```bash
pip install -r requirements.txt
```

**Database errors**
```bash
rm soap_notes.db
python main.py  # Will recreate DB
```

## Development

```bash
# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# View API docs
open http://localhost:8000/docs
```

## Deploy

**Quick Deploy (Render/Railway/Fly.io):**

1. Push code to GitHub
2. Connect to deployment service
3. Add environment variable: `OPENAI_API_KEY`
4. Deploy

See deployment docs for production setup.
