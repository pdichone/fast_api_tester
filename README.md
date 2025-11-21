# Voice SOAP Notes - MVP Demo

An AI-powered voice-to-SOAP note application designed specifically for massage therapists. This MVP transforms the clinical documentation workflow from cramped text boxes and poor voice recognition to a clean, professional system.

## The Problem It Solves

**Current Reality (Acuity + Phone Voice-to-Text):**
- Tiny text boxes in scheduling software
- Voice dictation misunderstands clinical terms ("levator" → "elevator")
- Manual formatting into SOAP structure
- Messy, unprofessional notes
- Difficult to review past sessions

**With This App:**
- Clean, dedicated interface
- AI-powered transcription that understands medical terminology
- Automatic SOAP formatting
- Professional, insurance-ready documentation
- Organized patient history timeline

## Features

### 1. Patient Dashboard
- Clean list of all patients
- Easy patient management
- Quick access to any patient's records

### 2. Voice Recording
- Simple one-button recording
- 20-second quick notes (or longer if needed)
- Real-time recording timer
- Accurate transcription using OpenAI Whisper

### 3. AI-Powered SOAP Generation
Automatically structures your spoken notes into:
- **Subjective**: Client's reported symptoms and complaints
- **Objective**: Observable findings and measurements
- **Assessment**: Clinical interpretation
- **Plan**: Follow-up recommendations and home care

### 4. Patient History
- Timeline view of all past sessions
- Each visit shows complete SOAP note
- Easy to review progress over time

## Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: SQLite
- **AI**: OpenAI Whisper (transcription) + GPT-4 (SOAP generation)
- **Frontend**: Vanilla HTML/CSS/JavaScript (mobile-friendly)

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd fast_api_tester
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```bash
cp .env.example .env
```

5. Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=sk-your-key-here
```

6. Run the application:
```bash
python main.py
```

7. Open your browser to:
```
http://localhost:8000
```

## Usage Guide

### Step 1: Add Patients
1. Click "Add New Patient"
2. Enter patient name
3. Patient appears in dashboard

### Step 2: Create SOAP Note
1. Select a patient from dashboard
2. Click "Create New SOAP Note"
3. Click the microphone button
4. Speak naturally about the session:
   - What the client reported
   - What you observed
   - What you did in the treatment
   - Response to treatment
   - Recommendations
5. Click stop when done

### Step 3: Review & Save
1. AI generates structured SOAP note automatically
2. Review each section (you can edit if needed)
3. Click "Save Note"
4. Note appears in patient's history timeline

### Example Voice Note

> "John came in today with right-sided neck pain, rating it a 7 out of 10. He said it's been bothering him for about a week, especially after long hours at his computer. On palpation, I found significant hypertonicity in the upper trapezius, levator scapulae, and suboccipital muscles on the right side. Cervical range of motion was reduced, especially right lateral flexion. I performed myofascial release on the upper traps and levator, followed by trigger point therapy on the suboccipitals. After treatment, his pain was down to a 3 out of 10, and cervical ROM improved noticeably. I recommended he continue weekly sessions and gave him stretches for the levator and upper traps to do at home."

**This automatically becomes:**

**Subjective**
Client reports right-sided neck pain at 7/10, present for approximately one week. Pain exacerbated by prolonged computer work.

**Objective**
Significant hypertonicity noted in right upper trapezius, levator scapulae, and suboccipital muscles upon palpation. Reduced cervical range of motion, particularly right lateral flexion.

**Assessment**
Muscular tension pattern consistent with postural strain and overuse. Positive response to manual therapy with pain reduction to 3/10 and improved cervical ROM post-treatment.

**Plan**
Continue weekly massage therapy sessions focusing on cervical mobility and myofascial release. Client educated on home stretching exercises for levator scapulae and upper trapezius muscles. Recommend ergonomic assessment of workstation.

## API Endpoints

### Patients
- `POST /api/patients/` - Create new patient
- `GET /api/patients/` - List all patients
- `GET /api/patients/{id}` - Get patient with notes

### SOAP Notes
- `POST /api/soap-notes/transcribe` - Transcribe audio file
- `POST /api/soap-notes/generate` - Generate SOAP from transcription
- `POST /api/soap-notes/` - Save SOAP note
- `GET /api/soap-notes/patient/{id}` - Get patient's notes

## Database Schema

### Patients Table
- `id`: Primary key
- `name`: Patient name
- `created_at`: Registration date

### SOAP Notes Table
- `id`: Primary key
- `patient_id`: Foreign key to patients
- `date`: Note creation date
- `raw_transcription`: Original spoken text
- `subjective`: S section
- `objective`: O section
- `assessment`: A section
- `plan`: P section

## Cost Estimate

**OpenAI API Usage per note:**
- Whisper transcription: ~$0.006 per minute
- GPT-4 SOAP generation: ~$0.03 per note
- **Total per note**: ~$0.036 (less than 4 cents!)

For 20 patients/day, 5 days/week: ~$14/month

## Future Enhancements

- [ ] Patient search and filtering
- [ ] Export notes to PDF
- [ ] Appointment scheduling integration
- [ ] Mobile app (iOS/Android)
- [ ] Voice note templates by condition
- [ ] Multi-user support for clinics
- [ ] HIPAA compliance features
- [ ] Integration with practice management software
- [ ] Treatment plan tracking
- [ ] Outcome metrics dashboard

## Demo Script

When showing this to your massage therapist friend:

1. **Show the problem**: "Right now you're using Acuity with phone voice-to-text, and it keeps changing 'levator' to 'elevator', right?"

2. **Show the solution**: "Watch this..."
   - Add a demo patient
   - Record a quick voice note
   - Show how it transcribes perfectly
   - Show the professional SOAP format
   - Show the clean history

3. **Highlight the transformation**: "No more tiny text boxes, no more manual formatting, no more fixing 'elevator' to 'levator'. Just speak naturally and get professional documentation."

4. **Ask**: "Would this save you time after every client?"

## License

MIT

## Support

For questions or issues, please open a GitHub issue.
