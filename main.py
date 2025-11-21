from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import os
from openai import OpenAI
from dotenv import load_dotenv

import database
import schemas

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="Voice SOAP Note Assistant",
    description="AI-powered voice-to-SOAP note generator for massage therapy"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory for frontend files
os.makedirs("static", exist_ok=True)


# --- Patient Endpoints ---

@app.post("/api/patients/", response_model=schemas.Patient)
def create_patient(patient: schemas.PatientCreate, db: Session = Depends(database.get_db)):
    """Create a new patient"""
    db_patient = database.Patient(name=patient.name)
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient


@app.get("/api/patients/", response_model=List[schemas.Patient])
def get_patients(db: Session = Depends(database.get_db)):
    """Get all patients"""
    patients = db.query(database.Patient).all()
    return patients


@app.get("/api/patients/{patient_id}", response_model=schemas.PatientWithNotes)
def get_patient(patient_id: int, db: Session = Depends(database.get_db)):
    """Get a specific patient with their SOAP notes"""
    patient = db.query(database.Patient).filter(database.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


# --- SOAP Note Endpoints ---

@app.post("/api/soap-notes/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file using OpenAI Whisper"""
    try:
        # Read the audio file
        audio_data = await file.read()

        # Save temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(audio_data)

        # Transcribe using Whisper
        with open(temp_file, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        # Clean up temp file
        os.remove(temp_file)

        return {"transcription": transcription.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/soap-notes/generate")
async def generate_soap_note(transcription: dict):
    """Generate structured SOAP note from transcription using GPT-4"""
    try:
        raw_text = transcription.get("transcription", "")

        prompt = f"""You are a medical documentation assistant for massage therapy. Convert the following clinical note into a properly structured SOAP format.

Input: {raw_text}

Generate a professional SOAP note with these four sections:

Subjective: What the client reported (their complaints, symptoms, pain levels, etc.)
Objective: What you observed and measured (muscle tension, ROM, palpation findings, etc.)
Assessment: Your clinical interpretation and response to treatment
Plan: Follow-up recommendations and home care advice

Make it professional, clear, and suitable for insurance documentation. Use proper medical terminology.

Return ONLY a JSON object with these exact keys: subjective, objective, assessment, plan"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical documentation assistant specializing in massage therapy SOAP notes. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        # Parse the response
        import json
        soap_data = json.loads(response.choices[0].message.content)

        return {
            "subjective": soap_data.get("subjective", ""),
            "objective": soap_data.get("objective", ""),
            "assessment": soap_data.get("assessment", ""),
            "plan": soap_data.get("plan", ""),
            "raw_transcription": raw_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SOAP generation failed: {str(e)}")


@app.post("/api/soap-notes/", response_model=schemas.SOAPNote)
def create_soap_note(soap_note: schemas.SOAPNoteCreate, db: Session = Depends(database.get_db)):
    """Save a SOAP note to the database"""
    # Verify patient exists
    patient = db.query(database.Patient).filter(database.Patient.id == soap_note.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    db_note = database.SOAPNote(
        patient_id=soap_note.patient_id,
        raw_transcription=soap_note.raw_transcription,
        subjective=soap_note.subjective,
        objective=soap_note.objective,
        assessment=soap_note.assessment,
        plan=soap_note.plan
    )
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    return db_note


@app.get("/api/soap-notes/patient/{patient_id}", response_model=List[schemas.SOAPNote])
def get_patient_notes(patient_id: int, db: Session = Depends(database.get_db)):
    """Get all SOAP notes for a specific patient"""
    notes = db.query(database.SOAPNote).filter(
        database.SOAPNote.patient_id == patient_id
    ).order_by(database.SOAPNote.date.desc()).all()
    return notes


# --- Frontend ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main dashboard"""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found. Please create static/index.html</h1>")


# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    pass  # Directory doesn't exist yet


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
