from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


class PatientBase(BaseModel):
    name: str


class PatientCreate(PatientBase):
    pass


class Patient(PatientBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class SOAPNoteBase(BaseModel):
    raw_transcription: str
    subjective: str
    objective: str
    assessment: str
    plan: str


class SOAPNoteCreate(BaseModel):
    patient_id: int
    raw_transcription: str
    subjective: str
    objective: str
    assessment: str
    plan: str


class SOAPNote(SOAPNoteBase):
    id: int
    patient_id: int
    date: datetime

    class Config:
        from_attributes = True


class PatientWithNotes(Patient):
    soap_notes: List[SOAPNote] = []

    class Config:
        from_attributes = True
