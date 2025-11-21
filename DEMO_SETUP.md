# Quick Demo Setup Guide

## Before the Demo

### 1. Get Your OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy it immediately (you won't be able to see it again)

### 2. Set Up the App (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Edit .env and add your API key
# OPENAI_API_KEY=sk-your-actual-key-here

# Run the app
python main.py
```

The app will start at: http://localhost:8000

### 3. Pre-load Demo Data (Optional)
Before your demo, you can add a few sample patients so the dashboard looks populated.

## Demo Script

### Opening Hook (30 seconds)
"Remember how you told me that Acuity's tiny text boxes are driving you crazy? And how your phone keeps changing 'levator' to 'elevator'? I built something to fix that. Want to see?"

### Part 1: The Problem (Show, Don't Tell) - 1 minute
**Pull up her current workflow on your phone:**
- "Show me what you do now"
- Let her describe the pain points as she navigates Acuity
- Key frustrations she'll mention:
  - Finding the appointment
  - Tiny text box
  - Voice-to-text mistakes
  - Manual SOAP formatting

**Empathize:** "Yeah, that's exactly what you told me. Let me show you what I built."

### Part 2: The Solution - 3 minutes

**Screen 1: Dashboard**
- "This is your patient list. Clean and simple."
- "No appointments, no scheduling clutter. Just your clients."

**Screen 2: Record Voice Note**
- Click on a patient (or create "Demo Patient")
- Click "Create New SOAP Note"
- **Say this exact example:**

> "John came in today with right-sided neck pain, rating it a 7 out of 10. He said it's been bothering him for about a week, especially after long hours at his computer. On palpation, I found significant hypertonicity in the upper trapezius, levator scapulae, and suboccipital muscles on the right side. Cervical range of motion was reduced, especially right lateral flexion. I performed myofascial release on the upper traps and levator, followed by trigger point therapy on the suboccipitals. After treatment, his pain was down to a 3 out of 10, and cervical ROM improved noticeably. I recommended he continue weekly sessions and gave him stretches for the levator and upper traps to do at home."

**Screen 3: The Magic Moment**
- Watch her face as the SOAP note generates
- **Stay quiet and let it sink in**

**Point out:**
- "Notice it got every term right?"
- "levator scapulae" - not "elevator"
- "suboccipital" - not "sub optimal"
- "myofascial" - not "my facial"

**Screen 4: Professional Output**
- Show the formatted SOAP sections
- "This is insurance-ready documentation"
- "You can edit anything before saving"

**Screen 5: History Timeline**
- "And here's every visit in one place"
- "Click any note to see the full SOAP details"

### Part 3: The Close - 1 minute

**Ask the key question:**
"How much time would this save you per client?"

**Listen to her response, then:**
"Imagine doing this for all 20 clients you see this week. How many hours back?"

**Address the obvious question:**
"This costs about 4 cents per note. That's less than a dollar per day."

**The decision:**
"Want to try it with a few real clients this week? I'll get you set up."

## Demo Day Checklist

- [ ] OpenAI API key set up and tested
- [ ] App running and accessible
- [ ] Microphone permissions granted in browser
- [ ] Test recording works on your device
- [ ] Example script practiced (you want to sound natural)
- [ ] At least 1 demo patient added
- [ ] Phone/laptop charged
- [ ] Good internet connection

## Troubleshooting

### Microphone Not Working
- Check browser permissions (usually a prompt appears)
- Try HTTPS if possible (some browsers require it for mic access)
- Use Chrome or Safari for best compatibility

### Transcription Fails
- Check your OpenAI API key in .env
- Verify you have credits in your OpenAI account
- Check internet connection

### SOAP Generation Fails
- Same as transcription - usually API key or credits
- The free trial gives you $5 credit (enough for 100+ notes)

## Cost Breakdown (Show This If Asked)

**Per Note:**
- Whisper transcription: $0.006 per minute (~$0.006 for 1 minute)
- GPT-4 SOAP generation: ~$0.03
- **Total: ~$0.036 per note**

**Monthly (20 clients/day, 5 days/week):**
- 100 notes/week = 400 notes/month
- 400 × $0.036 = **$14.40/month**

Compare to:
- Time saved: ~2-3 minutes per client
- 400 clients/month × 2.5 minutes = **1,000 minutes = 16+ hours saved**

**ROI: Save 16 hours for $14. That's less than $1/hour.**

## Follow-Up Questions She Might Ask

**Q: Is this HIPAA compliant?**
A: "Right now it's a demo using OpenAI's API. For production, we'd need to sign a BAA (Business Associate Agreement) with OpenAI - they offer that for $30/month. Or we could host our own AI model for full control."

**Q: Can I export these notes?**
A: "Not yet in this MVP, but that's a quick add. PDF export is on the roadmap."

**Q: What if I want to use my existing patient names from Acuity?**
A: "We can import them. Would you want automatic sync, or just a one-time import?"

**Q: Can other therapists in my practice use this?**
A: "The MVP is single-user, but multi-user is straightforward to add. How many therapists would need access?"

**Q: What about mobile?**
A: "This works on mobile browsers now. A native app would be better - that's the next phase after you validate this works for you."

## The Next Step

If she says yes:
1. Deploy this to a proper server (not just localhost)
2. Give her the URL and credentials
3. Ask her to use it for 1 week with real clients
4. Schedule a follow-up to discuss:
   - What worked
   - What needs adjustment
   - Whether to build this out further

## Your Pitch Position

You're not selling her a product. You're showing her a solution to HER specific problem that you built in one session.

The goal is:
1. Validation: Does this actually solve her problem?
2. Feedback: What would make this perfect for her workflow?
3. Decision: Should this become a real product?

You're offering to be her technical co-founder on this if she wants to turn it into a real business serving massage therapists.

**That's the real opportunity here.**
