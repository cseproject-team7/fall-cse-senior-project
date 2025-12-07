
import json
import random
import argparse
from datetime import datetime, timedelta

# --- CONFIGURATION ---
BLUEPRINTS_FILE = "shadow_simulation/blueprints.json"
OUTPUT_FILE = "raw_logs/logs.json"
TOTAL_SESSIONS = 5000  # Adjust as needed

# --- APPS & CATEGORIES ---
APPS = {
    "COMMUNICATION": ["Outlook", "Teams"],
    "PRODUCTIVITY": ["Word Online", "Excel Online", "PowerPoint Online", "OneDrive", "OneNote", "SharePoint"],
    "LMS": ["Canvas", "MyUSF"],
    "ADMIN": ["OASIS", "DegreeWorks", "Archivum", "Advisor Appointments", "Schedule Planner"],
    "CAREER": ["Handshake", "LinkedIn", "LinkedIn Learning"],
    "RESEARCH": ["Library Database", "Google Scholar"],
    "EXAM_PROCTORING": ["Respondus LockDown Browser", "Honorlock", "Turnitin"],
    "CLASSROOM": ["TopHat"],
    "CLUBS": ["BullsConnect"],
    "SPECIALIZED": {
        "Computer Science": ["GitHub", "MATLAB Online", "Copilot", "StackOverflow"],
        "Engineering": ["MATLAB Online", "AutoCAD Web", "Copilot"],
        "Arts": ["Adobe Creative Cloud", "Adobe Photoshop", "Adobe Illustrator", "Behance", "Copilot"],
        "Business": ["Excel Online", "PowerBI", "Copilot"],
        "Pre-Med": ["Khan Academy", "Kaplan", "Anki", "Copilot"],
        "Psychology": ["Qualtrics", "Copilot"],
        "Biology": ["LabArchives", "PubMed", "Copilot"],
        "General": ["Copilot"]
    }
}

# --- APP PAIRING & WORKFLOWS ---
# Apps that are commonly used together IN THE SAME SESSION (logical sequences)
APP_PAIRS = {
    # Coursework: Assignment workflow
    "Canvas": ["Word Online", "PowerPoint Online", "Excel Online", "Turnitin", "Canvas"],
    "Word Online": ["Canvas", "OneDrive", "Turnitin", "Word Online"],
    "PowerPoint Online": ["Canvas", "OneDrive", "PowerPoint Online"],
    "Excel Online": ["Canvas", "OneDrive", "Excel Online"],
    "OneDrive": ["Word Online", "Excel Online", "PowerPoint Online", "OneDrive"],
    "Turnitin": ["Canvas", "Word Online", "Turnitin"],
    
    # Coding: Development workflow
    "GitHub": ["StackOverflow", "Copilot", "MATLAB Online", "GitHub"],
    "StackOverflow": ["GitHub", "Copilot", "StackOverflow"],
    "Copilot": ["GitHub", "Word Online", "StackOverflow", "Copilot"],
    "MATLAB Online": ["GitHub", "MATLAB Online"],
    
    # Research: Paper research workflow
    "Library Database": ["Google Scholar", "Word Online", "Library Database"],
    "Google Scholar": ["Library Database", "Word Online", "Google Scholar"],
    
    # Admin: Registration/planning workflow
    "OASIS": ["DegreeWorks", "Schedule Planner", "MyUSF", "OASIS"],
    "DegreeWorks": ["OASIS", "Schedule Planner", "Advisor Appointments", "DegreeWorks"],
    "Schedule Planner": ["OASIS", "DegreeWorks", "MyUSF", "Schedule Planner"],
    "MyUSF": ["OASIS", "Canvas", "DegreeWorks", "MyUSF"],
    "Advisor Appointments": ["DegreeWorks", "OASIS", "Advisor Appointments"],
    
    # Communication: Team collaboration workflow
    "Outlook": ["Teams", "OneDrive", "Outlook"],
    "Teams": ["Outlook", "SharePoint", "OneDrive", "Word Online", "Teams"],
    "SharePoint": ["Teams", "OneDrive", "Word Online", "SharePoint"],
    "OneNote": ["Teams", "Word Online", "OneNote"],
    
    # Exam: Taking exam workflow
    "Respondus LockDown Browser": ["Canvas", "Respondus LockDown Browser"],
    "Honorlock": ["Canvas", "Honorlock"],
    
    # Classroom: Interactive class
    "TopHat": ["Canvas", "OneNote", "TopHat"],
    
    # Career: Single-app sessions (each used alone)
    "Handshake": ["Handshake"],  # Job searching
    "LinkedIn": ["LinkedIn"],     # Networking
    "LinkedIn Learning": ["LinkedIn Learning"],  # Course watching
    
    # Clubs: Event check-in (quick, single-app)
    "BullsConnect": ["BullsConnect"],
    
    # Design/Arts workflows
    "Adobe Creative Cloud": ["Behance", "Adobe Photoshop", "Adobe Illustrator", "Adobe Creative Cloud"],
    "Adobe Photoshop": ["Adobe Creative Cloud", "Behance", "Adobe Photoshop"],
    "Adobe Illustrator": ["Adobe Creative Cloud", "Behance", "Adobe Illustrator"],
    "Behance": ["Adobe Creative Cloud", "Adobe Photoshop", "Behance"],
    
    # Business workflows
    "PowerBI": ["Excel Online", "SharePoint", "PowerBI"],
    
    # Science workflows
    "LabArchives": ["PubMed", "Word Online", "LabArchives"],
    "PubMed": ["LabArchives", "Google Scholar", "PubMed"],
    "Qualtrics": ["Excel Online", "Word Online", "Qualtrics"],
    
    # Pre-Med study workflows
    "Khan Academy": ["Anki", "OneNote", "Khan Academy"],
    "Kaplan": ["Anki", "OneNote", "Kaplan"],
    "Anki": ["Khan Academy", "Kaplan", "OneNote", "Anki"],
    
    # CAD/Engineering
    "AutoCAD Web": ["OneDrive", "AutoCAD Web"],
}

# --- STUDENT PROFILES ---
DEPARTMENTS = ["Computer Science", "Engineering", "Arts", "Business", "Pre-Med", "Psychology", "Biology", "General"]
YEARS = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]

# --- SHARED RESOURCES (Social Correlation) ---
SHARED_DOCS = {dept: [str(uuid.uuid4()) for _ in range(50)] for dept in DEPARTMENTS}

# --- DEVICE PROFILES ---
DEVICE_PROFILES = {
    "iPhone": {"os": "iOS 17", "browser": "Safari 17.0", "client": "Mobile Apps and Desktop clients"},
    "Android": {"os": "Android 13", "browser": "Chrome 120.0", "client": "Mobile Apps and Desktop clients"},
    "Windows_Laptop": {"os": "Windows 10", "browser": "Chrome 120.0", "client": "Browser"},
    "Mac_Laptop": {"os": "macOS 14", "browser": "Safari 17.0", "client": "Browser"},
    "Lab_Desktop": {"os": "Windows 10", "browser": "Edge 120.0", "client": "Browser", "managed": True}
}

# --- ACADEMIC CALENDAR ---
FALL_START = datetime(2025, 8, 25)
FALL_END = datetime(2025, 12, 13)
WINTER_BREAK_START = datetime(2025, 12, 14)
WINTER_BREAK_END = datetime(2026, 1, 5)
SPRING_START = datetime(2026, 1, 6)
SPRING_BREAK_START = datetime(2026, 3, 9)
SPRING_BREAK_END = datetime(2026, 3, 15)
SPRING_END = datetime(2026, 5, 8)

FALL_FINALS = (datetime(2025, 12, 7), datetime(2025, 12, 13))
SPRING_FINALS = (datetime(2026, 4, 27), datetime(2026, 5, 3))

REGISTRATION_DATES = [datetime(2025, 11, 3), datetime(2026, 4, 7)]
TUITION_DUE_DATES = [datetime(2025, 8, 20), datetime(2026, 1, 10)]

# Campus Events (BullsConnect check-in spikes)
CAMPUS_EVENTS = [
    datetime(2025, 8, 25),  # Welcome Week
    datetime(2025, 10, 18), # Homecoming Week
    datetime(2026, 4, 15),  # Bulls Fest (Spring)
]

# ============================================================================
# TUNABLE PARAMETERS - Adjust these to control log generation behavior
# ============================================================================

# --- STUDENT POPULATION ---
NUM_STUDENTS = 50000  # Total number of students to generate

# --- STUDENT TYPE PROBABILITIES ---
PROB_RESIDENT = 0.20           # 20% live on campus
PROB_INTERNATIONAL = 0.10      # 10% international students
PROB_WORKING = 0.30            # 30% have jobs
PROB_PART_TIME = 0.10          # 10% part-time students
PROB_ATHLETE = 0.05            # 5% student athletes
PROB_CLUB_OFFICER = 0.10       # 10% club officers

# --- CLUB ENGAGEMENT DISTRIBUTION ---
PROB_CLUB_NONE = 0.40          # 40% only attend campus events
PROB_CLUB_CASUAL = 0.30        # 30% attend monthly
PROB_CLUB_REGULAR = 0.20       # 20% attend weekly
PROB_CLUB_ACTIVE = 0.10        # 10% very involved (multiple clubs)

# --- SESSION COUNTS (per student over 9 months) ---
SESSIONS_PART_TIME = (10, 20)  # Min, Max
SESSIONS_FOCUSED = (20, 35)
SESSIONS_REGULAR = (15, 30)

# --- SESSION LENGTH (number of app ACCESSES per session, not unique apps) ---
SESSION_SINGLE_APP_PROB = 0.15  # 15% chance of 1-2 app quick session
SESSION_SINGLE_APP_RANGE = (1, 2)

# Focused students: Fewer unique apps, but revisit them often (deep work)
SESSION_FOCUSED_RANGE = (5, 12)      # Reduced from (8, 20)

# Distracted students: Jump between apps frequently
SESSION_DISTRACTED_RANGE = (3, 8)    # Same

# Regular students
SESSION_REGULAR_RANGE = (4, 10)      # Reduced from (5, 15)

# --- APP REVISIT BEHAVIOR ---
# Probability of revisiting an app already used in this session
PROB_REVISIT_FOCUSED = 0.60      # 60% - Focused students stay on same apps
PROB_REVISIT_DISTRACTED = 0.20   # 20% - Distracted students jump around
PROB_REVISIT_REGULAR = 0.40      # 40% - Regular students moderate revisits

# --- REGISTRATION PANIC (per student type) ---
REG_FOCUSED_SENIOR_RANGE = (3, 8)    # Prepared students
REG_DISTRACTED_FRESH_RANGE = (8, 15) # Unprepared students
REG_REGULAR_RANGE = (5, 12)          # Average students

# --- CRITICAL EVENT PARTICIPATION ---
PROB_REGISTRATION_PARTICIPATION = 0.55  # 55% actively register on those days
PROB_TUITION_PARTICIPATION = 0.35       # 35% check tuition on due dates

# --- AUTHENTICATION PROBABILITIES ---
PROB_LOGIN_FAILURE = 0.02          # 2% of logins fail
PROB_MFA_COMMUTER = 0.08           # 8% MFA prompts for commuters
PROB_MFA_RESIDENT = 0.03           # 3% MFA prompts for residents
PROB_MYUSF_SSO = 0.45              # 45% of logins go through MyUSF SSO

# --- NOISE & DISTRACTION ---
NOISE_PROB_DISTRACTED = 0.30       # 30% chance distracted students check email mid-task
NOISE_PROB_FOCUSED = 0.05          # 5% chance focused students get distracted

# --- TIMING PARAMETERS (in seconds) ---
# Time spent on each app
APP_DURATION_FOCUSED = (5*60, 20*60)      # 5-20 minutes per app (deep work)
APP_DURATION_REGULAR = (30, 5*60)         # 30 seconds - 5 minutes per app

# Time for noise/distraction events
NOISE_DURATION = (30, 120)                # 30 seconds - 2 minutes

# Registration panic - rapid switching
REG_PANIC_SWITCH_TIME = (2, 15)           # 2-15 seconds between apps

# Tuition check timing
TUITION_CHECK_TIME = (5, 30)              # 5-30 seconds between checks

# SSO chain delay (MyUSF after sign-in)
SSO_CHAIN_DELAY_MS = (100, 500)           # 100-500 milliseconds

# --- INTRA-SESSION BREAKS & PATTERN TRANSITIONS ---
# Break probability depends on session length (longer sessions = more breaks)
PROB_BREAK_SHORT_SESSION = 0.05       # 5% for short sessions (< 5 apps)
PROB_BREAK_MEDIUM_SESSION = 0.20      # 20% for medium sessions (5-10 apps)
PROB_BREAK_LONG_SESSION = 0.40        # 40% for long sessions (> 10 apps)

# Session length thresholds
SESSION_LENGTH_MEDIUM_THRESHOLD = 5   # Apps (reduced from 8)
SESSION_LENGTH_LONG_THRESHOLD = 10    # Apps (reduced from 15)

# Break duration (creates gap between patterns in same session)
INTRA_SESSION_BREAK_DURATION = (10*60, 30*60)  # 10-30 minute break

# Session gap (defines when logs are grouped into separate sessions)
SESSION_GAP_MINUTES = 20                  # 20-minute gap with NO activity = new session
                                          # Note: Breaks within sessions can be longer than this!

# --- SUNDAY DEADLINE SPIKE ---
# Sunday 11:59 PM assignment deadline behavior
SUNDAY_DEADLINE_HOUR_START = 20          # 8:00 PM - spike begins
SUNDAY_DEADLINE_HOUR_PEAK = 23           # 11:00 PM - peak hour
PROB_SUNDAY_DEADLINE_SESSION = 0.70      # 70% of students work on assignments Sunday night
SUNDAY_DEADLINE_APPS = ["Canvas", "Word Online", "Turnitin", "PowerPoint Online", "Excel Online"]

# --- SSO TOKEN BEHAVIOR ---
# Probability of token refresh (non-interactive sign-in) vs new interactive sign-in
PROB_TOKEN_REFRESH = 0.95                # 95% of app accesses use existing token
TOKEN_REFRESH_INTERVAL_MINUTES = 60      # Tokens refresh every ~60 minutes

# ============================================================================

def get_student_specialized_apps(dept):
    return APPS["SPECIALIZED"].get(dept, APPS["SPECIALIZED"]["General"])

def create_student_profile():
    dept = random.choice(DEPARTMENTS)
    year = random.choice(YEARS)
    
    # Housing Status
    housing = "Resident" if random.random() < PROB_RESIDENT else "Commuter"
    
    # International Student
    is_international = random.random() < PROB_INTERNATIONAL
    
    # Student Type
    is_working = random.random() < PROB_WORKING
    is_part_time = random.random() < PROB_PART_TIME
    is_athlete = random.random() < PROB_ATHLETE
    is_club_officer = random.random() < PROB_CLUB_OFFICER
    
    # Club Engagement
    club_engagement_roll = random.random()
    if club_engagement_roll < PROB_CLUB_NONE:
        club_engagement = "None"
    elif club_engagement_roll < PROB_CLUB_NONE + PROB_CLUB_CASUAL:
        club_engagement = "Casual"
    elif club_engagement_roll < PROB_CLUB_NONE + PROB_CLUB_CASUAL + PROB_CLUB_REGULAR:
        club_engagement = "Regular"
    else:
        club_engagement = "Active"
    
    # Persona probabilities
    persona_weights = {"Focused": 25, "Distracted": 25, "NightOwl": 25, "MorningPerson": 25}
    
    if dept in ["Computer Science", "Engineering"]:
        persona_weights["NightOwl"] += 30
        persona_weights["Focused"] += 10
    elif dept == "Arts":
        persona_weights["NightOwl"] += 20
        persona_weights["Distracted"] += 10
    
    if year == "Freshman":
        persona_weights["Distracted"] += 20
    elif year in ["Senior", "Graduate"]:
        persona_weights["Focused"] += 20
        
    personas = list(persona_weights.keys())
    weights = list(persona_weights.values())
    persona = random.choices(personas, weights=weights)[0]
    
    # Device Assignment (1-3 devices per student)
    devices = []
    # Everyone has a phone
    devices.append(random.choice(["iPhone", "Android"]))
    # 80% have a personal laptop
    if random.random() < 0.8:
        devices.append(random.choice(["Windows_Laptop", "Mac_Laptop"]))
    # CS/Engineering students use lab computers
    if dept in ["Computer Science", "Engineering"] and random.random() < 0.5:
        devices.append("Lab_Desktop")
    
    # Class Schedule (MWF or TR)
    class_schedule = random.choice(["MWF", "TR", "MIXED"])
    
    return {
        "department": dept,
        "year": year,
        "persona": persona,
        "housing": housing,
        "is_international": is_international,
        "is_working": is_working,
        "is_part_time": is_part_time,
        "is_athlete": is_athlete,
        "is_club_officer": is_club_officer,
        "club_engagement": club_engagement,
        "devices": devices,
        "class_schedule": class_schedule,
        "specialized_apps": get_student_specialized_apps(dept)
    }

def get_ip_address(student, hour, device):
    """Determines IP based on Housing, Time, and Device."""
    RESNET = f"131.247.{random.randint(100, 150)}.{random.randint(1, 254)}"
    EDUROAM = f"131.247.{random.randint(0, 50)}.{random.randint(1, 254)}"
    ISP = f"{random.randint(60, 90)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
    VPN = f"131.247.200.{random.randint(1, 254)}"  # VPN gateway
    
    # International students use VPN more often
    if student["is_international"] and random.random() < 0.3:
        return VPN
    
    # Lab computers always on Eduroam
    if device == "Lab_Desktop":
        return EDUROAM
    
    if student["housing"] == "Resident":
        if 9 <= hour <= 17:
            return EDUROAM
        else:
            return RESNET
    else:
        if 9 <= hour <= 16:
            return EDUROAM
        else:
            return ISP

def get_device_for_time(student, hour):
    """Select device based on time of day."""
    devices = student["devices"]
    
    # Morning (6-9 AM): Mobile
    if 6 <= hour < 9:
        return next((d for d in devices if d in ["iPhone", "Android"]), devices[0])
    
    # Class hours (9-5): Laptop or Lab
    elif 9 <= hour <= 17:
        # Prefer lab if available and CS/Eng
        if "Lab_Desktop" in devices and random.random() < 0.4:
            return "Lab_Desktop"
        # Otherwise laptop
        laptop = next((d for d in devices if "Laptop" in d), None)
        return laptop if laptop else devices[0]
    
    # Evening: Laptop or Mobile
    else:
        laptop = next((d for d in devices if "Laptop" in d), None)
        if laptop and random.random() < 0.7:
            return laptop
        return devices[0]

def is_academic_period(date):
    """Check if date is during academic semester."""
    return (FALL_START <= date <= FALL_END) or (SPRING_START <= date <= SPRING_END)

def is_finals_week(date):
    """Check if date is during finals."""
    return (FALL_FINALS[0] <= date <= FALL_FINALS[1]) or (SPRING_FINALS[0] <= date <= SPRING_FINALS[1])

def is_syllabus_week(date):
    """Check if date is first week of semester."""
    return (FALL_START <= date <= FALL_START + timedelta(days=7)) or \
           (SPRING_START <= date <= SPRING_START + timedelta(days=7))

def is_class_day(date, schedule):
    """Check if student has class on this day."""
    weekday = date.weekday()  # 0=Monday, 6=Sunday
    
    if schedule == "MWF":
        return weekday in [0, 2, 4]  # Mon, Wed, Fri
    elif schedule == "TR":
        return weekday in [1, 3]  # Tue, Thu
    else:  # MIXED
        return weekday in [0, 1, 2, 3, 4]  # Mon-Fri

def create_log(user_id, student, app, timestamp, sequence_num, device, ip_address, status="Success", correlation_id=None, is_interactive=True):
    """Create log simulating an Entra ID sign-in log."""
    upn = f"{student['department'].lower().replace(' ', '_')}_{student['year'].lower()}_{user_id}@usf.edu"
    
    device_info = DEVICE_PROFILES[device]
    
    # SSO Token Behavior: Interactive vs Non-Interactive
    if is_interactive:
        client_app = device_info["client"]  # "Browser" or "Mobile Apps and Desktop clients"
    else:
        client_app = "Non-Interactive"  # Token refresh, no user interaction
    
    location = {
        "city": "Tampa", "state": "Florida", "countryOrRegion": "US",
        "geoCoordinates": {"latitude": 28.0587, "longitude": -82.4139}
    }
    
    # Auth Noise
    error_code = 0
    failure_reason = None
    if status == "Interrupted":
        error_code = 50058
        failure_reason = "User needs to perform multi-factor authentication."
    elif status == "Failure":
        error_code = 50126
        failure_reason = "Invalid username or password or invalid on-premise username or password."
    elif status == "Locked":
        error_code = 50053
        failure_reason = "Account is locked. Too many sign-in attempts."

    device_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}_{device}")) if device != "Lab_Desktop" else str(uuid.uuid4())
    
    body = {
        "id": str(uuid.uuid4()),
        "createdDateTime": timestamp.isoformat() + "Z",
        "userPrincipalName": upn,
        "appDisplayName": app,
        "ipAddress": ip_address,
        "clientAppUsed": client_app,
        "deviceDetail": {
            "deviceId": device_id,
            "operatingSystem": device_info["os"],
            "browser": device_info["browser"],
            "isManaged": device_info.get("managed", False)
        },
        "location": location,
        "correlationId": correlation_id if correlation_id else str(uuid.uuid4()),
        "status": {"errorCode": error_code, "failureReason": failure_reason},
        "resourceDisplayName": app
    }
    
    return {
        "SequenceNumber": sequence_num,
        "Offset": str(sequence_num * 500),
        "EnqueuedTimeUtc": timestamp.isoformat() + "Z",
        "Body": json.dumps(body)
    }

def get_category_weights(student, hour, date):
    """Returns weights for app categories based on context."""
    weights = {
        "LMS": 20, "PRODUCTIVITY": 20, "COMMUNICATION": 15, 
        "SPECIALIZED": 15, "RESEARCH": 10, "ADMIN": 10, "CAREER": 5,
        "EXAM_PROCTORING": 0, "CLASSROOM": 5, "CLUBS": 0
    }
    
    # Campus Events: BullsConnect spikes (everyone checking in)
    if any((date.date() - event.date()).days in range(0, 7) for event in CAMPUS_EVENTS):
        weights["CLUBS"] += 40  # Everyone uses BullsConnect during big events
    
    # Syllabus Week: Heavy LMS
    if is_syllabus_week(date):
        weights["LMS"] += 40
        weights["ADMIN"] += 20
    
    # Finals Week: Exam apps + late night studying
    if is_finals_week(date):
        weights["EXAM_PROCTORING"] += 30
        weights["LMS"] += 20
        weights["PRODUCTIVITY"] += 20
        if 22 <= hour or hour <= 4:  # Late night cramming
            weights["LMS"] += 30
    
    # FIX: Sunday 11:59 PM Deadline Spike
    # Massive spike in Canvas/Turnitin on Sunday nights (8 PM - 11:59 PM)
    if date.weekday() == 6 and SUNDAY_DEADLINE_HOUR_START <= hour <= SUNDAY_DEADLINE_HOUR_PEAK:
        weights["LMS"] += 100  # Massive boost to Canvas
        weights["PRODUCTIVITY"] += 80  # Word/Excel for assignments
        weights["EXAM_PROCTORING"] += 60  # Turnitin submissions
        weights["COMMUNICATION"] -= 10  # Less social activity
        weights["CAREER"] = 0  # No career browsing during deadline crunch
    
    # Weekend behavior
    if date.weekday() >= 5:  # Sat/Sun
        weights["LMS"] -= 10
        weights["COMMUNICATION"] += 20
        weights["CAREER"] += 10
    
    # Time of Day
    if 8 <= hour <= 17:
        weights["LMS"] += 20
        weights["ADMIN"] += 10
        weights["CLASSROOM"] += 10
    elif 18 <= hour <= 23 or 0 <= hour < 4:
        weights["SPECIALIZED"] += 30
        weights["PRODUCTIVITY"] += 20
        weights["RESEARCH"] += 10
    
    # Dead Hours (2-4 AM)
    if 2 <= hour < 4 and student["persona"] != "NightOwl":
        for k in weights:
            weights[k] = weights[k] * 0.1  # 90% reduction
        
    # Department
    if student["department"] in ["Computer Science", "Engineering"]:
        weights["SPECIALIZED"] += 40
        weights["RESEARCH"] += 10
    elif student["department"] == "Arts":
        weights["SPECIALIZED"] += 40
    elif student["department"] == "Business":
        weights["PRODUCTIVITY"] += 30
        weights["COMMUNICATION"] += 20
    elif student["department"] in ["Pre-Med", "Biology", "Psychology"]:
        weights["RESEARCH"] += 30
        weights["LMS"] += 10
        
    # Year
    if student["year"] == "Freshman":
        weights["LMS"] += 20
        weights["COMMUNICATION"] += 20
        weights["CAREER"] = 0
    elif student["year"] == "Senior":
        weights["CAREER"] += 40
        weights["SPECIALIZED"] += 20
    elif student["year"] == "Graduate":
        weights["RESEARCH"] += 50
        weights["SPECIALIZED"] += 30
        weights["LMS"] -= 10
    
    # Working students: Less daytime activity
    if student["is_working"] and 9 <= hour <= 17:
        for k in weights:
            weights[k] = weights[k] * 0.5
    
    # Athletes: Early morning
    if student["is_athlete"] and 6 <= hour < 8:
        weights["COMMUNICATION"] += 30
    
    # Club officers: Heavy Teams/SharePoint + BullsConnect
    if student["is_club_officer"]:
        weights["COMMUNICATION"] += 20
        weights["PRODUCTIVITY"] += 15
        weights["CLUBS"] += 40  # Officers check in frequently
    else:
        # Club engagement levels
        if student["club_engagement"] == "Active":
            weights["CLUBS"] += 25  # Multiple clubs, checking in often
        elif student["club_engagement"] == "Regular":
            weights["CLUBS"] += 15  # Weekly meetings (e.g., Bulls Racing)
        elif student["club_engagement"] == "Casual":
            weights["CLUBS"] += 8   # Monthly meetings
        else:  # None
            weights["CLUBS"] += 2   # Only campus events/free food
        
    return weights

def generate_session(student, start_time, session_ip, is_registration_panic=False, is_tuition_panic=False):
    """Generates a sequence of apps for a single session."""
    session_apps = []
    current_time = start_time
    device = get_device_for_time(student, start_time.hour)
    
    # FIX: Track token refresh for SSO behavior
    last_token_refresh = start_time
    
    # Registration Panic - Different behaviors based on student type
    if is_registration_panic:
        apps = ["OASIS", "DegreeWorks", "Schedule Planner", "MyUSF"]
        
        # Focused/Seniors: Prepared, planned ahead
        if student["persona"] == "Focused" or student["year"] in ["Senior", "Graduate"]:
            num_events = random.randint(*REG_FOCUSED_SENIOR_RANGE)
        # Distracted/Freshmen: Moderate panic
        elif student["persona"] == "Distracted" or student["year"] == "Freshman":
            num_events = random.randint(*REG_DISTRACTED_FRESH_RANGE)
        # Others: Some panic
        else:
            num_events = random.randint(*REG_REGULAR_RANGE)
            
        for _ in range(num_events):
            app = random.choice(apps)
            session_apps.append((app, current_time, device, session_ip, "Success", None, False))  # Non-interactive
            current_time += timedelta(seconds=random.randint(*REG_PANIC_SWITCH_TIME))
        return session_apps, current_time
    
    # Tuition Panic
    if is_tuition_panic:
        apps = ["OASIS", "MyUSF"]
        num_events = random.randint(3, 8)
        for _ in range(num_events):
            app = random.choice(apps)
            session_apps.append((app, current_time, device, session_ip, "Success", None, False))  # Non-interactive
            current_time += timedelta(seconds=random.randint(*TUITION_CHECK_TIME))
        return session_apps, current_time

    # Normal Session
    cat_weights = get_category_weights(student, start_time.hour, start_time)
    cats = list(cat_weights.keys())
    weights = list(cat_weights.values())
    
    current_category = random.choices(cats, weights=weights)[0]
    
    # Session length based on persona and quick session probability
    if random.random() < SESSION_SINGLE_APP_PROB:
        num_events = random.randint(*SESSION_SINGLE_APP_RANGE)
    elif student["persona"] == "Focused":
        num_events = random.randint(*SESSION_FOCUSED_RANGE)
    elif student["persona"] == "Distracted":
        num_events = random.randint(*SESSION_DISTRACTED_RANGE)
    else:
        num_events = random.randint(*SESSION_REGULAR_RANGE)
    
    # Track apps used in this session for revisit logic
    apps_used_in_session = []
    last_app = None  # Track the most recent app for logical transitions
    
    # Revisit probability based on persona
    if student["persona"] == "Focused":
        revisit_prob = PROB_REVISIT_FOCUSED
    elif student["persona"] == "Distracted":
        revisit_prob = PROB_REVISIT_DISTRACTED
    else:
        revisit_prob = PROB_REVISIT_REGULAR
        
    for _ in range(num_events):
        # Decide: revisit an existing app or pick a new one?
        if last_app and random.random() < revisit_prob:
            # Use logical pairing if available
            if last_app in APP_PAIRS:
                # Pick from apps that logically follow the last app
                paired_apps = APP_PAIRS[last_app]
                # Filter to only apps already used in session (for revisits)
                available_revisits = [a for a in paired_apps if a in apps_used_in_session]
                
                if available_revisits:
                    # Revisit a logically related app
                    app = random.choice(available_revisits)
                else:
                    # No logical revisit available, pick from paired apps (might be new)
                    app = random.choice(paired_apps)
                    if app not in apps_used_in_session:
                        apps_used_in_session.append(app)
            else:
                # No pairing defined, random revisit from session history
                if apps_used_in_session:
                    app = random.choice(apps_used_in_session)
                else:
                    # Fallback to new app
                    if current_category == "SPECIALIZED":
                        pool = student["specialized_apps"]
                    else:
                        pool = APPS[current_category]
                    app = random.choice(pool)
                    apps_used_in_session.append(app)
        else:
            # Pick a new app from current category
            if current_category == "SPECIALIZED":
                pool = student["specialized_apps"]
            else:
                pool = APPS[current_category]
            
            app = random.choice(pool)
            
            # Add to session history
            if app not in apps_used_in_session:
                apps_used_in_session.append(app)
        
        last_app = app  # Update for next iteration
        
        # Social Correlation
        res_id = None
        if current_category == "PRODUCTIVITY" and random.random() < 0.3:
            res_id = random.choice(SHARED_DOCS[student["department"]])
        
        # Noise
        noise_prob = NOISE_PROB_DISTRACTED if student["persona"] == "Distracted" else NOISE_PROB_FOCUSED
        if random.random() < noise_prob:
            noise_app = random.choice(["Outlook", "Teams"])
            
            # Check if token needs refresh (every ~60 minutes)
            is_interactive = (current_time - last_token_refresh).total_seconds() > (TOKEN_REFRESH_INTERVAL_MINUTES * 60)
            if is_interactive:
                last_token_refresh = current_time
            
            session_apps.append((noise_app, current_time, device, session_ip, "Success", None, is_interactive))
            current_time += timedelta(seconds=random.randint(*NOISE_DURATION))
            
            if student["persona"] == "Distracted" and random.random() < 0.4:
                current_category = "COMMUNICATION"

        # Check if token needs refresh for main app access
        is_interactive = (current_time - last_token_refresh).total_seconds() > (TOKEN_REFRESH_INTERVAL_MINUTES * 60)
        if is_interactive:
            last_token_refresh = current_time
        
        session_apps.append((app, current_time, device, session_ip, "Success", res_id, is_interactive))
        
        if student["persona"] == "Focused":
            duration = random.randint(*APP_DURATION_FOCUSED)
        else:
            duration = random.randint(*APP_DURATION_REGULAR)
            
        current_time += timedelta(seconds=duration)
        
        # Transition to next category
        next_cat_weights = get_category_weights(student, current_time.hour, current_time)
        next_cat_weights[current_category] += 50  # Sticky - likely to stay in same category
        
        cats = list(next_cat_weights.keys())
        weights = list(next_cat_weights.values())
        next_category = random.choices(cats, weights=weights)[0]
        
        # Intra-session break (between patterns within same session)
        # Probability depends on session length
        if num_events < SESSION_LENGTH_MEDIUM_THRESHOLD:
            break_prob = PROB_BREAK_SHORT_SESSION
        elif num_events < SESSION_LENGTH_LONG_THRESHOLD:
            break_prob = PROB_BREAK_MEDIUM_SESSION
        else:
            break_prob = PROB_BREAK_LONG_SESSION
        
        # Only happens when switching to a very different category
        if next_category != current_category and random.random() < break_prob:
            # Take a break (10-30 minutes of no activity)
            break_duration = random.randint(*INTRA_SESSION_BREAK_DURATION)
            current_time += timedelta(seconds=break_duration)
        
        current_category = next_category
        
    return session_apps, current_time

def generate_logs(output_file):
    sequence_num = 0
    
    # Total number of students
    num_students = NUM_STUDENTS
    students = [create_student_profile() for _ in range(num_students)]
    
    print(f"Generating logs for {len(students)} students...")
    print(f"Streaming output to {output_file}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_logs_generated = 0
    
    with open(output_file, 'w') as f:
        for i, student in enumerate(students):
            if i % 100 == 0: 
                print(f"Processing student {i}/{len(students)}... ({total_logs_generated:,} logs generated)")
            
            student_logs = []
            
            # Session counts based on student type
            if student["is_part_time"]:
                num_sessions = random.randint(*SESSIONS_PART_TIME)
            elif student["persona"] == "Focused":
                num_sessions = random.randint(*SESSIONS_FOCUSED)
            else:
                num_sessions = random.randint(*SESSIONS_REGULAR)
                
            # Generate Sessions
            session_dates = []
            
            # Distribute sessions across academic year
            for _ in range(num_sessions):
                # Pick a random date between Fall Start and Spring End
                days_range = (SPRING_END - FALL_START).days
                random_days = random.randint(0, days_range)
                candidate_date = FALL_START + timedelta(days=random_days)
                
                # Skip Winter Break
                if WINTER_BREAK_START <= candidate_date <= WINTER_BREAK_END:
                    continue
                    
                # Skip Spring Break (mostly)
                if SPRING_BREAK_START <= candidate_date <= SPRING_BREAK_END and random.random() > 0.1:
                    continue
                    
                # Adjust time based on student type
                if student["is_working"]:
                    hour = random.choice([18, 19, 20, 21, 22])  # Evening
                elif student["is_athlete"]:
                    hour = random.choice([6, 7, 8, 16, 17, 18])  # Early or after practice
                else:
                    hour = random.randint(8, 22)
                    
                session_time = candidate_date.replace(hour=hour, minute=random.randint(0, 59))
                session_dates.append(session_time)
                
            # Add Critical Events
            for reg_date in REGISTRATION_DATES:
                if random.random() < PROB_REGISTRATION_PARTICIPATION:
                    panic_time = reg_date.replace(hour=random.choice([6, 13]), minute=random.randint(0, 59))
                    session_dates.append(panic_time)
            
            for tuition_date in TUITION_DUE_DATES:
                if random.random() < PROB_TUITION_PARTICIPATION:
                    tuition_time = tuition_date.replace(hour=random.randint(9, 20), minute=random.randint(0, 59))
                    session_dates.append(tuition_time)
            
            # FIX: Sunday 11:59 PM Deadline Spike
            # Add Sunday night sessions (8 PM - 11:59 PM) for 70% of students
            if random.random() < PROB_SUNDAY_DEADLINE_SESSION:
                # Find all Sundays during academic periods
                current_date = FALL_START
                while current_date <= SPRING_END:
                    if current_date.weekday() == 6 and is_academic_period(current_date):  # Sunday
                        # Random time between 8 PM and 11:59 PM
                        deadline_hour = random.randint(SUNDAY_DEADLINE_HOUR_START, SUNDAY_DEADLINE_HOUR_PEAK)
                        deadline_minute = random.randint(0, 59)
                        deadline_time = current_date.replace(hour=deadline_hour, minute=deadline_minute)
                        session_dates.append(deadline_time)
                    current_date += timedelta(days=1)
                
            session_dates.sort()
            
            # Track failed login attempts for lockout logic
            failed_attempts = 0
            last_failure_time = None
            
            for start_time in session_dates:
                is_panic = any(start_time.date() == d.date() for d in REGISTRATION_DATES)
                is_tuition = any(start_time.date() == d.date() for d in TUITION_DUE_DATES)
                
                device = get_device_for_time(student, start_time.hour)
                session_ip = get_ip_address(student, start_time.hour, device)  # Generate IP for this session
                
                # Auth Noise
                mfa_prob = PROB_MFA_COMMUTER if student["housing"] == "Commuter" else PROB_MFA_RESIDENT
                
                # Account Lockout Logic
                if last_failure_time and (start_time - last_failure_time).total_seconds() < 1800:  # 30 min
                    if failed_attempts >= 3:
                        student_logs.append(create_log(i, student, "Microsoft 365 Sign-in", start_time, sequence_num, device, session_ip, status="Locked", is_interactive=True))
                        sequence_num += 1
                        continue  # Skip session
                else:
                    failed_attempts = 0  # Reset after 30 min
                
                # Failed Login (Interactive)
                if random.random() < PROB_LOGIN_FAILURE:
                    student_logs.append(create_log(i, student, "Microsoft 365 Sign-in", start_time, sequence_num, device, session_ip, status="Failure", is_interactive=True))
                    sequence_num += 1
                    failed_attempts += 1
                    last_failure_time = start_time
                    start_time += timedelta(seconds=5)
                
                # MFA Prompt (Interactive)
                if random.random() < mfa_prob:
                    student_logs.append(create_log(i, student, "Microsoft 365 Sign-in", start_time, sequence_num, device, session_ip, status="Interrupted", is_interactive=True))
                    sequence_num += 1
                    start_time += timedelta(seconds=15)
                    
                # Successful Login (Interactive - first sign-in of session)
                student_logs.append(create_log(i, student, "Microsoft 365 Sign-in", start_time, sequence_num, device, session_ip, status="Success", is_interactive=True))
                sequence_num += 1
                
                # SSO Chain (not everyone goes through MyUSF) - Non-Interactive
                if random.random() < PROB_MYUSF_SSO:
                    start_time += timedelta(milliseconds=random.randint(*SSO_CHAIN_DELAY_MS))
                    student_logs.append(create_log(i, student, "MyUSF", start_time, sequence_num, device, session_ip, status="Success", is_interactive=False))
                    sequence_num += 1
                
                # Generate Session
                events, end_time = generate_session(student, start_time + timedelta(seconds=30), 
                                                   session_ip,
                                                   is_registration_panic=is_panic, 
                                                   is_tuition_panic=is_tuition)
                
                # Process session events (now includes ip and is_interactive)
                for app, timestamp, dev, ip, status, res_id, is_interactive in events:
                    student_logs.append(create_log(i, student, app, timestamp, sequence_num, dev, ip, status=status, correlation_id=res_id, is_interactive=is_interactive))
                    sequence_num += 1
                    
                # Logout (Interactive)
                student_logs.append(create_log(i, student, "Microsoft 365 Sign-out", end_time + timedelta(seconds=30), sequence_num, device, session_ip, is_interactive=True))
                sequence_num += 1

            # Sort logs for this student by timestamp
            student_logs.sort(key=lambda x: json.loads(x["Body"])["createdDateTime"])
            
            # Write logs for this student immediately
            for log in student_logs:
                f.write(json.dumps(log) + '\n')
            
            total_logs_generated += len(student_logs)
            
            # Explicitly clear memory
            del student_logs

    print(f"\nGeneration Summary:")
    print(f"Total logs: {total_logs_generated:,}")
    print(f"Students: {len(students)}")
        
    return total_logs_generated

if __name__ == "__main__":
    output_path = os.path.join(OUTPUT_DIR, "logs.json")
    generate_logs(output_path)
    print(f"Logs generated at {output_path}")

