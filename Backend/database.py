from pymongo import MongoClient
from datetime import datetime

# Connection
uri = "mongodb+srv://Try:radhey@gymkaro.pqsy3gu.mongodb.net/?appName=GymKaro"
client = MongoClient(uri)
db = client["User_Info"]

# --- USER MANAGEMENT ---

def new_user(email, password, attributes):
    """Creates a new user with password and personal details."""
    if db.Personal_Info.count_documents({'email': email}) != 0:
        print("User Already Exists")
        return False
    
    # In a real app, hash this password!
    user_data = {
        'email': email,
        'password': password, 
        **attributes
    }
    db.Personal_Info.insert_one(user_data)
    print(f"User {email} created successfully.")
    return True

def verify_user(email, password):
    """Checks if email and password match for login."""
    user = db.Personal_Info.find_one({'email': email})
    if user and user.get('password') == password:
        return True
    return False

def fetch_personal_info(email, *args) -> dict | None | list:
    result = db.Personal_Info.find_one({'email': email}, {'_id': 0, 'password': 0}) # Don't return ID or Pass
    if result is None:
        return None
    if not args:
        return result
    return [result.get(value) for value in args]

# --- HISTORY & SESSION MANAGEMENT ---

def save_exercise_session(email, session_data):
    """
    Saves a complete session history.
    session_data should be a dict containing: 
    {'exercise_name', 'mistakes', 'feedback', 'frames', 'summary'}
    """
    log_entry = {
        'email': email,
        'timestamp': datetime.now().isoformat(),
        'session_details': session_data
    }
    
    # specific collection for logs prevents the main document from getting too large
    db.Exercise_Logs.insert_one(log_entry)
    print("Session history saved.")
    return True

def fetch_user_history(email):
    """Fetches ALL past sessions for this user."""
    # Find all logs for the user, sort by latest first
    cursor = db.Exercise_Logs.find({'email': email}, {'_id': 0}).sort('timestamp', -1)
    history = list(cursor)
    
    if not history:
        return []
    return history
