import re
import bcrypt

from modules.db import create_user, get_user_by_email


EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password):
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")


def verify_password(password, password_hash):
    return bcrypt.checkpw(
        password.encode("utf-8"), password_hash.encode("utf-8")
    )


def signup(name, email, password, confirm_password):
    """
    Validate and create a new account.
    Returns (success: bool, message: str, user: dict or None)
    """

    name = name.strip()
    email = email.strip().lower()

    if not name:
        return False, "Please enter your name.", None

    if not EMAIL_REGEX.match(email):
        return False, "Please enter a valid email address.", None

    if len(password) < 6:
        return False, "Password must be at least 6 characters long.", None

    if password != confirm_password:
        return False, "Passwords do not match.", None

    password_hash = hash_password(password)
    success, message = create_user(name, email, password_hash)

    if not success:
        return False, message, None

    user = get_user_by_email(email)
    return True, "Account created successfully.", user


def login(email, password):
    """
    Validate login credentials.
    Returns (success: bool, message: str, user: dict or None)
    """

    email = email.strip().lower()

    user = get_user_by_email(email)

    if not user:
        return False, "No account found with this email.", None

    if not verify_password(password, user["password_hash"]):
        return False, "Incorrect password.", None

    return True, "Login successful.", user
