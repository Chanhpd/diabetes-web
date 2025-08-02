from app import app, db
from models import User

with app.app_context():
    # Kiểm tra dữ liệu trong bảng user
    users = User.query.all()
    print("\nUsers in database:")
    for user in users:
        print(f"ID: {user.id}")
        print(f"Username: {user.username}")
        print(f"Email: {user.email}")
        print(f"Password Hash: {user.password_hash}")
        print("-" * 50)
