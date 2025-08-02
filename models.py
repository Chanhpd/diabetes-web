from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(512))  # Tăng độ dài để lưu được hash đầy đủ
    full_name = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    address = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    
    # Relationships
    diabetes_predictions = db.relationship('DiabetesPrediction', backref='user', lazy=True)
    heart_predictions = db.relationship('HeartPrediction', backref='user', lazy=True)
    orders = db.relationship('Order', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        print(f"Generated hash: {self.password_hash}")  # Debug line
        
    def check_password(self, password):
        result = check_password_hash(self.password_hash, password)
        print(f"Password check result: {result}")  # Debug line
        return result

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    image_url = db.Column(db.String(500))
    category = db.Column(db.String(50))  # 'diabetes' or 'heart'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50))  # 'diabetes' or 'heart'
    image_url = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    author_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class DiabetesPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pregnancies = db.Column(db.Integer)
    glucose = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    skin_thickness = db.Column(db.Float)
    insulin = db.Column(db.Float)
    bmi = db.Column(db.Float)
    diabetes_pedigree_function = db.Column(db.Float)
    age = db.Column(db.Integer)
    prediction_result = db.Column(db.Boolean)
    prediction_probability = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class HeartPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Integer)
    sex = db.Column(db.String(10))
    chest_pain_type = db.Column(db.String(50))
    resting_bp = db.Column(db.Integer)
    cholesterol = db.Column(db.Integer)
    fasting_bs = db.Column(db.Integer)
    resting_ecg = db.Column(db.String(50))
    max_hr = db.Column(db.Integer)
    exercise_angina = db.Column(db.String(10))
    oldpeak = db.Column(db.Float)
    st_slope = db.Column(db.String(20))
    prediction_result = db.Column(db.Boolean)
    prediction_probability = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    total_amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, confirmed, shipping, delivered, cancelled
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship với OrderItem
    items = db.relationship('OrderItem', backref='order', lazy=True, cascade="all, delete-orphan")

class OrderItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)  # Giá tại thời điểm mua
    
    # Relationship với Product
    product = db.relationship('Product')
