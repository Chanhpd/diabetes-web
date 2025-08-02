from flask import Flask, render_template, request, redirect, url_for, flash, session
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
from functools import wraps

from extensions import db
from models import User, Product, Article, DiabetesPrediction, HeartPrediction, Order, OrderItem

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:@localhost/health_predictor"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Khởi tạo các extension với app
db.init_app(app)

# Create database tables
with app.app_context():
    db.create_all()

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()  # Rollback session in case of error
    app.logger.error(f'Server Error: {error}')
    return render_template('error.html', error=error), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error=error), 404

# Decorator để kiểm tra đăng nhập
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Vui lòng đăng nhập để tiếp tục', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Load Diabetes model
try:
    diabetes_model = joblib.load("diabetes/diabetes_model.pkl")
    print("✅ Diabetes model loaded successfully!")
    
    # Load feature names from file
    try:
        diabetes_feature_names = joblib.load("diabetes/feature_names.pkl")
        print(f"✅ Diabetes feature names loaded: {diabetes_feature_names}")
    except:
        print("⚠️ Could not load diabetes feature_names.pkl, using default names")
        diabetes_feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
    
    # Mapping từ tên field trong HTML sang tên chuẩn cho diabetes
    diabetes_field_mapping = {
        'npreg': 'Pregnancies',
        'glu': 'Glucose', 
        'bp': 'BloodPressure',
        'skin': 'SkinThickness',
        'insu': 'Insulin',
        'bmi': 'BMI',
        'ped': 'DiabetesPedigreeFunction',
        'age': 'Age'
    }
    
    # Thứ tự features cho diabetes model
    diabetes_feature_order = diabetes_feature_names
    
    print(f"📊 Expected diabetes features: {diabetes_feature_order}")
    
except Exception as e:
    print(f"❌ Error loading diabetes model: {e}")
    diabetes_model = None
    diabetes_field_mapping = {}
    diabetes_feature_order = []

# Load Heart Disease model
try:
    heart_model = joblib.load("heart_package/heart_disease_model.pkl")
    print("✅ Heart disease model loaded successfully!")
    
    # Load feature names from file
    try:
        heart_feature_names = joblib.load("heart_package/feature_names.pkl")
        print(f"✅ Heart feature names loaded: {heart_feature_names}")
    except:
        print("⚠️ Could not load heart feature_names.pkl, using default names")
        heart_feature_names = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
        ]
    
    # Load label encoders
    try:
        heart_label_encoders = joblib.load("heart_package/label_encoders.pkl")
        print(f"✅ Heart label encoders loaded: {list(heart_label_encoders.keys())}")
    except:
        print("⚠️ Could not load heart label_encoders.pkl")
        heart_label_encoders = {}
    
    # Mapping từ tên field trong HTML sang tên chuẩn cho heart
    heart_field_mapping = {
        'age': 'Age',
        'sex': 'Sex',
        'chest_pain': 'ChestPainType',
        'resting_bp': 'RestingBP',
        'cholesterol': 'Cholesterol',
        'fasting_bs': 'FastingBS',
        'resting_ecg': 'RestingECG',
        'max_hr': 'MaxHR',
        'exercise_angina': 'ExerciseAngina',
        'oldpeak': 'Oldpeak',
        'st_slope': 'ST_Slope'
    }
    
    # Thứ tự features cho heart model
    heart_feature_order = heart_feature_names
    
    print(f"📊 Expected heart features: {heart_feature_order}")
    
except Exception as e:
    print(f"❌ Error loading heart model: {e}")
    heart_model = None
    heart_field_mapping = {}
    heart_feature_order = []
    heart_label_encoders = {}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/articles")
def articles():
    return render_template("articles.html")

@app.route("/products")
def products():
    # Lấy sản phẩm từ database
    diabetes_products = Product.query.filter_by(category='diabetes').all()
    heart_products = Product.query.filter_by(category='heart').all()
    
    # Lấy thông tin user nếu đã đăng nhập
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    
    return render_template("products.html", 
                         diabetes_products=diabetes_products,
                         heart_products=heart_products,
                         user=user)

@app.route("/buy/<int:product_id>", methods=["POST"])
@login_required
def buy_product(product_id):
    user = User.query.get(session['user_id'])
    product = Product.query.get_or_404(product_id)
    
    # Lấy thông tin từ form
    quantity = int(request.form.get('quantity', 1))
    full_name = request.form.get('full_name', user.full_name)
    phone = request.form.get('phone', user.phone)
    address = request.form.get('address', user.address)
    
    if not all([full_name, phone, address]):
        flash('Vui lòng điền đầy đủ thông tin giao hàng', 'error')
        return redirect(url_for('products'))
    
    # Tạo đơn hàng mới
    order = Order(
        user_id=user.id,
        full_name=full_name,
        phone=phone,
        address=address,
        total_amount=product.price * quantity
    )
    db.session.add(order)
    
    # Thêm sản phẩm vào đơn hàng
    order_item = OrderItem(
        product_id=product.id,
        quantity=quantity,
        price=product.price
    )
    order.items.append(order_item)
    
    # Cập nhật thông tin user nếu cần
    if not user.full_name or not user.phone or not user.address:
        user.full_name = full_name
        user.phone = phone
        user.address = address
    
    try:
        db.session.commit()
        flash('Đặt hàng thành công! Chúng tôi sẽ liên hệ với bạn sớm.', 'success')
    except:
        db.session.rollback()
        flash('Có lỗi xảy ra, vui lòng thử lại sau.', 'error')
    
    return redirect(url_for('products'))

@app.route("/my-orders")
@login_required
def my_orders():
    user = User.query.get(session['user_id'])
    orders = Order.query.filter_by(user_id=user.id).order_by(Order.created_at.desc()).all()
    return render_template("my_orders.html", orders=orders)

@app.route("/about")
def about():
    return render_template("about.html")

# Decorator để kiểm tra đăng nhập
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Vui lòng đăng nhập để tiếp tục', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/register", methods=["GET", "POST"])
def register():
    try:
        # Nếu người dùng đã đăng nhập, chuyển hướng về trang chủ
        if 'user_id' in session:
            return redirect(url_for('home'))

        if request.method == "POST":
            # Lấy dữ liệu từ form
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '')
            
            # Log dữ liệu nhận được (không log password)
            app.logger.info(f"Register attempt - Username: {username}, Email: {email}")
            
            # Kiểm tra dữ liệu đầu vào
            if not username or not email or not password:
                flash('Vui lòng điền đầy đủ thông tin', 'error')
                return render_template('register.html')
            
            # Kiểm tra độ dài username
            if len(username) < 3:
                flash('Tên đăng nhập phải có ít nhất 3 ký tự', 'error')
                return render_template('register.html')
            
            # Kiểm tra độ dài và định dạng email
            if len(email) < 5 or '@' not in email:
                flash('Email không hợp lệ', 'error')
                return render_template('register.html')
            
            # Kiểm tra độ dài password
            if len(password) < 6:
                flash('Mật khẩu phải có ít nhất 6 ký tự', 'error')
                return render_template('register.html')
            
            # Kiểm tra username đã tồn tại
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash('Tên đăng nhập đã tồn tại', 'error')
                return render_template('register.html')
                
            # Kiểm tra email đã tồn tại
            existing_email = User.query.filter_by(email=email).first()
            if existing_email:
                flash('Email đã tồn tại', 'error')
                return render_template('register.html')
            
            # Tạo user mới
            user = User(
                username=username,
                email=email,
                is_admin=False  # Mặc định là user thường
            )
            user.set_password(password)
            
            # Lưu vào database
            db.session.add(user)
            db.session.commit()
            app.logger.info(f"User registered successfully: {username}")
            
            flash('Đăng ký thành công! Vui lòng đăng nhập', 'success')
            return redirect(url_for('login'))
            
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Registration error: {str(e)}")
        flash('Có lỗi xảy ra, vui lòng thử lại', 'error')
        return render_template('register.html')
            
    return render_template('register.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    # Nếu người dùng đã đăng nhập, chuyển hướng về trang chủ
    if 'user_id' in session:
        return redirect(url_for('home'))

    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Vui lòng điền đầy đủ thông tin', 'error')
            return render_template("login.html")
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            # Lưu thông tin user vào session
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            
            flash('Đăng nhập thành công!', 'success')
            next_page = request.args.get('next')
            if next_page and next_page != '/logout':  # Tránh redirect loop
                return redirect(next_page)
            return redirect(url_for('home'))
        
        flash('Email hoặc mật khẩu không đúng', 'error')
    return render_template("login.html")

# Hàm tiện ích để lấy thông tin user hiện tại
def get_current_user():
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None

@app.context_processor
def utility_processor():
    # Thêm các hàm và biến để sử dụng trong tất cả templates
    return {
        'get_current_user': get_current_user
    }

@app.route("/logout")
def logout():
    # Xóa tất cả thông tin session
    session.clear()
    flash('Đã đăng xuất thành công', 'success')
    return redirect(url_for('home'))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html", feature_names=diabetes_feature_order)
    if diabetes_model is None:
        return render_template("index.html", 
                             prediction_text="❌ Diabetes model chưa được load!", 
                             feature_names=diabetes_feature_order)
    try:
        # Debug: In ra tất cả dữ liệu nhận được
        print("📥 Form data received:")
        for key, value in request.form.items():
            print(f"  {key}: {value}")
        # Lấy features từ form
        features = []
        for i, feature_name in enumerate(diabetes_feature_order):
            value = request.form.get(feature_name, request.form.get(f'feature_{i}', ''))
            if value == '':
                value = 0  # Giá trị mặc định nếu bỏ trống
            features.append(float(value))
        print(f"🔢 Parsed features ({len(features)}): {features}")
        # Kiểm tra số lượng features
        if len(features) != 8:
            raise ValueError(f"Expected 8 features, got {len(features)}")
        # Tạo array và predict
        data = np.array([features])
        print(f"📊 Input array shape: {data.shape}")
        prediction = diabetes_model.predict(data)
        probability = diabetes_model.predict_proba(data)[0]
        print(f"🎯 Prediction: {prediction[0]}")
        print(f"📈 Probability: {probability}")
        # Format kết quả
        if prediction[0] == 1:
            result = f"🚨 Có nguy cơ mắc tiểu đường (Xác suất: {probability[1]:.2%})"
        else:
            result = f"✅ Không có nguy cơ tiểu đường (Xác suất khỏe mạnh: {probability[0]:.2%})"
        # Debug: In ra feature importance
        if hasattr(diabetes_model, 'feature_importances_'):
            print("🔍 Feature values vs importance:")
            for name, value, importance in zip(diabetes_feature_order, features, diabetes_model.feature_importances_):
                print(f"  {name}: {value} (importance: {importance:.4f})")
    except ValueError as ve:
        print(f"❌ ValueError: {ve}")
        result = f"❌ Lỗi dữ liệu: {str(ve)}"
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"❌ Error type: {type(e)}")
        result = f"❌ Lỗi không xác định: {str(e)}"
    return render_template("index.html", 
                         prediction_text=result, 
                         feature_names=diabetes_feature_order)

@app.route("/heart", methods=["GET", "POST"])
def heart_predict():
    if request.method == "GET":
        return render_template("heart.html", feature_names=heart_feature_order)
    if heart_model is None:
        return render_template("heart.html", 
                             prediction_text="❌ Heart disease model chưa được load!", 
                             feature_names=heart_feature_order)
    try:
        # Debug: In ra tất cả dữ liệu nhận được
        print("📥 Heart form data received:")
        for key, value in request.form.items():
            print(f"  {key}: {value}")
        
        # Tạo input DataFrame cho heart disease
        input_data = {}
        for feature_name in heart_feature_order:
            value = request.form.get(feature_name, '')
            if value == '':
                value = 0  # Giá trị mặc định nếu bỏ trống
            input_data[feature_name] = [value]
        
        # Tạo DataFrame
        import pandas as pd
        input_df = pd.DataFrame(input_data)
        
        # Áp dụng label encoding cho các cột categorical
        categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        for col in categorical_columns:
            if col in input_df.columns and col in heart_label_encoders:
                try:
                    input_df[col] = heart_label_encoders[col].transform(input_df[col])
                except ValueError as e:
                    print(f"Lỗi: Giá trị '{input_df[col].iloc[0]}' không có trong training data cho cột '{col}'")
                    return render_template("heart.html", 
                                         prediction_text=f"❌ Lỗi: Giá trị không hợp lệ cho {col}", 
                                         feature_names=heart_feature_order)
        
        # Convert to numeric
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
        print(f"🔢 Parsed heart features: {input_df.values[0]}")
        
        # Predict
        prediction = heart_model.predict(input_df)[0]
        probability = heart_model.predict_proba(input_df)[0]
        print(f"🎯 Heart prediction: {prediction}")
        print(f"📈 Heart probability: {probability}")
        
        # Format kết quả
        if prediction == 1:
            result = f"🚨 Có nguy cơ mắc bệnh tim (Xác suất: {probability[1]:.2%})"
        else:
            result = f"✅ Không có nguy cơ bệnh tim (Xác suất khỏe mạnh: {probability[0]:.2%})"
        
        # Debug: In ra feature importance
        if hasattr(heart_model, 'feature_importances_'):
            print("🔍 Heart feature values vs importance:")
            for name, value, importance in zip(heart_feature_order, input_df.values[0], heart_model.feature_importances_):
                print(f"  {name}: {value} (importance: {importance:.4f})")
                
    except ValueError as ve:
        print(f"❌ Heart ValueError: {ve}")
        result = f"❌ Lỗi dữ liệu: {str(ve)}"
    except Exception as e:
        print(f"❌ Heart Unexpected error: {e}")
        print(f"❌ Heart Error type: {type(e)}")
        result = f"❌ Lỗi không xác định: {str(e)}"
    
    return render_template("heart.html", 
                         prediction_text=result, 
                         feature_names=heart_feature_order)

@app.route("/debug")
def debug():
    """Route để debug thông tin model"""
    if diabetes_model is None and heart_model is None:
        return "❌ No models loaded"
    
    html_output = "<h2>🔧 Model Debug Info</h2>"
    
    # Diabetes model info
    if diabetes_model is not None:
        html_output += "<h3>🩺 Diabetes Model</h3><pre>"
        info = {
            "Model type": type(diabetes_model).__name__,
            "Expected HTML fields": list(diabetes_field_mapping.keys()),
            "Model features": diabetes_feature_order,
            "Field mapping": diabetes_field_mapping
        }
        
        if hasattr(diabetes_model, 'feature_importances_'):
            info["Feature importances"] = {
                name: f"{imp:.4f}" for name, imp in zip(diabetes_feature_order, diabetes_model.feature_importances_)
            }
        
        for key, value in info.items():
            html_output += f"{key}: {value}\n"
        html_output += "</pre>"
    
    # Heart model info
    if heart_model is not None:
        html_output += "<h3>❤️ Heart Disease Model</h3><pre>"
        info = {
            "Model type": type(heart_model).__name__,
            "Expected HTML fields": list(heart_field_mapping.keys()),
            "Model features": heart_feature_order,
            "Label encoders": list(heart_label_encoders.keys())
        }
        
        if hasattr(heart_model, 'feature_importances_'):
            info["Feature importances"] = {
                name: f"{imp:.4f}" for name, imp in zip(heart_feature_order, heart_model.feature_importances_)
            }
        
        for key, value in info.items():
            html_output += f"{key}: {value}\n"
        html_output += "</pre>"
    
    return html_output

@app.route("/test")
def test():
    """Route để test với dữ liệu mẫu"""
    html_output = "<h2>🧪 Model Test Cases</h2>"
    
    # Test Diabetes model
    if diabetes_model is not None:
        html_output += "<h3>🩺 Diabetes Model Tests</h3>"
        
        # Dữ liệu mẫu theo format HTML form
        diabetes_test_cases = [
            {
                "name": "Trường hợp bình thường",
                "data": {"Pregnancies": "1", "Glucose": "85", "BloodPressure": "66", "SkinThickness": "29", 
                        "Insulin": "0", "BMI": "26.6", "DiabetesPedigreeFunction": "0.351", "Age": "31"}
            },
            {
                "name": "Trường hợp có nguy cơ",
                "data": {"Pregnancies": "8", "Glucose": "183", "BloodPressure": "64", "SkinThickness": "0", 
                        "Insulin": "0", "BMI": "23.3", "DiabetesPedigreeFunction": "0.672", "Age": "32"}
            }
        ]
        
        for case in diabetes_test_cases:
            html_output += f"<h4>{case['name']}</h4>"
            html_output += f"<p><strong>Input data:</strong> {case['data']}</p>"
            
            try:
                # Convert to features array
                features = []
                for feature_name in diabetes_feature_order:
                    features.append(float(case['data'][feature_name]))
                
                # Predict
                data = np.array([features])
                prediction = diabetes_model.predict(data)[0]
                probability = diabetes_model.predict_proba(data)[0]
                
                result = "Có nguy cơ" if prediction == 1 else "Không có nguy cơ"
                html_output += f"<p><strong>Kết quả:</strong> {result}</p>"
                html_output += f"<p><strong>Xác suất:</strong> [Không: {probability[0]:.3f}, Có: {probability[1]:.3f}]</p>"
                
            except Exception as e:
                html_output += f"<p><strong>Lỗi:</strong> {e}</p>"
            
            html_output += "<hr>"
    else:
        html_output += "<p>❌ Diabetes model not loaded</p>"
    
    # Test Heart model
    if heart_model is not None:
        html_output += "<h3>❤️ Heart Disease Model Tests</h3>"
        
        # Dữ liệu mẫu theo format HTML form
        heart_test_cases = [
            {
                "name": "Trường hợp bình thường",
                "data": {"Age": "40", "Sex": "M", "ChestPainType": "ATA", "RestingBP": "140", 
                        "Cholesterol": "289", "FastingBS": "0", "RestingECG": "Normal", "MaxHR": "172", 
                        "ExerciseAngina": "N", "Oldpeak": "0", "ST_Slope": "Up"}
            },
            {
                "name": "Trường hợp có nguy cơ",
                "data": {"Age": "54", "Sex": "M", "ChestPainType": "NAP", "RestingBP": "150", 
                        "Cholesterol": "195", "FastingBS": "0", "RestingECG": "Normal", "MaxHR": "122", 
                        "ExerciseAngina": "N", "Oldpeak": "0", "ST_Slope": "Up"}
            }
        ]
        
        for case in heart_test_cases:
            html_output += f"<h4>{case['name']}</h4>"
            html_output += f"<p><strong>Input data:</strong> {case['data']}</p>"
            
            try:
                # Tạo input DataFrame
                import pandas as pd
                input_data = {}
                for feature_name in heart_feature_order:
                    input_data[feature_name] = [case['data'][feature_name]]
                
                input_df = pd.DataFrame(input_data)
                
                # Áp dụng label encoding
                categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
                for col in categorical_columns:
                    if col in input_df.columns and col in heart_label_encoders:
                        input_df[col] = heart_label_encoders[col].transform(input_df[col])
                
                # Convert to numeric
                for col in input_df.columns:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                
                # Predict
                prediction = heart_model.predict(input_df)[0]
                probability = heart_model.predict_proba(input_df)[0]
                
                result = "Có nguy cơ" if prediction == 1 else "Không có nguy cơ"
                html_output += f"<p><strong>Kết quả:</strong> {result}</p>"
                html_output += f"<p><strong>Xác suất:</strong> [Không: {probability[0]:.3f}, Có: {probability[1]:.3f}]</p>"
                
            except Exception as e:
                html_output += f"<p><strong>Lỗi:</strong> {e}</p>"
            
            html_output += "<hr>"
    else:
        html_output += "<p>❌ Heart model not loaded</p>"
    
    return html_output

if __name__ == "__main__":
    app.run(debug=True)