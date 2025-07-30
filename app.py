from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

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
    return render_template("products.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    return render_template("login.html")

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