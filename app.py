from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
try:
    model = joblib.load("diabetes_model.pkl")
    print("✅ Model loaded successfully!")
    
    # Mapping từ tên field trong HTML sang tên chuẩn
    field_mapping = {
        'npreg': 'Pregnancies',
        'glu': 'Glucose', 
        'bp': 'BloodPressure',
        'skin': 'SkinThickness',
        'insu': 'Insulin',
        'bmi': 'BMI',
        'ped': 'DiabetesPedigreeFunction',
        'age': 'Age'
    }
    
    # Thứ tự features cho model (theo thứ tự training)
    feature_order = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    print(f"📊 Expected features: {feature_order}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    field_mapping = {}
    feature_order = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", 
                             prediction_text="❌ Model chưa được load!")
    
    try:
        # Debug: In ra tất cả dữ liệu nhận được
        print("📥 Form data received:")
        for key, value in request.form.items():
            print(f"  {key}: '{value}'")
        
        # Lấy và convert features theo đúng thứ tự
        features = []
        for i, feature_name in enumerate(feature_order):
            value = request.form.get(feature_name, request.form.get(f'feature_{i}', ''))
            if value == '':
                value = 0  # Giá trị mặc định nếu bỏ trống
            features.append(float(value))
        
        print(f"🔢 Final features ({len(features)}): {features}")
        
        # Kiểm tra số lượng features
        if len(features) != 8:
            raise ValueError(f"Cần 8 giá trị, nhận được {len(features)}")
        
        # Validate ranges (optional)
        validations = [
            (features[0] >= 0, "Số lần mang thai phải >= 0"),
            (features[1] > 0, "Nồng độ glucose phải > 0"), 
            (features[2] > 0, "Huyết áp phải > 0"),
            (features[4] >= 0, "Insulin phải >= 0"),
            (features[5] > 0, "BMI phải > 0"),
            (features[6] >= 0, "Chỉ số di truyền phải >= 0"),
            (features[7] > 0, "Tuổi phải > 0")
        ]
        
        for is_valid, error_msg in validations:
            if not is_valid:
                raise ValueError(error_msg)
        
        # Tạo array và predict
        data = np.array([features])
        print(f"📊 Input array shape: {data.shape}")
        
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0]
        
        print(f"🎯 Prediction: {prediction}")
        print(f"📈 Probability: {probability}")
        
        # Format kết quả với emoji và phần trăm
        if prediction == 1:
            result = f"🚨 Có nguy cơ mắc tiểu đường (Xác suất: {probability[1]:.1%})"
        else:
            result = f"✅ Không có nguy cơ tiểu đường (Xác suất khỏe mạnh: {probability[0]:.1%})"
        
        # Thêm thông tin chi tiết
        result += f"\n\n📊 Chi tiết dự đoán:"
        result += f"\n• Xác suất không mắc: {probability[0]:.1%}"
        result += f"\n• Xác suất mắc bệnh: {probability[1]:.1%}"
        
        # Highlight các yếu tố quan trọng nếu có feature_importances_
        if hasattr(model, 'feature_importances_'):
            important_features = []
            for i, (name, value, importance) in enumerate(zip(feature_order, features, model.feature_importances_)):
                if importance > 0.1:  # Chỉ hiển thị features quan trọng
                    important_features.append(f"• {name}: {value} (tầm quan trọng: {importance:.1%})")
            
            if important_features:
                result += f"\n\n🔍 Các yếu tố quan trọng:\n" + "\n".join(important_features[:3])
        
    except ValueError as ve:
        print(f"❌ ValueError: {ve}")
        result = f"❌ Lỗi dữ liệu: {str(ve)}"
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        result = f"❌ Lỗi hệ thống: {str(e)}"
    
    return render_template("index.html", prediction_text=result)

@app.route("/debug")
def debug():
    """Route để debug thông tin model"""
    if model is None:
        return "❌ Model not loaded"
    
    info = {
        "Model type": type(model).__name__,
        "Expected HTML fields": list(field_mapping.keys()),
        "Model features": feature_order,
        "Field mapping": field_mapping
    }
    
    if hasattr(model, 'feature_importances_'):
        info["Feature importances"] = {
            name: f"{imp:.4f}" for name, imp in zip(feature_order, model.feature_importances_)
        }
    
    html_output = "<h2>🔧 Model Debug Info</h2><pre>"
    for key, value in info.items():
        html_output += f"{key}: {value}\n"
    html_output += "</pre>"
    
    return html_output

@app.route("/test")
def test():
    """Route để test với dữ liệu mẫu"""
    if model is None:
        return "❌ Model not loaded"
    
    # Dữ liệu mẫu theo format HTML form
    test_cases = [
        {
            "name": "Trường hợp bình thường",
            "data": {"npreg": "1", "glu": "85", "bp": "66", "skin": "29", 
                    "insu": "0", "bmi": "26.6", "ped": "0.351", "age": "31"}
        },
        {
            "name": "Trường hợp có nguy cơ",
            "data": {"npreg": "8", "glu": "183", "bp": "64", "skin": "0", 
                    "insu": "0", "bmi": "23.3", "ped": "0.672", "age": "32"}
        }
    ]
    
    html_output = "<h2>🧪 Model Test Cases</h2>"
    
    for case in test_cases:
        html_output += f"<h3>{case['name']}</h3>"
        html_output += f"<p><strong>Input data:</strong> {case['data']}</p>"
        
        try:
            # Convert to features array
            features = []
            for field in ['npreg', 'glu', 'bp', 'skin', 'insu', 'bmi', 'ped', 'age']:
                features.append(float(case['data'][field]))
            
            # Predict
            data = np.array([features])
            prediction = model.predict(data)[0]
            probability = model.predict_proba(data)[0]
            
            result = "Có nguy cơ" if prediction == 1 else "Không có nguy cơ"
            html_output += f"<p><strong>Kết quả:</strong> {result}</p>"
            html_output += f"<p><strong>Xác suất:</strong> [Không: {probability[0]:.3f}, Có: {probability[1]:.3f}]</p>"
            
        except Exception as e:
            html_output += f"<p><strong>Lỗi:</strong> {e}</p>"
        
        html_output += "<hr>"
    
    return html_output

if __name__ == "__main__":
    app.run(debug=True)