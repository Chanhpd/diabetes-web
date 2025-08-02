from app import app, db
from models import Product

def seed_products():
    with app.app_context():
        # Xóa tất cả sản phẩm cũ
        Product.query.delete()
        
        # Sản phẩm cho người tiểu đường
        products = [
            Product(
                name='Máy đo đường huyết Omron',
                description='Máy đo đường huyết chính xác, dễ sử dụng với bộ nhớ lưu trữ 500 kết quả đo',
                price=1200000,
                image_url='https://images.unsplash.com/photo-1550572017-edd951aa4fdc?auto=format&fit=crop&w=200&q=80',
                category='diabetes'
            ),
            Product(
                name='Thực phẩm chức năng kiểm soát đường huyết',
                description='Hỗ trợ kiểm soát đường huyết, giảm cholesterol với thành phần tự nhiên',
                price=650000,
                image_url='https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?auto=format&fit=crop&w=200&q=80',
                category='diabetes'
            ),
            Product(
                name='Máy đo huyết áp tự động',
                description='Máy đo huyết áp điện tử chính xác, có cảnh báo rối loạn nhịp tim',
                price=850000,
                image_url='https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&w=200&q=80',
                category='diabetes'
            ),
            # Sản phẩm cho người bệnh tim
            Product(
                name='Máy đo nhịp tim đeo tay',
                description='Đồng hồ thông minh theo dõi nhịp tim 24/7, cảnh báo bất thường',
                price=2500000,
                image_url='https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&w=200&q=80',
                category='heart'
            ),
            Product(
                name='Omega-3 hỗ trợ tim mạch',
                description='Viên uống Omega-3 cao cấp, hỗ trợ sức khỏe tim mạch và giảm cholesterol',
                price=450000,
                image_url='https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?auto=format&fit=crop&w=200&q=80',
                category='heart'
            ),
            Product(
                name='Máy đo cholesterol tại nhà',
                description='Thiết bị đo cholesterol nhanh chóng, kết quả chính xác trong 2 phút',
                price=1800000,
                image_url='https://images.unsplash.com/photo-1550572017-edd951aa4fdc?auto=format&fit=crop&w=200&q=80',
                category='heart'
            )
        ]
        
        # Thêm sản phẩm vào database
        for product in products:
            db.session.add(product)
        
        # Lưu thay đổi
        db.session.commit()
        print("✅ Đã thêm sản phẩm mẫu vào database!")

if __name__ == "__main__":
    seed_products()
