from pathlib import Path

# Giả sử script python_file1 nằm trong folder dir_1
# Lấy đường dẫn của file python_file1
current_dir = Path(__file__).parent

# Tạo đường dẫn đến model1
model_path = str(current_dir.parent) + '/models/convnext_rac_model.pth'

print("Đường dẫn tới model1:", model_path)