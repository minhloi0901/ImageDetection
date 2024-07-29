import os

def count_files(directory):
    # Liệt kê tất cả các tệp trong thư mục
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)

# Ví dụ sử dụng
directory = '../MRE/fakes'
print(f"Number of files: {count_files(directory)}")
