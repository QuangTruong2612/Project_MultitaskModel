import os
import zipfile
import boto3
from urllib.parse import urlparse
from pathlib import Path
from src.Project_MultitaskModel.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Tải file ZIP từ S3 về máy local
        """
        if not os.path.exists(self.config.local_data_file):
            # Parse URL: s3://bucket-name/path/to/data.zip
            parsed_url = urlparse(self.config.source_URL)
            bucket_name = parsed_url.netloc
            s3_key = parsed_url.path.lstrip('/')

            print(f"Downloading {s3_key} from bucket {bucket_name}...")

            try:
                s3 = boto3.client('s3')
                s3.download_file(bucket_name, s3_key, str(self.config.local_data_file))
                print(f"✅ Downloaded to: {self.config.local_data_file}")
            except Exception as e:
                print(f"❌ Error downloading: {e}")
                raise e
        else:
            print(f"File already exists: {self.config.local_data_file}")

    def extract_zip_file(self):
        """
        Giải nén file zip vào thư mục unzip_dir
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        print(f"Extracting zip file to: {unzip_path}...")
        
        try:
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            print("✅ Extracted successfully!")
        except Exception as e:
            print(f"❌ Error extracting: {e}")
            raise e