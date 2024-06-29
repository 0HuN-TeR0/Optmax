import os

class Config:  
    SECRET_KEY = os.environ.get('486837') or '486837'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://postgres:486837@localhost:5432/opt-max'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'