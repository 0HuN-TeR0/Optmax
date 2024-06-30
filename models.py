from flask_login import UserMixin

from extensions import db


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    preferences = db.relationship('Preference', backref='users', lazy=True)


class Preference(db.Model):
    __tablename__ = 'preferences'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    price_range = db.Column(db.String(50))
    mem_size = db.Column(db.Integer)
    gpu_clock_range = db.Column(db.String(50))
    mem_clock_range = db.Column(db.String(50))
    unified_shader_range = db.Column(db.String(50))
    release_year = db.Column(db.Integer)
    mem_type = db.Column(db.String(50))


class GPU(db.Model):
    __tablename__ = 'gpu_data'
    id = db.Column(db.Integer, primary_key=True)
    manufacturer = db.Column(db.Text)
    productname = db.Column(db.Text)
    releaseyear = db.Column(db.Float)
    memsize = db.Column(db.Float)
    membuswidth = db.Column(db.Float)
    gpuclock = db.Column(db.Float)
    memclock = db.Column(db.Float)
    unifiedshader = db.Column(db.Float)
    tmu = db.Column(db.Float)
    rop = db.Column(db.Float)
    pixelshader = db.Column(db.Float)
    vertexshader = db.Column(db.Float)
    igp = db.Column(db.Boolean)
    bus = db.Column(db.Text)
    memtype = db.Column(db.Text)
    gpuchip = db.Column(db.Text)
    g3dmark = db.Column(db.Float)
    g2dmark = db.Column(db.Float)
    price = db.Column(db.Float)
    gpuvalue = db.Column(db.Float)
    tdp = db.Column(db.Float)
    powerperformance = db.Column(db.Float)
    testdate = db.Column(db.Float)
    category = db.Column(db.Text)
    picture = db.Column(db.Text)


class Blog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False)
    content = db.Column(db.Text, nullable=False)
    author = db.Column(db.String(150), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())
    category = db.Column(db.String(100), nullable=False)
