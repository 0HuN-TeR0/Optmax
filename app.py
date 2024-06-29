from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from extensions import db
from models import User, GPU, Blog
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, FloatField, TextAreaField
from wtforms.validators import InputRequired, Email, EqualTo, NumberRange
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from flask import Flask, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Email, EqualTo
from models import User, GPU, Blog
import bcrypt
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo
from sqlalchemy.exc import SQLAlchemyError

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or '486837'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://postgres:486837@localhost/opt-max'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
# Load and preprocess data
data = pd.read_csv(r'D:\CollegeFiles\Sem_6\trial_optmax_webapp\datas\cleaned_dropped_data.csv')
label_encoder = LabelEncoder()
data['memType'] = label_encoder.fit_transform(data['memType'])

# Define ranges for each feature
price_ranges = ['0-200', '201-400', '401-600', '601-800', '801-1000', '1000+', 'Custom']
mem_size_range = ['Less than 1', '1', '2', '4', '6', '8', '12', '16', '24', 'Custom']
gpu_clock_range = ['0-200', '201-400', '401-600', '601-800', '801-1000', '1001-1200', '1200+', 'Custom']
mem_clock_range = ['0-1000', '1001-2000', '2001-3000', '3001-4000', '4001-5000', '5000+', 'Custom']
unified_shader_range = ['0-500', '501-1000', '1001-1500', '1501-2000', '2001-2500', '2500+', 'Custom']
release_year_range = sorted(set(int(year) for year in data['releaseYear'].unique())) + ['Custom']
mem_type_range = list(label_encoder.classes_) + ['Custom']

predefined_profiles = {
    "Student": {
        "price": "201-400",
        "mem_size": "4",
        "gpu_clock": "801-1000",
        "mem_clock": "2001-3000",
        "unified_shader": "501-1000",
        "release_year": max(release_year_range[:-1]) - 2,  # 2 years old
        "mem_type": "GDDR5"
    },
    "Business": {
        "price": "401-600",
        "mem_size": "6",
        "gpu_clock": "1001-1200",
        "mem_clock": "3001-4000",
        "unified_shader": "1001-1500",
        "release_year": max(release_year_range[:-1]) - 1,  # 1 year old
        "mem_type": "GDDR6"
    },
    "Gamer": {
        "price": "801-1000",
        "mem_size": "8",
        "gpu_clock": "1200+",
        "mem_clock": "4001-5000",
        "unified_shader": "2001-2500",
        "release_year": max(release_year_range[:-1]),  # Latest
        "mem_type": "GDDR6X"
    },
    "Casual": {
        "price": "0-200",
        "mem_size": "2",
        "gpu_clock": "601-800",
        "mem_clock": "1001-2000",
        "unified_shader": "0-500",
        "release_year": max(release_year_range[:-1]) - 3,  # 3 years old
        "mem_type": "GDDR5"
    }
}

# KNN model
X = data[['price', 'memSize', 'gpuClock', 'memClock', 'unifiedShader', 'releaseYear', 'memType']]
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X)

def recommend_gpu(user_input):
    distances, indices = knn.kneighbors([user_input])
    if len(indices[0]) == 0:
        distances = ((X - user_input) ** 2).sum(axis=1).argsort()
        return data.iloc[distances[:5]]
    return data.iloc[indices[0]]

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Role', choices=[('user', 'User'), ('admin', 'Admin'),('general','General')], validators=[DataRequired()])
    submit = SubmitField('Sign Up')


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[InputRequired()])

class GPUForm(FlaskForm):
    name = StringField('Name', validators=[InputRequired()])
    specs = TextAreaField('Specs', validators=[InputRequired()])
    price = FloatField('Price', validators=[InputRequired(), NumberRange(min=0)])
    image = StringField('Image URL') # Placeholder for file upload handling
    
@app.route('/')
def home():
    return render_template('index.html')
# @app.route('/gpus', methods=['GET'])
# def gpus():
#     offset = request.args.get('offset', default=0, type=int)
#     limit = request.args.get('limit', default=6, type=int)
#     gpus = GPU.query.offset(offset).limit(limit).all()
#     gpu_data = [{
#         'id': gpu.id,
#         'manufacturer': gpu.manufacturer,
#         'productname': gpu.productname,
#         'price': gpu.price,
#         'picture': gpu.picture,
#         'memSize': gpu.memsize,
#         'gpuClock': gpu.gpuclock,
#         'memClock': gpu.memclock,
#         'unifiedShader': gpu.unifiedshader,
#         'releaseYear': gpu.releaseyear,
#         'memType': gpu.memtype,
#         'memBusWidth': gpu.membuswidth,
#         'rop': gpu.rop,
#         'pixelShader': gpu.pixelshader,
#         'vertexShader': gpu.vertexshader,
#         'igp': gpu.igp,
#         'bus': gpu.bus,
#         'gpuChip': gpu.gpuchip,
#         'G3Dmark': gpu.g3dmark,
#         'G2Dmark': gpu.g2dmark,
#         'gpuValue': gpu.gpuvalue,
#         'TDP': gpu.tdp,
#         'powerPerformance': gpu.powerperformance,
#         'testDate': gpu.testdate,
#         'category': gpu.category
#     } for gpu in gpus]
#     more_gpus = GPU.query.count() > offset + limit
#     next_offset = offset + limit if more_gpus else None
    
#     return render_template('gpus.html', 
#                            gpus=gpu_data, 
#                            more_gpus=more_gpus, 
#                            next_offset=next_offset)

@app.route('/gpus', methods=['GET'])
def gpus():
    offset = request.args.get('offset', default=0, type=int)
    limit = request.args.get('limit', default=6, type=int)
    gpus = GPU.query.offset(offset).limit(limit).all()
    gpu_data = [{
         'id': gpu.id,
        'manufacturer': gpu.manufacturer,
        'productname': gpu.productname,
        'price': gpu.price,
        'picture': gpu.picture,
        'memSize': gpu.memsize,
        'gpuClock': gpu.gpuclock,
        'memClock': gpu.memclock,
        'unifiedShader': gpu.unifiedshader,
        'releaseYear': gpu.releaseyear,
        'memType': gpu.memtype,
        'memBusWidth': gpu.membuswidth,
        'rop': gpu.rop,
        'pixelShader': gpu.pixelshader,
        'vertexShader': gpu.vertexshader,
        'igp': gpu.igp,
        'bus': gpu.bus,
        'gpuChip': gpu.gpuchip,
        'G3Dmark': gpu.g3dmark,
        'G2Dmark': gpu.g2dmark,
        'gpuValue': gpu.gpuvalue,
        'TDP': gpu.tdp,
        'powerPerformance': gpu.powerperformance,
        'testDate': gpu.testdate,
        'category': gpu.category
    } for gpu in gpus]
    more_gpus = GPU.query.count() > offset + limit
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(gpus=gpu_data, more_gpus=more_gpus)
    else:
        return render_template('gpus.html', 
                               gpus=gpu_data, 
                               more_gpus=more_gpus, 
                               next_offset=offset + limit if more_gpus else None)
@app.route('/gpu/<int:id>', methods=['GET'])
def gpu_details(id):
    gpu = GPU.query.get_or_404(id)
    gpu_data = {
        'id': gpu.id,
        'manufacturer': gpu.manufacturer,
        'productname': gpu.productname,
        'price': gpu.price,
        'picture': gpu.picture,
        'memsize': gpu.memsize,
        'gpuclock': gpu.gpuclock,
        'memclock': gpu.memclock,
        'unifiedshader': gpu.unifiedshader,
        'releaseyear': gpu.releaseyear,
        'memtype': gpu.memtype,
        'membuswidth': gpu.membuswidth,
        'rop': gpu.rop,
        'pixelshader': gpu.pixelshader,
        'vertexshader': gpu.vertexshader,
        'igp': gpu.igp,
        'bus': gpu.bus,
        'gpuchip': gpu.gpuchip,
        'g3dmark': gpu.g3dmark,
        'g2dmark': gpu.g2dmark,
        'gpuvalue': gpu.gpuvalue,
        'tdp': gpu.tdp,
        'powerperformance': gpu.powerperformance,
        'testdate': gpu.testdate,
        'category': gpu.category
    }
    return jsonify(gpu_data)

@app.route('/add_gpu', methods=['POST'])
def add_gpu():
    data = request.form
    new_gpu = GPU(
        manufacturer=data['manufacturer'],
        productname=data['productName'],
        price=data['price'],
        memsize=data['memSize'],
        gpuclock=data['gpuClock'],
        memclock=data['memClock'],
        unifiedshader=data['unifiedShader'],
        releaseyear=data['releaseYear'],
        memtype=data['memType'],
        picture=data['picture'],
        membuswidth=data.get('memBusWidth'),
        rop=data.get('rop'),
        pixelshader=data.get('pixelShader'),
        vertexshader=data.get('vertexShader'),
        igp=bool(data.get('igp')),
        bus=data.get('bus'),
        gpuchip=data.get('gpuChip'),
        g3dmark=data.get('G3Dmark'),
        g2dmark=data.get('G2Dmark'),
        gpuvalue=data.get('gpuValue'),
        tdp=data.get('TDP'),
        powerperformance=data.get('powerPerformance'),
        testdate=data.get('testDate'),
        category=data.get('category')
    )
    db.session.add(new_gpu)
    db.session.commit()
    return jsonify(success=True)


@app.route('/blogs')
def blogs():
    blogs = Blog.query.all()
    return render_template('blogs.html', blogs=blogs)

@app.route('/for-you', methods=['GET', 'POST'])
def for_you():
    recommendations = None
    if request.method == 'POST':
        user_input = []

        # Check if a predefined profile is selected
        selected_profile = request.form.get('profile', 'Custom')
        if (selected_profile != 'Custom'):
            profile = predefined_profiles[selected_profile]
        else:
            profile = None

        # Price
        price_input = profile['price'] if profile else request.form['price']
        if price_input == 'Custom':
            price = float(request.form['custom_price'])
        else:
            try:
                price = float(price_input.split('-')[1])
            except (IndexError, ValueError):
                price = float(price_input.split('+')[0])
        user_input.append(price)

        # Memory Size
        mem_size_input = profile['mem_size'] if profile else request.form['mem_size']
        if mem_size_input == 'Custom':
            mem_size = float(request.form['custom_mem_size'])
        elif mem_size_input == 'Less than 1':
            mem_size = 0.5
        else:
            mem_size = float(mem_size_input)
        user_input.append(mem_size)

        # GPU Clock
        gpu_clock_input = profile['gpu_clock'] if profile else request.form['gpu_clock']
        if gpu_clock_input == 'Custom':
            gpu_clock = float(request.form['custom_gpu_clock'])
        else:
            try:
                gpu_clock = float(gpu_clock_input.split('-')[1])
            except (IndexError, ValueError):
                gpu_clock = float(gpu_clock_input.split('+')[0])
        user_input.append(gpu_clock)

        # Memory Clock
        mem_clock_input = profile['mem_clock'] if profile else request.form['mem_clock']
        if mem_clock_input == 'Custom':
            mem_clock = float(request.form['custom_mem_clock'])
        else:
            try:
                mem_clock = float(mem_clock_input.split('-')[1])
            except (IndexError, ValueError):
                mem_clock = float(mem_clock_input.split('+')[0])
        user_input.append(mem_clock)

        # Unified Shader
        unified_shader_input = profile['unified_shader'] if profile else request.form['unified_shader']
        if unified_shader_input == 'Custom':
            unified_shader = float(request.form['custom_unified_shader'])
        else:
            try:
                unified_shader = float(unified_shader_input.split('-')[1])
            except (IndexError, ValueError):
                unified_shader = float(unified_shader_input.split('+')[0])
        user_input.append(unified_shader)

        # Release Year
        release_year_input = profile['release_year'] if profile else request.form['release_year']
        if release_year_input == 'Custom':
            release_year = int(request.form['custom_release_year'])
        else:

            release_year = int(release_year_input)
        user_input.append(release_year)

        # Memory Type
        mem_type_input = profile['mem_type'] if profile else request.form['mem_type']
        if mem_type_input == 'Custom':
            mem_type = request.form['custom_mem_type']
        else:
            mem_type = mem_type_input
        user_input.append(label_encoder.transform([mem_type])[0])

        # Ensure all inputs are valid numbers
        user_input = [float(x) if isinstance(x, (int, float)) else 0 for x in user_input]
        recommendations = recommend_gpu(user_input)[['manufacturer', 'productName', 'price', 'memSize', 'gpuClock', 'memClock', 'unifiedShader', 'releaseYear', 'memType']].to_html()

    return render_template('for_you.html',
                           price_ranges=price_ranges,
                           mem_size_range=mem_size_range,
                           gpu_clock_range=gpu_clock_range,
                           mem_clock_range=mem_clock_range,
                           unified_shader_range=unified_shader_range,
                           release_year_range=release_year_range,
                           mem_type_range=mem_type_range,
                           recommendations=recommendations,
                           predefined_profiles=predefined_profiles)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            session['user_id'] = user.id
            flash('Login successful', 'success')
            return redirect(url_for('profile'))
        flash('Invalid email or password', 'danger')
    return render_template('login.html', form=form)
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash('Email already registered. Please use a different email.', 'danger')
            return render_template('register.html', form=form)
        
        hashed_password = generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password, role=form.role.data)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except SQLAlchemyError as e:
            db.session.rollback()
            flash('An error occurred. Please try again.', 'danger')
            app.logger.error(f"Error during registration: {str(e)}")
    
    return render_template('register.html', form=form)
def logout():
    session.pop('user_id', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('Please login to access your profile', 'warning')
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    return render_template('profile.html', user=user)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)