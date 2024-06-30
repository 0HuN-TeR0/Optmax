import os
from datetime import datetime, UTC

import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify,
)
from flask_login import (
    current_user,
    login_user,
    logout_user,
    login_required,
    LoginManager,
)
from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    PasswordField,
    FloatField,
    TextAreaField,
    SubmitField,
    SelectField,
)
from wtforms.validators import (
    InputRequired,
    Email,
    EqualTo,
    NumberRange,
    DataRequired
)
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sqlalchemy.exc import SQLAlchemyError

from models import User, GPU, Blog, Preference
from extensions import db, migrate


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or '486837'
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL') or 'postgresql://postgres:486837@localhost/opt-max'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'


app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
migrate.init_app(app, db)
login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def user_loader(email):
    user = User.query.filter_by(email=email).first()
    if user is None:
        return
    return user

@login_manager.request_loader
def request_loader(request):
    email = request.form.get('email')
    user = User.query.filter_by(email=email).first()
    if user is None:
        return
    return user


def load_gpu_data():
    with app.app_context():
        gpu_data = GPU.query.all()
        data = pd.DataFrame([
            {
                'id': gpu.id,
                'manufacturer': gpu.manufacturer,
                'productName': gpu.productname,
                'price': gpu.price,
                'memSize': gpu.memsize,
                'gpuClock': gpu.gpuclock,
                'memClock': gpu.memclock,
                'unifiedShader': gpu.unifiedshader,
                'releaseYear': gpu.releaseyear,
                'memType': gpu.memtype,
            }
            for gpu in gpu_data
        ])
    return data


# Load and preprocess data
data = load_gpu_data()
# Adjust label encoding
label_encoder = LabelEncoder()
data['memType'] = label_encoder.fit_transform(data['memType'])

# Define ranges for each feature
price_ranges = ['0-200', '201-400', '401-600',
                '601-800', '801-1000', '1000+', 'Custom']
mem_size_range = ['Less than 1', '1', '2',
                  '4', '6', '8', '12', '16', '24', 'Custom']
gpu_clock_range = ['0-200', '201-400', '401-600',
                   '601-800', '801-1000', '1001-1200', '1200+', 'Custom']
mem_clock_range = ['0-1000', '1001-2000', '2001-3000',
                   '3001-4000', '4001-5000', '5000+', 'Custom']
unified_shader_range = ['0-500', '501-1000', '1001-1500',
                        '1501-2000', '2001-2500', '2500+', 'Custom']
release_year_range = sorted(
    set(int(year) for year in data['releaseYear'].unique())) + ['Custom']
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
X = data[['price', 'memSize', 'gpuClock', 'memClock',
          'unifiedShader', 'releaseYear', 'memType']]
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X)


def recommend_gpu(user_input, user=None):
    if user and user.is_authenticated:
        # Check if the user has a preference
        preference = Preference.query.filter_by(user_id=user.id).first()
        if preference:
            # Convert preferences to correct data types
            price_range_min, price_range_max = map(float, preference.price_range.split('-'))
            mem_size = float(preference.mem_size.replace('Less than ', ''))
            gpu_clock = float(preference.gpu_clock_range.split('-')[1])
            mem_clock = float(preference.mem_clock_range.split('-')[1])
            unified_shader = float(preference.unified_shader_range.split('-')[1])
            release_year = float(preference.release_year)
            mem_type = float(label_encoder.transform([preference.mem_type])[0])
            
            # Combine the user's preference with the new input
            updated_input = {
                'price': (user_input['price'] + (price_range_min + price_range_max) / 2) / 2,
                'memSize': (user_input['memSize'] + mem_size) / 2,
                'gpuClock': (user_input['gpuClock'] + gpu_clock) / 2,
                'memClock': (user_input['memClock'] + mem_clock) / 2,
                'unifiedShader': (user_input['unifiedShader'] + unified_shader) / 2,
                'releaseYear': (user_input['releaseYear'] + release_year) / 2,
                'memType': (user_input['memType'] + mem_type) / 2
            }
        else:
            # If no user is authenticated or the user has no preference, use the original user_input
            updated_input = user_input
    else:
        # If no user is authenticated or the user has no preference, use the original user_input
        updated_input = user_input

    # Perform the recommendation based on the updated input
    distances, indices = knn.kneighbors([updated_input])

    if len(indices[0]) > 0:
        recommendations = data.iloc[indices[0]]
    else:
        # If no matches are found, fall back to the regular recommendation
        distances = ((X - updated_input) ** 2).sum(axis=1).argsort()
        recommendations = data.iloc[distances[:5]]

    # Assuming 'id' is the correct column name for GPU IDs
    recommendations['details_link'] = recommendations['id'].apply(
        lambda x: f'<a href="/gpus/{x}">Details</a>')

    return recommendations



class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Role', choices=[('user', 'User'), ('admin', 'Admin'),('editor','Editor')], validators=[DataRequired()])
    submit = SubmitField('Sign Up')


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[InputRequired()])


class GPUForm(FlaskForm):
    name = StringField('Name', validators=[InputRequired()])
    specs = TextAreaField('Specs', validators=[InputRequired()])
    price = FloatField('Price', validators=[
                       InputRequired(), NumberRange(min=0)])
    image = StringField('Image URL')  # Placeholder for file upload handling

class BlogForm(FlaskForm):
    title = StringField('Title', validators=[InputRequired()])
    content = TextAreaField('Content', validators=[InputRequired()])
    category = StringField('Category', validators=[InputRequired()])
    # image = StringField('Image URL')  # Placeholder for file upload handling


@app.route('/')
def home():
    return render_template('index.html')

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


@app.route('/gpus/<int:id>', methods=['GET'])
def gpus_details(id):
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
    return render_template('gpu_details.html', gpu=gpu_data)


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

    if 'user_id' not in session:
        user_dict = {"role": "normal"}
    else:
        user = User.query.get(session['user_id'])
        user_dict = {"role": user.role}

    # Convert blog objects to dictionaries with all necessary fields
    blog_list = []
    for blog in blogs:
        blog_dict = {
            'title': blog.title,
            'content': blog.content,
            'author': blog.author,
            'date': blog.date,
            'category': blog.category,
            # 'image_url': blog.image_url  # Assuming you have an image_url field
        }
        blog_list.append(blog_dict)

    return render_template(
        'blogs.html',
        blogs=blog_list[::-1],
        user = user_dict,
        )


@app.route('/add_blog', methods=["GET", "POST"])
# @login_required
def add_blog():
    if request.method == "GET":
        return redirect(url_for(''))

    if 'user_id' not in session:
        flash('Please login to access page', 'warning')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])

    if user.role not in {"admin", "editor"}:
        return redirect(url_for('blogs'))

    form = BlogForm()
    # print("*" * 100)
    # print(form.category.data)
    # print(dir(form))
    # print(form.validate_on_submit())
    # if not form.validate_on_submit():
    #     return render_template('blogs.html')
    
    # print("_" * 100)
    new_blog = Blog(
        title=form.title.data,
        content=form.content.data,
        author=user.email,
        date=datetime.now(UTC),
        category=form.category.data,
    )
    try:
        db.session.add(new_blog)
        db.session.commit()
        db.session.refresh(new_blog)
        flash('Blog added successful.', 'success')
    except SQLAlchemyError as e:
        db.session.rollback()
        flash('An error occurred. Please try again.', 'danger')
        app.logger.error(f"Error during posting blog: {str(e)}")
        return redirect(url_for('blogs'))
    new_blog_dict = new_blog.as_dict()
    res = {
            "success": True,
            "blog": new_blog_dict,
    }
    return jsonify(res)
    # Convert blog objects to dictionaries with all necessary fields
    


@app.route('/for-you', methods=['GET', 'POST'])
def for_you():
    recommendations = None
    if request.method == 'POST':
        user_input = []

        # Check if a predefined profile is selected
        selected_profile = request.form.get('profile', 'Custom')
        if selected_profile != 'Custom':
            profile = predefined_profiles[selected_profile]
        else:
            profile = None

        # Extract form data
        price_input = profile['price'] if profile else request.form['price']
        price = float(request.form['custom_price']) if price_input == 'Custom' else float(price_input.split('-')[1]) if '-' in price_input else float(price_input.split('+')[0])
        user_input.append(price)

        mem_size_input = profile['mem_size'] if profile else request.form['mem_size']
        mem_size = float(request.form['custom_mem_size']) if mem_size_input == 'Custom' else 0.5 if mem_size_input == 'Less than 1' else float(mem_size_input)
        user_input.append(mem_size)

        gpu_clock_input = profile['gpu_clock'] if profile else request.form['gpu_clock']
        gpu_clock = float(request.form['custom_gpu_clock']) if gpu_clock_input == 'Custom' else float(gpu_clock_input.split('-')[1]) if '-' in gpu_clock_input else float(gpu_clock_input.split('+')[0])
        user_input.append(gpu_clock)

        mem_clock_input = profile['mem_clock'] if profile else request.form['mem_clock']
        mem_clock = float(request.form['custom_mem_clock']) if mem_clock_input == 'Custom' else float(mem_clock_input.split('-')[1]) if '-' in mem_clock_input else float(mem_clock_input.split('+')[0])
        user_input.append(mem_clock)

        unified_shader_input = profile['unified_shader'] if profile else request.form['unified_shader']
        unified_shader = float(request.form['custom_unified_shader']) if unified_shader_input == 'Custom' else float(unified_shader_input.split('-')[1]) if '-' in unified_shader_input else float(unified_shader_input.split('+')[0])
        user_input.append(unified_shader)

        release_year_input = profile['release_year'] if profile else request.form['release_year']
        release_year = int(request.form['custom_release_year']) if release_year_input == 'Custom' else int(release_year_input)
        user_input.append(release_year)

        mem_type_input = profile['mem_type'] if profile else request.form['mem_type']
        mem_type = request.form['custom_mem_type'] if mem_type_input == 'Custom' else mem_type_input
        user_input.append(label_encoder.transform([mem_type])[0])

        user_input = [float(x) if isinstance(x, (int, float)) else 0 for x in user_input]
        recommendations = recommend_gpu(user_input)[['manufacturer', 'productName', 'price', 'memSize', 'gpuClock',
                                                     'memClock', 'unifiedShader', 'releaseYear', 'memType', 'details_link']].to_html(escape=False)

        # Check for AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return recommendations

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


@app.route('/save_preferences', methods=['POST'])
def save_preferences():
    if 'user_id' not in session:
        flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    log_file_path = request.form.get('log_file_path')

    try:
        # Extract form data
        price_range = request.form.get('price')
        mem_size = request.form.get('mem_size')
        gpu_clock_range = request.form.get('gpu_clock')
        mem_clock_range = request.form.get('mem_clock')
        unified_shader_range = request.form.get('unified_shader')
        release_year = request.form.get('release_year')
        mem_type = request.form.get('mem_type')
        custom_price = request.form.get('custom_price')
        custom_mem_size = request.form.get('custom_mem_size')
        custom_gpu_clock = request.form.get('custom_gpu_clock')
        custom_mem_clock = request.form.get('custom_mem_clock')
        custom_unified_shader = request.form.get('custom_unified_shader')
        custom_release_year = request.form.get('custom_release_year')
        custom_mem_type = request.form.get('custom_mem_type')

        # Prepare preferences dictionary
        preferences = {
            'price_range': price_range,
            'custom_price': custom_price,
            'mem_size': mem_size,
            'custom_mem_size': custom_mem_size,
            'gpu_clock_range': gpu_clock_range,
            'custom_gpu_clock': custom_gpu_clock,
            'mem_clock_range': mem_clock_range,
            'custom_mem_clock': custom_mem_clock,
            'unified_shader_range': unified_shader_range,
            'custom_unified_shader': custom_unified_shader,
            'release_year': release_year,
            'custom_release_year': custom_release_year,
            'mem_type': mem_type,
            'custom_mem_type': custom_mem_type,
        }

        # Log preferences to a file
        if log_file_path:
            try:
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"{preferences}\n")
            except IOError as e:
                return jsonify(success=False, message=f'Error writing to log file: {str(e)}'), 500

        # Check if the user already has preferences
        existing_preference = Preference.query.filter_by(user_id=user.id).order_by(Preference.id.desc()).first()

        # Prepare new preferences
        new_preference = Preference(
            user_id=user.id,
            price_range=price_range,
            mem_size=mem_size,
            gpu_clock_range=gpu_clock_range,
            mem_clock_range=mem_clock_range,
            unified_shader_range=unified_shader_range,
            release_year=release_year,
            mem_type=mem_type,
        )

        if existing_preference and (
            existing_preference.price_range == new_preference.price_range and
            existing_preference.mem_size == new_preference.mem_size and
            existing_preference.gpu_clock_range == new_preference.gpu_clock_range and
            existing_preference.mem_clock_range == new_preference.mem_clock_range and
            existing_preference.unified_shader_range == new_preference.unified_shader_range and
            existing_preference.release_year == new_preference.release_year and
            existing_preference.mem_type == new_preference.mem_type
        ):
            # If preferences are the same, just display success
            return jsonify(success=True, message='No changes to preferences.')

        # Save new preferences if they are different
        db.session.add(new_preference)
        db.session.commit()
        db.session.refresh(new_preference)

        return jsonify(success=True, message='Preferences saved successfully.')
    except Exception as e:
        db.session.rollback()
        print(f"Error saving preferences: {str(e)}")
        return jsonify(success=False, message='Error saving preferences.'), 500

@app.route('/get_preferences', methods=['POST'])
def get_preferences():
    log_file_path = request.form.get('log_file_path')
    if not log_file_path or not os.path.exists(log_file_path):
        return jsonify({'success': False, 'message': 'Log file does not exist.'})

    # Read from the log file
    try:
        with open(log_file_path, 'r') as log_file:
            preferences = log_file.readlines()
        
        # Parse the last entry in the log file
        last_preferences = preferences[-1] if preferences else '{}'
        last_preferences_dict = eval(last_preferences)  # Convert string to dictionary

    except IOError as e:
        return jsonify({'success': False, 'message': f'Error reading from log file: {str(e)}'})

    return jsonify({'success': True, 'message': 'Preferences loaded successfully.', 'preferences': last_preferences_dict})

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            session['user_id'] = user.id
            # login_user(user=user)
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
        new_user = User(username=form.username.data, email=form.email.data,
                        password=hashed_password, role=form.role.data)

        try:
            db.session.add(new_user)
            db.session.commit()
            db.session.refresh(new_user)
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except SQLAlchemyError as e:
            db.session.rollback()
            flash('An error occurred. Please try again.', 'danger')
            app.logger.error(f"Error during registration: {str(e)}")

    return render_template('register.html', form=form)

@app.route("/logout")
def logout():
    # Clear the log file data
    log_file_path = 'preferences_log.txt'
    open(log_file_path, 'w').close()  # Truncate the log file

    # Clear the user session
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))


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
