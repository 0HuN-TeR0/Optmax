import os
from datetime import datetime, UTC

from flask_migrate import current
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
# from flask_login import (
#     current_user,
#     login_user,
#     logout_user,
#     login_required,
#     LoginManager,
# )
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
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neighbors import NearestNeighbors
from sqlalchemy.exc import SQLAlchemyError

from models import User, GPU, Blog, Preference
from extensions import db, migrate
from log_utils import create_log


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
                'tmu': gpu.tmu,  # Added TMU feature
                'rop': gpu.rop,  # Added ROP feature
                'releaseYear': gpu.releaseyear,
                'G3Dmark': gpu.g3dmark,  # Added G3Dmark feature
                'TDP': gpu.tdp,  # Added TDP feature
                'gpuValue': gpu.gpuvalue,  # Added GPU Value feature
                'memType': gpu.memtype,
            }
            for gpu in gpu_data
        ])
        return data



# Load and preprocess data
data = load_gpu_data()
scaler = StandardScaler()
features =['price', 'memSize', 'gpuClock', 'memClock',
          'unifiedShader', 'releaseYear', 'memType']
# Adjust label encoding
label_encoder = LabelEncoder()
data['memType'] = label_encoder.fit_transform(data['memType'])
data_selected = data[features]
data_normalized = pd.DataFrame(scaler.fit_transform(data_selected), columns=features)


# Define ranges for each feature based on the dataset
price_ranges = ['0-200', '201-400', '401-600', '601-800', '801-1000', '1001+', 'Custom']
mem_size_range = ['Less than 1', '1', '2', '4', '6', '8', '12', '16', '24', 'Custom']
gpu_clock_range = ['0-200', '201-400', '401-600', '601-800', '801-1000', '1001-1200', '1200+', 'Custom']
mem_clock_range = ['0-1000', '1001-2000', '2001-3000', '3001-4000', '4001-5000', '5000+', 'Custom']
unified_shader_range = ['0-500', '501-1000', '1001-1500', '1501-2000', '2001-2500', '2500+', 'Custom']
tmu_range = ['0-50', '51-100', '101-150', '151-200', '200+', 'Custom']
rop_range = ['0-25', '26-50', '51-75', '76-100', '100+', 'Custom']
release_year_range = sorted(set(int(year) for year in data['releaseYear'].unique())) + ['Custom']
g3dmark_range = ['0-5000', '5001-10000', '10001-15000', '15001-20000', '20000+', 'Custom']
tdp_range = ['0-50', '51-100', '101-150', '151-200', '200+', 'Custom']
gpu_value_range = ['0-20', '21-40', '41-60', '61-80', '80+', 'Custom']
mem_type_range = list(label_encoder.classes_) + ['Custom']
predefined_profiles = {
    "Student": {
        "price": "201-400",
        "mem_size": "4",
        "gpu_clock": "801-1000",
        "mem_clock": "2001-3000",
        "unified_shader": "501-1000",
        "tmu": "51-100",
        "rop": "26-50",
        "release_year": max(release_year_range[:-1]) - 2,  # 2 years old
        "g3dmark": "5001-10000",
        "tdp": "51-100",
        "gpu_value": "21-40",
        "mem_type": "GDDR5"
    },
    "Business": {
        "price": "401-600",
        "mem_size": "6",
        "gpu_clock": "1001-1200",
        "mem_clock": "3001-4000",
        "unified_shader": "1001-1500",
        "tmu": "101-150",
        "rop": "51-75",
        "release_year": max(release_year_range[:-1]) - 1,  # 1 year old
        "g3dmark": "10001-15000",
        "tdp": "101-150",
        "gpu_value": "21-40",
        "mem_type": "GDDR6"
    },
    "Gamer": {
        "price": "801-1000",
        "mem_size": "8",
        "gpu_clock": "1200+",
        "mem_clock": "4001-5000",
        "unified_shader": "2001-2500",
        "tmu": "151-200",
        "rop": "76-100",
        "release_year": max(release_year_range[:-1]),  # Latest
        "g3dmark": "20000+",
        "tdp": "200+",
        "gpu_value": "41-60",
        "mem_type": "GDDR6X"
    },
    "Casual": {
        "price": "0-200",
        "mem_size": "2",
        "gpu_clock": "601-800",
        "mem_clock": "1001-2000",
        "unified_shader": "0-500",
        "tmu": "0-50",
        "rop": "0-25",
        "release_year": max(release_year_range[:-1]) - 3,  # 3 years old
        "g3dmark": "0-5000",
        "tdp": "0-50",
        "gpu_value": "0-20",
        "mem_type": "GDDR5"
    }
}


# KNN model
# Normalize the data




X = data_normalized[['price', 'memSize', 'gpuClock', 'memClock',
          'unifiedShader', 'releaseYear', 'memType']]
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X)


def recommend_gpu(user_input, user=None):
    if user and user.is_authenticated:
        # Check if the user has a preference
        preference = Preference.query.filter_by(user_id=user.id).first()
        if preference:
            # Convert preferences to correct data types
            price_range_min, price_range_max = map(
                float, preference.price_range.split('-'))
            mem_size = float(preference.mem_size.replace('Less than ', ''))
            gpu_clock = float(preference.gpu_clock_range.split('-')[1])
            mem_clock = float(preference.mem_clock_range.split('-')[1])
            unified_shader = float(
                preference.unified_shader_range.split('-')[1])
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
    print(distances)

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
        price = float(request.form['custom_price']) if price_input == 'Custom' else float(
            price_input.split('-')[1]) if '-' in price_input else float(price_input.split('+')[0])
        user_input.append(price)

        mem_size_input = profile['mem_size'] if profile else request.form['mem_size']
        mem_size = float(request.form['custom_mem_size']
                         ) if mem_size_input == 'Custom' else 0.5 if mem_size_input == 'Less than 1' else float(mem_size_input)
        user_input.append(mem_size)

        gpu_clock_input = profile['gpu_clock'] if profile else request.form['gpu_clock']
        gpu_clock = float(request.form['custom_gpu_clock']) if gpu_clock_input == 'Custom' else float(
            gpu_clock_input.split('-')[1]) if '-' in gpu_clock_input else float(gpu_clock_input.split('+')[0])
        user_input.append(gpu_clock)

        mem_clock_input = profile['mem_clock'] if profile else request.form['mem_clock']
        mem_clock = float(request.form['custom_mem_clock']) if mem_clock_input == 'Custom' else float(
            mem_clock_input.split('-')[1]) if '-' in mem_clock_input else float(mem_clock_input.split('+')[0])
        user_input.append(mem_clock)

        unified_shader_input = profile['unified_shader'] if profile else request.form['unified_shader']
        unified_shader = float(request.form['custom_unified_shader']) if unified_shader_input == 'Custom' else float(
            unified_shader_input.split('-')[1]) if '-' in unified_shader_input else float(unified_shader_input.split('+')[0])
        user_input.append(unified_shader)

        tmu_input = profile['tmu'] if profile else request.form['tmu']
        tmu = float(request.form['custom_tmu']) if tmu_input == 'Custom' else float(
            tmu_input.split('-')[1]) if '-' in tmu_input else float(tmu_input.split('+')[0])
        user_input.append(tmu)

        rop_input = profile['rop'] if profile else request.form['rop']
        rop = float(request.form['custom_rop']) if rop_input == 'Custom' else float(
            rop_input.split('-')[1]) if '-' in rop_input else float(rop_input.split('+')[0])
        user_input.append(rop)

        release_year_input = profile['release_year'] if profile else request.form['release_year']
        release_year = int(request.form['custom_release_year']
                           ) if release_year_input == 'Custom' else int(release_year_input)
        user_input.append(release_year)

        g3dmark_input = profile['g3dmark'] if profile else request.form['g3dmark']
        g3dmark = float(request.form['custom_g3dmark']) if g3dmark_input == 'Custom' else float(
            g3dmark_input.split('-')[1]) if '-' in g3dmark_input else float(g3dmark_input.split('+')[0])
        user_input.append(g3dmark)

        tdp_input = profile['tdp'] if profile else request.form['tdp']
        tdp = float(request.form['custom_tdp']) if tdp_input == 'Custom' else float(
            tdp_input.split('-')[1]) if '-' in tdp_input else float(tdp_input.split('+')[0])
        user_input.append(tdp)

        gpu_value_input = profile['gpu_value'] if profile else request.form['gpu_value']
        gpu_value = float(request.form['custom_gpu_value']) if gpu_value_input == 'Custom' else float(
            gpu_value_input.split('-')[1]) if '-' in gpu_value_input else float(gpu_value_input.split('+')[0])
        user_input.append(gpu_value)

        mem_type_input = profile['mem_type'] if profile else request.form['mem_type']
        mem_type = request.form['custom_mem_type'] if mem_type_input == 'Custom' else mem_type_input
        user_input.append(label_encoder.transform([mem_type])[0])

        user_input = [float(x) if isinstance(
            x, (int, float)) else 0 for x in user_input]
        recommendations_df = recommend_gpu(user_input)[['manufacturer', 'productName', 'price', 'memSize', 'gpuClock',
                                                        'memClock', 'unifiedShader', 'tmu', 'rop', 'releaseYear', 'G3Dmark', 'TDP', 'gpuValue', 'memType', 'details_link']]
        if 'user_id' in session:
            print(session['user_id'])
            create_log(
                user_id=session['user_id'],
                ts=datetime.now(UTC),
                data=recommendations_df.to_dict()
            )

        recommendations = recommendations_df.to_html(escape=False)

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return recommendations

    return render_template('for_you.html', recommendations=recommendations, 
                           price_ranges=price_ranges,
                           mem_size_range=mem_size_range,
                           gpu_clock_range=gpu_clock_range,
                           mem_clock_range=mem_clock_range,
                           unified_shader_range=unified_shader_range,
                           tmu_range=tmu_range,
                           rop_range=rop_range,
                           release_year_range=release_year_range,
                           g3dmark_range=g3dmark_range,
                           tdp_range=tdp_range,
                           gpu_value_range=gpu_value_range,
                           mem_type_range=mem_type_range,
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
        existing_preference = Preference.query.filter_by(
            user_id=user.id).order_by(Preference.id.desc()).first()

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
        # Convert string to dictionary
        last_preferences_dict = eval(last_preferences)

    except IOError as e:
        return jsonify({'success': False, 'message': f'Error reading from log file: {str(e)}'})

    return jsonify({'success': True, 'message': 'Preferences loaded successfully.', 'preferences': last_preferences_dict})

