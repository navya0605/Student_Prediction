from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.io as pio
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.db'
db = SQLAlchemy(app)

genai.configure(api_key='AIzaSyDd24FzkOPyc5Bkk0w1-ZPlapfHgq79Bk8')  # Your API key
genai_model = genai.GenerativeModel('gemini-2.0-flash')

# Load the model, scaler, encoders, and preprocessed data
model = joblib.load('models/oulad_model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoders = joblib.load('models/encoders.pkl')
preprocessed_data = joblib.load('models/preprocessed_data.pkl')

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    notifications = db.Column(db.Text, default='No new notifications')

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(100), db.ForeignKey('student.student_id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

with app.app_context():
    hashed_password = generate_password_hash("admin123", method="pbkdf2:sha256")
    if not Admin.query.filter_by(username="admin").first():
        admin_user = Admin(username="admin", password=hashed_password)
        db.session.add(admin_user)
        db.session.commit()

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        student_id = request.form['student_id'].strip()
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password', '')

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        existing_student = Student.query.filter_by(student_id=student_id).first()
        student_in_data = preprocessed_data[preprocessed_data['id_student'] == int(student_id)]
        if existing_student:
            flash('Student ID already exists!', 'danger')
            return redirect(url_for('signup'))
        elif student_in_data.empty:
            flash('Invalid user! Student ID not found in records.', 'danger')
            return redirect(url_for('signup'))
        else:
            new_student = Student(student_id=student_id, email=email, phone=phone, password=hashed_password)
            db.session.add(new_student)
            db.session.commit()
            flash('Signup successful! Please log in.', 'success')
            return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id_or_username = request.form.get('user_id', '').strip()
        password = request.form.get('password', '').strip()
        role = request.form.get('role', '').strip()

        if not user_id_or_username or not password or not role:
            flash('All fields are required!', 'danger')
            return redirect(url_for('login'))

        if role == 'student':
            student = Student.query.filter_by(student_id=user_id_or_username).first()
            if student and check_password_hash(student.password, password):
                session.clear()
                session['student_id'] = student.student_id
                flash('Student login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid student credentials!', 'danger')
                return redirect(url_for('login'))
        elif role == 'admin':
            admin = Admin.query.filter_by(username=user_id_or_username).first()
            if admin and check_password_hash(admin.password, password):
                session.clear()
                session['admin'] = admin.username
                flash('Admin login successful!', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid admin credentials!', 'danger')
                return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin' not in session:
        flash('Please log in as an admin first.', 'danger')
        return redirect(url_for('login'))
    # 🎯 Generate Overall Performance Pie Chart with Labels & Bright Colors
    label_mapping = {0: "Withdrawn", 1: "Fail", 2: "Pass", 3: "Distinction"}  # Proper labels
    preprocessed_data['final_result_label'] = preprocessed_data['final_result'].map(label_mapping)

    all_students_results = preprocessed_data['final_result_label'].value_counts()

    fig_pie = px.pie(
        values=all_students_results, 
        names=all_students_results.index, 
        title="Overall Student Performance",
        color_discrete_sequence=["#FF5733", "#33FF57", "#3357FF", "#FFD700"],  # Bright Colors
        labels={"index": "Performance", "value": "Count"}
    )

    fig_pie.update_traces(textinfo="label+percent",insidetextorientation="horizontal")  # Show both labels and percentages

    # Save the chart
    pio.write_html(fig_pie, file="static/overall_performance.html", auto_open=False)

    return render_template('admin_dashboard.html',overall_chart='static/overall_performance.html')

@app.route('/dashboard')
def dashboard():
    if 'student_id' not in session:
        flash("No student ID registered", 'danger')
        return redirect(url_for('login'))
    
    student_id = session['student_id']
    student = Student.query.filter_by(student_id=student_id).first()
    notifications = Notification.query.filter_by(student_id=student_id).order_by(Notification.created_at.desc()).all()
    has_unread_notifications = any(not notif.is_read for notif in notifications)
    # Fetch preprocessed data for the student
    student_data = preprocessed_data[preprocessed_data['id_student'] == int(student_id)]
    if student_data.empty:
        flash("No preprocessed data found for this student.", 'danger')
        return redirect(url_for('dashboard'))
    
    student_data = student_data.iloc[0]
    
    # Prepare features for prediction (already preprocessed)
    features = ['avg_score', 'total_clicks', 'studied_credits', 'days_to_start', 'gender', 'disability', 'highest_education', 'age_band']
    new_data = pd.DataFrame([student_data[features].values], columns=features)

    # Predict
    prediction = model.predict(new_data)
    prediction_label = encoders['final_result'].inverse_transform(prediction.reshape(-1, 1))[0].item()
    recommendation = {
        'Distinction': "This student is predicted to achieve a Distinction! Encourage them to continue their outstanding work.",
        'Pass': "This student is predicted to Pass. Continue monitoring their progress and provide encouragement.",
        'Fail': "This student is at risk of failing. Consider additional support and encourage more engagement.",
        'Withdrawn': "This student is at risk of withdrawing. Immediate intervention is recommended."
    }.get(prediction_label, "No recommendation available.")

    # Generate chart (optional)
    fig = px.bar(x=[prediction_label], title='Predicted Final Result')
    pio.write_html(fig, file='static/student_result.html', auto_open=False)

    # 🎯 Student vs Other Students Chart (Comparison)
    comparison_data = preprocessed_data.copy()
    comparison_data['Student Type'] = comparison_data['id_student'].apply(lambda x: 'Selected Student' if x == int(student_id) else 'Other Students')

    # Generate the comparison bar chart
    fig_comparison = px.box(
        comparison_data, 
        x='Student Type', 
        y='avg_score', 
        title="Student's Average Score vs Other Students"
    )
    pio.write_html(fig_comparison, file='static/student_vs_others.html', auto_open=False)
 
    return render_template('dashboard.html', 
                           student_id=student_id, 
                           notifications=notifications,
                           has_unread_notifications=has_unread_notifications, 
                           student_chart='static/student_result.html',
                           comparison_chart='static/student_vs_others.html',
                           prediction=prediction_label,
                           recommendation=recommendation)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'student_id' not in session:
        return redirect(url_for('login'))
    student_id = session['student_id']
    student = Student.query.filter_by(student_id=student_id).first()
    
    if request.method == 'POST':
        student.email = request.form['email']
        student.phone = request.form['phone']
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', student=student)

@app.route('/logout')
def logout():
    session.pop('student_id', None)
    session.pop('admin', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/admin/notifications', methods=['GET', 'POST'])
def update_notifications():
    if 'admin' not in session:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))
    
    students = Student.query.all()
    predictions = {}
    notifications = Notification.query.order_by(Notification.created_at.desc()).all()
    
    # Precompute predictions for all students
    for student in students:
        student_id = student.student_id
        student_data = preprocessed_data[preprocessed_data['id_student'] == int(student_id)]
        if not student_data.empty:
            student_data = student_data.iloc[0]
            features = ['avg_score', 'total_clicks', 'studied_credits', 'days_to_start', 'gender', 'disability', 'highest_education', 'age_band']
            new_data = pd.DataFrame([student_data[features].values], columns=features)
            try:
                prediction = model.predict(new_data)
                prediction_label = encoders['final_result'].inverse_transform(prediction.reshape(-1, 1))[0].item()
                predictions[student_id] = prediction_label
            except Exception as e:
                predictions[student_id] = f"Error: {str(e)}"
        else:
            predictions[student_id] = "No data"

    if request.method == 'POST':
        action = request.form.get('action')
        student_id = request.form.get('student_id')
        student = Student.query.filter_by(student_id=student_id).first()

        if not student:
            flash('Student not found.', 'danger')
            return redirect(url_for('update_notifications'))

        if action == 'generate':
            prediction = predictions.get(student_id, "No data")
            if prediction in ["Error", "No data"]:
                flash('Cannot generate suggestion due to missing or erroneous data.', 'danger')
                return redirect(url_for('update_notifications'))
            prompt = f"""
            For a student with a predicted performance of '{prediction}', provide specific suggestions for improvement:
            - If 'Withdrawn' or 'Fail': How to achieve a 'Pass'.
            - If 'Pass': How to achieve a 'Distinction'.
            - If 'Distinction': How to maintain this level.
            -Include actionable steps based on factors like average score, engagement (clicks), study credits, and time management.
            give in 5 points no star marks and no long explanations sharp to the point to enhance online learning based on their performance. 
            """
            try:
                response = genai_model.generate_content(prompt)
                suggestion = response.text
            except Exception as e:
                flash(f"Failed to generate suggestion: {str(e)}", 'danger')
                suggestion = ""
            return render_template('admin_notifications.html', students=students, predictions=predictions, 
                                   notifications=notifications, student_id=student_id, suggestion=suggestion)

        elif action == 'send':
            notification_message = request.form.get('notification')
            if notification_message:
                new_notification = Notification(student_id=student_id, message=notification_message)
                db.session.add(new_notification)
                db.session.commit()
                flash('Notification sent successfully!', 'success')
            else:
                flash('Notification message cannot be empty.', 'danger')
            return redirect(url_for('update_notifications'))

    return render_template('admin_notifications.html', students=students, predictions=predictions, notifications=notifications)

@app.route('/mark_notifications_read', methods=['POST'])
def mark_notifications_read():
    if 'student_id' not in session:
        return jsonify({"error": "Unauthorized"}), 403
    
    student_id = session['student_id']
    notifications = Notification.query.filter_by(student_id=student_id, is_read=False).all()
    for notif in notifications:
        notif.is_read = True
    db.session.commit()
    
    return jsonify({"success": True})

@app.route('/notifications')
def notifications():
    if 'student_id' not in session:
        return redirect(url_for('login'))

    student_id = session['student_id']

    # Fetch all notifications
    student_notifications = Notification.query.filter_by(student_id=student_id).order_by(Notification.created_at.desc()).all()

    # Mark all notifications as read
    for notif in student_notifications:
        notif.is_read = True
    db.session.commit()

    has_unread_notifications = Notification.query.filter_by(student_id=student_id, is_read=False).count() > 0

    return render_template('notifications.html', notifications=student_notifications, has_unread_notifications=has_unread_notifications)



@app.route('/check_new_notifications')
def check_new_notifications():
    student_id = session.get('student_id')
    has_unread = Notification.query.filter_by(student_id=student_id, is_read=False).count() > 0

    return jsonify({"has_unread": has_unread})


@app.route('/delete_notification/<int:notification_id>', methods=['POST'])
def delete_notification(notification_id):
    if 'admin' not in session:
        return jsonify({"error": "Unauthorized"}), 403

    notification = Notification.query.get(notification_id)
    if notification:
        db.session.delete(notification)
        db.session.commit()
        flash('Notification deleted successfully!', 'success')
    else:
        flash('Notification not found.', 'danger')

    return redirect(url_for('update_notifications'))




@app.route('/admin/student_performance/<student_id>')
def student_performance(student_id):
    if 'admin' not in session:
        return jsonify({"error": "Unauthorized"}), 403

    try:
        # Fetch preprocessed data for the student
        print(f"Fetching data for student_id: {student_id}")
        student_data = preprocessed_data[preprocessed_data['id_student'] == int(student_id)]
        if student_data.empty:
            print(f"No data found for student_id: {student_id}")
            return jsonify({"error": "No data found for this student"}), 404

        print(f"Student data: {student_data}")
        student_data = student_data.iloc[0]
        
        # Prepare features for prediction
        features = ['avg_score', 'total_clicks', 'studied_credits', 'days_to_start', 'gender', 'disability', 'highest_education', 'age_band']
        print(f"Preparing features: {features}")
        new_data = pd.DataFrame([student_data[features].values], columns=features)
        print(f"Input data for prediction: {new_data}")

        # Predict
        prediction = model.predict(new_data)
        print(f"Raw prediction: {prediction}")
        prediction_label = encoders['final_result'].inverse_transform(prediction.reshape(-1, 1))[0].item()
        print(f"Prediction label: {prediction_label}")

        # Generate performance chart
        fig = px.bar(x=[prediction_label], title=f'Predicted Final Result for Student {student_id}')
        performance_chart_path = 'static/admin_student_result.html'
        pio.write_html(fig, file=performance_chart_path, auto_open=False)
        print(f"Performance chart saved to: {performance_chart_path}")

        # Generate comparison chart (student vs. other students)
        comparison_data = preprocessed_data.copy()
        comparison_data['Student Type'] = comparison_data['id_student'].apply(lambda x: 'Selected Student' if x == int(student_id) else 'Other Students')
        fig_comparison = px.box(
            comparison_data, 
            x='Student Type', 
            y='avg_score', 
            title=f"{student_id}'s Average Score vs Other Students"
        )
        comparison_chart_path = 'static/admin_student_vs_others.html'
        pio.write_html(fig_comparison, file=comparison_chart_path, auto_open=False)
        print(f"Comparison chart saved to: {comparison_chart_path}")

        # Return JSON response
        return jsonify({
            "prediction": prediction_label,
            "student_chart": url_for('static', filename='admin_student_result.html'),
            "comparison_chart": url_for('static', filename='admin_student_vs_others.html')
        })

    except Exception as e:
        print(f"Error in student_performance route: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'student_id' not in session:
        flash("Please log in first.", 'danger')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Get form data
        avg_score = float(request.form['avg_score'])
        total_clicks = float(request.form['total_clicks'])
        studied_credits = float(request.form['studied_credits'])
        days_to_start = float(request.form['days_to_start'])
        gender = request.form['gender']
        disability = request.form['disability']
        highest_education = request.form['highest_education']
        age_band = request.form['age_band']
      
        total_clicks = total_clicks * 0.0000005  # Reduce importance by scaling down 
        
        # Create DataFrame for new data
        new_data = pd.DataFrame({
            'avg_score': [avg_score],
            'total_clicks': [total_clicks],
            'studied_credits': [studied_credits],
            'days_to_start': [days_to_start],
            'gender': [gender],
            'disability': [disability],
            'highest_education': [highest_education],
            'age_band': [age_band]
        })

        # Preprocess new data
        new_data['gender'] = encoders['gender'].transform(new_data[['gender']])
        new_data['disability'] = encoders['disability'].transform(new_data[['disability']])
        new_data['highest_education'] = encoders['highest_education'].transform(new_data[['highest_education']])
        new_data['age_band'] = encoders['age_band'].transform(new_data[['age_band']])
        new_data[['avg_score', 'total_clicks', 'studied_credits', 'days_to_start']] = scaler.transform(
            new_data[['avg_score', 'total_clicks', 'studied_credits', 'days_to_start']]
        )

        # Predict
        prediction = model.predict(new_data)
        prediction_label = encoders['final_result'].inverse_transform(prediction.reshape(-1, 1))[0].item()
        recommendation = {
            'Distinction': "With these inputs, you’re predicted to achieve a Distinction! Keep up the excellent work.",
            'Pass': "With these inputs, you’re predicted to Pass. Maintain your effort to ensure success.",
            'Fail': "With these inputs, you’re at risk of failing. Consider increasing your engagement and seeking support.",
            'Withdrawn': "With these inputs, you’re at risk of withdrawing. Immediate action is recommended."
        }.get(prediction_label, "No recommendation available.")

        return render_template('result.html', prediction=prediction_label, recommendation=recommendation)
    
    return render_template('index.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)