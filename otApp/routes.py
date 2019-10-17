import os
import secrets
from flask import render_template, url_for, flash, redirect, request, abort
from otApp import app
from otApp.forms import HomeForm, RegistrationForm, LoginForm
from otApp.models import User
from flask_login import login_user, current_user, logout_user, login_required
from otApp import app, db, bcrypt
from otApp.chat import chatbot


@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():
    form = HomeForm()
    return render_template('home.html', form=form)


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(
            form.password.data).decode('utf-8')
        user = User(username=form.username.data,
                    email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        login_user(user)
        next_page = request.args.get('next')

        return redirect(next_page) if next_page else redirect(url_for('chat'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')

            return redirect(next_page) if next_page else redirect(url_for('chat'))
        else:
            flash('Log in Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route("/chat", methods=['GET', 'POST'])
def chat():
    # form = HomeForm()
    return render_template('chat.html')


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    response = chatbot(userText)
    return response
