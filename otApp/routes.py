import os
import secrets
from flask import render_template, url_for, flash, redirect, request, abort, session
from otApp import app
from otApp.forms import FeelingsForm, RegistrationForm, LoginForm
from otApp.models import User
from flask_login import login_user, current_user, logout_user, login_required
from otApp import app, db, bcrypt
from otApp.chat import chatbot
from otApp.emotion import check_emo
from otApp.scrape import anx, sad, angry
import json


@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():

    return render_template('home.html')


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
@login_required
def chat():
    # form = HomeForm()
    return render_template('chat.html')


@app.route("/feelings", methods=['GET', 'POST'])
def feelings():
    form = FeelingsForm()
    if form.validate_on_submit():
        inp = form.inp.data
        emo = check_emo(inp)

        # flash(emo, 'success')
        # return redirect(url_for('results', emo=emo))
    return render_template('feelings.html', form=form)


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    response, tag = chatbot(userText)
    if tag == "feelings":
        return response
    else:
        return response


@app.route("/results", methods=['GET', 'POST'])
def results():
    h_list = []
    form = FeelingsForm()
    if request.method == 'POST':
        inp = form.inp.data
        emo = check_emo(inp)
    else:
        emo = "sad"

    if emo == "worry":
        h_list = anx()
    elif emo == "hate":
        h_list = angry()
    elif emo == "sad":
        h_list = sad()

    return render_template('results.html', h_list=h_list, emo=emo)
