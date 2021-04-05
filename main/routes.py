from flask import render_template, url_for, flash, redirect
from main import app

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/register", methods=['GET', 'POST'])
def register():
    return render_template('register.html', title='Register')


@app.route("/login", methods=['GET', 'POST'])
def login():
    return render_template('login.html', title='Login')