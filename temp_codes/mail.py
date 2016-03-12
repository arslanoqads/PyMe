import os
from flask import Flask
from flask.ext.mail import Mail, Message


app = Flask(__name__)

app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'mssk.me@gmail.com'
app.config['MAIL_PASSWORD'] = 'arazalan'
app.config['DEFAULT_MAIL_SUBJECT'] = '[Politburo Voting Results]'
app.config['DEFAULT_MAIL_SENDER'] = 'Admin <admin@example.com>'
app.config['SECRET_KEY'] = 'random_string'
app.config['DEFAULT_ADMIN'] = 'Admin <hypnopompicindex@gmail.com>'

mail = Mail(app)




def send_email(to, subject, template, **kwargs):
    msg = Message(app.config['DEFAULT_MAIL_SUBJECT'] + ' ' + subject,
        sender=app.config['DEFAULT_MAIL_SENDER'], recipients=[to])
    msg.attach(filename='abc.png', content_type="image/png")
    mail.send(msg)

