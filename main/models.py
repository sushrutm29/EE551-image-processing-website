from main import db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.objects(id=user_id).first()

class User(db.Document, UserMixin):
    username = db.StringField()
    email = db.StringField()
    password = db.StringField()
    household = db.StringField()

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

class Household(db.Document):
    chores = db.ListField()
    users = db.ListField()

    def __repr__(self):
        return f"Household('{self.chores}', '{self.users}')"