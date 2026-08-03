"""Admin-only user-management routes."""

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import or_
from werkzeug.security import generate_password_hash

from extensions import db
from models.database import User
from utils.access import admin_required

users_bp = Blueprint("users", __name__, url_prefix="/users")


def _validated_user_form(user=None):
    full_name = request.form.get("full_name", "").strip()
    username = request.form.get("username", "").strip()
    role = request.form.get("role", "Employee")
    password = request.form.get("password", "")

    if not full_name or not username:
        return None, "Full name and username are required."
    if role not in {"Admin", "Employee", "Staff"}:
        return None, "Invalid user role."
    if user is None and len(password) < 6:
        return None, "Password must contain at least 6 characters."
    if user is not None and password and len(password) < 6:
        return None, "Password must contain at least 6 characters."

    duplicate_query = User.query.filter_by(username=username)
    if user is not None:
        duplicate_query = duplicate_query.filter(User.id != user.id)
    if duplicate_query.first() is not None:
        return None, "Username is already in use."

    return {
        "full_name": full_name,
        "username": username,
        "role": role,
        "password": password,
    }, None


@users_bp.get("/")
@admin_required
def index():
    search = request.args.get("search", "").strip()
    query = User.query
    if search:
        pattern = f"%{search}%"
        query = query.filter(or_(User.full_name.ilike(pattern), User.username.ilike(pattern)))
    users = query.order_by(User.created_at.desc()).all()
    return render_template("users/index.html", users=users, search=search)


@users_bp.route("/create", methods=["GET", "POST"])
@admin_required
def create():
    if request.method == "POST":
        values, error = _validated_user_form()
        if error:
            flash(error, "danger")
        else:
            db.session.add(
                User(
                    full_name=values["full_name"],
                    username=values["username"],
                    role=values["role"],
                    password_hash=generate_password_hash(values["password"]),
                )
            )
            db.session.commit()
            flash("User created successfully.", "success")
            return redirect(url_for("users.index"))
    return render_template("users/form.html", user=None)


@users_bp.route("/<int:user_id>/edit", methods=["GET", "POST"])
@admin_required
def edit(user_id):
    user = db.get_or_404(User, user_id)
    if request.method == "POST":
        values, error = _validated_user_form(user)
        if user.id == current_user.id and values and values["role"] != "Admin":
            error = "You cannot remove your own Admin role."
        if error:
            flash(error, "danger")
        else:
            user.full_name = values["full_name"]
            user.username = values["username"]
            user.role = values["role"]
            if values["password"]:
                user.password_hash = generate_password_hash(values["password"])
            db.session.commit()
            flash("User updated successfully.", "success")
            return redirect(url_for("users.index"))
    return render_template("users/form.html", user=user)


@users_bp.post("/<int:user_id>/delete")
@admin_required
def delete(user_id):
    user = db.get_or_404(User, user_id)
    if user.id == current_user.id:
        flash("You cannot delete your own account.", "danger")
    elif user.role == "Admin" and User.query.filter_by(role="Admin").count() <= 1:
        flash("At least one Admin account must remain.", "danger")
    else:
        db.session.delete(user)
        db.session.commit()
        flash("User deleted successfully.", "success")
    return redirect(url_for("users.index"))
