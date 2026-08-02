"""Access-control decorators for administrative routes."""

from functools import wraps

from flask import abort
from flask_login import current_user, login_required


def admin_required(view):
    """Allow access only to authenticated users with the Admin role."""
    @wraps(view)
    @login_required
    def wrapped_view(*args, **kwargs):
        if current_user.role != "Admin":
            abort(403)
        return view(*args, **kwargs)

    return wrapped_view
