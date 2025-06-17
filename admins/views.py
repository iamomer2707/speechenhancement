from django.shortcuts import render, HttpResponse, redirect
from django.contrib import messages
from users.models import UserRegistrationModel

# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def AdminHome(request):
    return render(request, 'admins/AdminHome.html')


def RegisterUsersView(request):
    # Fetch all users
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewregisterusers.html', {'data': data})


def ActivaUsers(request):
    if request.method == 'GET':
        # Get user ID from request
        user_id = request.GET.get('uid')
        # Update user status to 'activated'
        try:
            user = UserRegistrationModel.objects.get(id=user_id)
            if user.status == 'waiting':  # Only activate if user is in 'waiting' status
                user.status = 'activated'
                user.save()
                messages.success(request, f"User {user.name} has been activated.")
            else:
                messages.warning(request, f"User {user.name} is already activated.")
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, "User not found.")
        # Redirect to the view users page
        return redirect('view_register_users')  # Assuming you have a named URL for viewing users


def DeleteUsers(request):
    if request.method == 'GET':
        user_id = request.GET.get('uid')
        try:
            user = UserRegistrationModel.objects.get(id=user_id)
            user.delete()
            messages.success(request, f"User {user.name} has been deleted.")
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, "User not found.")
        # Redirect to the view users page after deletion
        return redirect('RegisterUsersView')  # Assuming you have a named URL for viewing users
