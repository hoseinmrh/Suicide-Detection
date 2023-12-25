from django.shortcuts import render

# Create your views here.

def home(request):
    ''' home view for making a test'''
    return render(request, "home.html")
