from django.shortcuts import HttpResponse, render


# Create your views here.

def chart(request):
    return render(request, 'chart.html')


def chart_query(request):
    return HttpResponse
