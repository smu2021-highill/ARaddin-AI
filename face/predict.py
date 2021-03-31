from face.model import *
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core import serializers
import json

model_dict = dict()

@csrf_exempt
def encode_img(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        path = data['path']
        try:
            save_feature(path)
        except Exception as e:
            print(e)
            return HttpResponse(status=400)

        return HttpResponse(status=201)

@csrf_exempt
def prepare_model(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        path = data['path']
        room_num = data['room_num']

        model_dict[room_num]=train_model(path,0.0002,100).eval()

    return HttpResponse(status=201)

@csrf_exempt
def match_user(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        path = data['path']
        room_num = data['room_num']

        model = model_dict[room_num]
        name, prob = predict(model,path)

        if prob > 0.9:
            response = {"name": name}
            return JsonResponse(response)
        else:
            response = {"name":""}
            return JsonResponse(response)

@csrf_exempt
def delete_model(request,room_num):
    if request.method =='DELETE':
        model_dict.__delattr__(room_num)

    return HttpResponse(status=200)
