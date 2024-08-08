import json

# from app01.views import graph_rag
from django.shortcuts import HttpResponse, render
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
OPENAI_API_KEY = 'sk-YoZZU2QvKG6Kzqhv9c18648062D94c08965128Ce4196E2E8'


def qa(request):
    return render(request, 'qa.html')


@csrf_exempt
def chat(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_input = data['message']
        print(user_input)

        # output = graph_rag.return_response(user_input)

        output = '测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试'
        with open('app01/datasets/entity_qa_test.json', 'r', encoding='utf-8') as file:
            entity = json.load(file)

        print(entity)

        # 调用 OpenAI API 生成响应
        # headers = {
        #     "Authorization": f"Bearer {OPENAI_API_KEY}",
        #     "Content-Type": "application/json"
        # }
        #
        # data = {'model': 'gpt-3.5-turbo-0125', 'messages': [{'role': 'user', 'content': user_input}],
        #         'stream': False}
        # response = requests.post('https://api.chsdw.top/v1/chat/completions', headers=headers, data=json.dumps(data),
        #                          stream=True)
        #
        # res = response.json()['choices'][0]['message']['content']
        return HttpResponse(json.dumps({'response': output, 'entity': entity}))
