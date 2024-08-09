import json

from django.shortcuts import HttpResponse, render
from django.views.decorators.csrf import csrf_exempt

from app01.views import graph_rag

# Create your views here.
OPENAI_API_KEY = 'sk-YoZZU2QvKG6Kzqhv9c18648062D94c08965128Ce4196E2E8'


def qa(request):
    return render(request, 'qa.html')


@csrf_exempt
def chat(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_input = data['message']

        # QA提示模版
        answer_template = """Answer the question based only on the following context:
                    {context}
                    Question: {question}
                    Use natural language and be concise.
                    Answer:"""
        answer_prompt = graph_rag.ChatPromptTemplate.from_template(answer_template)
        print(answer_prompt)
        # QA链
        chain = (
                graph_rag.RunnableParallel({
                    "context": graph_rag._search_query | graph_rag.retriever,
                    "question": graph_rag.RunnablePassthrough(),
                })
                | answer_prompt
                | graph_rag.ChatOpenAI(temperature=0)
                | graph_rag.StrOutputParser()
        )

        # 运行QA链
        output = chain.invoke({"question": user_input.lower()})
        # output = '1111111'
        with open('app01/datasets/test.json', 'r', encoding='utf-8') as f:
            paper_entity = json.load(f)
        # ans = chain.invoke(
        #     {
        #         "question": "Please give me the relavant paper names ",
        #         "chat_history": [(user_input.lower(), output)],
        #     }
        # )
        # print(ans)
        # # 调用 OpenAI API 生成响应
        # headers = {
        #     "Authorization": f"Bearer {OPENAI_API_KEY}",
        #     "Content-Type": "application/json"
        # }

        # data = {'model': 'gpt-3.5-turbo-0125', 'messages': [{'role': 'user', 'content': user_input}],
        #         'stream': False}
        # response = requests.post('https://api.chsdw.top/v1/chat/completions', headers=headers, data=json.dumps(data),
        #                          stream=True)

        # res = response.json()['choices'][0]['message']['content']
        return HttpResponse(json.dumps({'response': output, 'entity': paper_entity}))
