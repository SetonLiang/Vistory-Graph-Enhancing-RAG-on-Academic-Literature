import json,time

from django.shortcuts import HttpResponse, render
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

from app01.views import graph_rag

# Create your views here.
OPENAI_API_KEY = 'sk-YoZZU2QvKG6Kzqhv9c18648062D94c08965128Ce4196E2E8'


def qa(request):
    return render(request, 'qa.html')

chat_history = []
@csrf_exempt
def chat(request):
    global chat_history

    if request.method == 'POST':
        data = json.loads(request.body)
        user_input = data['message']

        # Use natural language and provide a concise, detailed answer.
        # Use natural language and be concise.
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
                    | graph_rag.ChatOpenAI(temperature=0,model='gpt-4o',streaming=True)
                    | graph_rag.StrOutputParser()
            )
        start_time = time.time()
        if user_input == "What are the latest research findings in the area of virtual reality?":
            output = """
            The latest research findings in the area of virtual reality (VR) include the development of immersive simulators for enhancing chemistry education, use of VR interview simulators for improving interview skills and reducing anxiety, systematic reviews of the benefits of meditation and mindfulness in VR, explorations into asymmetric collaborative visualization to aid problem-solving, and studies on how VR can support remote communication and reduce the generational gap between grandparents and grandchildren.
            """
            # output = chain.invoke({"question": user_input})
            with open('app01/datasets/user_study_test1.json', 'r', encoding='utf-8') as f:
                paper_entity = json.load(f)
        else: 
            if not chat_history:
                # 运行QA链
                output = chain.invoke({"question": user_input})
                chat_history.append((user_input, output))

            # output = '1111111'
        
            else:
                previous_question, previous_answer = chat_history[-1]
                output = chain.invoke(
                    {
                        "question": user_input,
                        "chat_history": [(previous_question, previous_answer)],
                    }
                )
            with open('app01/datasets/test.json', 'r', encoding='utf-8') as f:
                paper_entity = json.load(f)
        
        duration = time.time()-start_time
        print(duration)
        
        
        def generate():
            for chunk in chain.stream({"question": user_input.lower()}):
                if chunk is not None:
                    yield json.dumps({"response": chunk}) + '\n'
                        # 加载实体数据
                        #     with open('app01/datasets/test.json', 'r', encoding='utf-8') as f:
                        #         paper_entity = json.load(f)
                        #         yield f'data: {json.dumps({"entity": paper_entity})}\n\n'
            # full_message = ''
            # for chunk in chain.stream({"question": user_input.lower()}):
            #     full_message += chunk
            #     # 处理到达一定长度后发送一个完整消息块
            #     if len(full_message) > 400:  # 调整阈值以满足实际需求
            #         yield json.dumps({"response": full_message}) + '\n'
            #         full_message = ''
        

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
        # return StreamingHttpResponse(generate(), content_type='application/json')
    # return HttpResponse(status=405) 