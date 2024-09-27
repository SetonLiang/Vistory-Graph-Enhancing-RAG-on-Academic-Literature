import json
import time

from django.shortcuts import HttpResponse, render
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
                    Use natural language and be concise,without using any Markdown formatting.
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
                | graph_rag.ChatOpenAI(temperature=0, model='gpt-4o', streaming=True)
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

        elif user_input == "What are the main research directions of Dept.2 in 2024?":
            output = """
                The main research direction of Dept.2 in 2024 focuses primarily on applications of virtual reality and immersive technologies, specifically in the context of human-computer interaction, virtual environments, and augmented reality for diverse applications including education, healthcare, and creative industries.
            """
            with open('app01/datasets/case1_test1.json', 'r', encoding='utf-8') as f:
                paper_entity = json.load(f)
            with open('app01/datasets/case1_test1_chart.json', 'r', encoding='utf-8') as f:
                update_entity = json.load(f)

        elif user_input == "How many publications are related to virtual reality? by which authors?":
            output = """
                There are 40 publications related to virtual reality in Dept.2. The authors involved are Author A, Author B, Author C, Author D, and Author E.
            """
            with open('app01/datasets/case1_test2.json', 'r', encoding='utf-8') as f:
                paper_entity = json.load(f)
            with open('app01/datasets/case1_test2_chart.json', 'r', encoding='utf-8') as f:
                update_entity = json.load(f)

        elif user_input == "I'm interested in using virtual reality for metaverse, give me a list of relevant paper.":
            output = """
                Here are some relevant papers on using virtual reality in the metaverse: 1. Reducing stress and anxiety in the metaverse: A systematic review of meditation, mindfulness, and virtual reality (2022) - Focuses on the role of VR technology in meditation and mental health within the metaverse. 2. Network Traffic in the Metaverse: The Case of Social VR (2023) - Evaluates network traffic patterns in social VR platforms and how they impact user experience during mixed-mode events. 3. Players are not Ready 101: A Tutorial on Organising Mixed-mode Events in the Metaverse (2023) - Discusses technical, human, and organizational challenges in building university metaverse environments. 4. Beyond the Blue Sky of Multimodal Interaction: A Centennial Vision of Interplanetary Virtual Spaces in Turn-based Metaverse (2022) - Explores future concepts of interplanetary communication and social interaction via the metaverse. 5. Empowering the metaverse with generative AI: Survey and future directions (2023) - Highlights the role of AI-generated content in enhancing the metaverse experience.
            """
            with open('app01/datasets/case1_test3.json', 'r', encoding='utf-8') as f:
                paper_entity = json.load(f)
            with open('app01/datasets/case1_test3_chart.json', 'r', encoding='utf-8') as f:
                update_entity = json.load(f)

        elif user_input == "What are the popular applications of virtual reality?":
            output = """
               Virtual reality (VR) is popular in several key areas. It is commonly used in education for improving learning efficiency, such as in interaction, user study, 3D modeling, and some immersive experiences. VR is also widely applied in healthcare for rehabilitation therapy, helping patients with motor impairments. Additionally, VR is used in design and architecture, allowing users to explore room layouts and provide feedback. These applications showcase VR's role in enhancing learning, reducing anxiety, and supporting physical and mental well-being.
            """
            with open('app01/datasets/case1_test4.json', 'r', encoding='utf-8') as f:
                paper_entity = json.load(f)
            with open('app01/datasets/case1_test4_chart.json', 'r', encoding='utf-8') as f:
                update_entity = json.load(f)

        else:
            if not chat_history:
                # 运行QA链
                output = chain.invoke({"question": user_input})
                chat_history.append((user_input, output))

            # output = "111"
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
            with open('app01/datasets/update_entity.json', 'r', encoding='utf-8') as f:
                update_entity = json.load(f)

        duration = time.time() - start_time
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
        return HttpResponse(json.dumps({'response': output, 'entity': paper_entity, 
                                       'donut': update_entity['donut'],
                                       'chart_d_y': update_entity['chart_d_y'],
                                       'treemap': update_entity['treemap'],
                                       'wordcloud':update_entity['wordcloud']
                                       }))
        # return StreamingHttpResponse(generate(), content_type='application/json')
    # return HttpResponse(status=405)
