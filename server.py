import asyncio
from aiohttp import web
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# 1. 初始化基础组件
ollama = ChatOllama(
    base_url="http://127.0.0.1:11434",
    model="deepseek-r1:7b",
    temperature=0.8,
    num_predict=10000,
)

localmodel = ollama
str_parser = StrOutputParser()

# 2. 从外部文件读取操作说明
def load_operation_instruction(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except:
        return "默认操作说明"

operation_instruction = load_operation_instruction("SFJ.txt")

# 3. 构建提示词模板
system = """
你是一个网站导航助手，要根据用户的问题和网站导航说明书为用户进行操作导航。
    网站操作说明如下：{operationinstruction}
"""
prompt = ChatPromptTemplate.from_messages(["{system} {chat_history} {question}"])

# 4. 构建链
chain = (
        RunnablePassthrough.assign(
            system=lambda x: system.format(operationinstruction=operation_instruction)
        )
        | prompt
        | localmodel
        | str_parser
)

# 5. 添加CORS中间件
@web.middleware
async def cors_middleware(request, handler):
    response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# 6. 处理聊天请求
async def handle_chat(request):
    try:
        data = await request.json()
        question = data.get('message', '')
        chat_history = data.get('history', '')

        # 使用 chain 处理问题
        response = await chain.ainvoke({
            "question": question,
            "chat_history": chat_history
        })

        return web.Response(
            text=json.dumps({"response": response}),
            content_type='application/json'
        )
    except Exception as e:
        return web.Response(
            text=json.dumps({"error": str(e)}),
            status=500,
            content_type='application/json'
        )

# 7. 处理OPTIONS请求
async def handle_options(request):
    return web.Response(status=200)

# 8. 主程序
async def init_app():
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_post('/chat', handle_chat)
    app.router.add_options('/chat', handle_options)
    return app

if __name__ == "__main__":
    app = asyncio.get_event_loop().run_until_complete(init_app())
    web.run_app(app, host='127.0.0.1', port=8080)