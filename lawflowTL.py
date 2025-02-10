import asyncio

from dashscope import api_key
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_ollama import ChatOllama
from langchain_core import

# 1. 初始化基础组件
ollama = ChatOllama(
    base_url="http://127.0.0.1:11434",
    model="qwen2.5:7b",
    temperature=0.8,
    num_predict=10000,
)

localmodel = ollama
str_parser = StrOutputParser()
json_parser = JsonOutputParser()

# 2. 从外部文件读取操作说明
def load_operation_instruction(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

operation_instruction = load_operation_instruction("SFJ.txt")

# 3. 构建提示词模板
system = """
你是一个网站导航助手，要根据用户的问题和网站导航说明书为用户进行操作导航。
    网站操作说明如下：{operationinstruction}
"""
propmt = ChatPromptTemplate.from_messages(["{system} {chat_history} {question}"])

# 4. 构建链
chain = (
    RunnablePassthrough.assign(
        system=lambda x: system.format(operationinstruction=operation_instruction)
    )
    | propmt
    | localmodel
    | str_parser
)

# 5. 构建工作流
async def answer_question(question, chat_history):
    current_response = ""

    # 流式输出
    async for chunk in chain.astream({"question": question, "chat_history": chat_history}):
        print(chunk, end="", flush=True)
        current_response += chunk
    return current_response

# 6. 主程序
# 对话运行模块
async def main():
    chat_history = ""

    while True:
        question = input("\n请输入您的问题：")
        current_response = await answer_question(question, chat_history)  # 接收返回值
        chat_history = f"{chat_history} 用户: {question}\nAI: {current_response}\n"

if __name__ == "__main__":
    asyncio.run(main())
