import asyncio
import os
import logging
import sys

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFaceHub
from transformers import pipeline

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")
MEMORY_WINDOW = os.getenv("MEMORY_WINDOW")


login(HUGGINGFACE_API_TOKEN)

llm = HuggingFaceHub(
    repo_id=MODEL_NAME,
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    model_kwargs={"temperature":0.1},
)


#  # Initialize the Hugging Face model and tokenizer for the pipeline
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# hf_pipeline = pipeline("text-generation", model=MODEL_NAME, tokenizer=tokenizer)


# #  Wrap the Hugging Face pipeline with LangChain's LLM
# llm = HuggingFacePipeline(pipeline=hf_pipeline)

memory = ConversationBufferWindowMemory(k=MEMORY_WINDOW)  

conversation = ConversationChain(llm=llm, memory=memory)

dp = Dispatcher()

@dp.message(Command(commands=['clear']))
async def clear(message: Message):
    """
    Clear the previous conversation and context.
    """
    memory.clear()  # Clear the conversation memory in LangChain
    await message.answer("I've cleared the past conversation and context.")


@dp.message(Command(commands=['start']))
async def command_start_handler(message: Message) -> None:
    """
    Handle the `/start` command.
    """
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!")


@dp.message(Command(commands=['help']))
async def helper(message: Message):
    """
    Display the help menu.
    """
    help_command = """
    Hi There! I'm a Telegram bot with conversation memory. Here are some commands:
    /start - Start the conversation
    /clear - Clear conversation context
    /help - Get this help menu
    """
    await message.reply(help_command)


@dp.message()
async def chat_handler(message: Message):
    """
    Process the user's input and generate a response with conversation context.
    """
    print(f">>> USER: \n\t{message.text}")

    # Generate response with conversation context
    response = conversation.invoke(input=message.text)
    print(f">>> MODEL: \n\t{response}")
    await message.answer(response["response"])


async def main() -> None:
    # Initialize Bot instance with default bot properties for HTML formatting
    bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # Start polling
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())