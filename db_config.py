import pymongo
from urllib.parse import quote_plus

mongoclient = pymongo.MongoClient("mongodb://localhost:27017/")

db_name = "ai-colining"
character_table_name = "chatbots"
chat_history_table_name = "chathistories"
user_table_name = "users"

mongodb = mongoclient[db_name]
character_table = mongodb[character_table_name]
chat_history = mongodb[chat_history_table_name]
user_table = mongodb[user_table_name]
tts_voices_table = mongodb['tts_voices']
rooms_table = mongodb['rooms']
generated_images_table = mongodb['generated_images']