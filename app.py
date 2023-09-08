import io
import os
import re
import json
import uuid
import time
import boto3
import redis
import shutil
import string
import random
import asyncio
import uvicorn
import datetime
import requests
from PIL import Image
from pathlib import Path
from bson import json_util
from subprocess import Popen
from threading import Thread
from bson.objectid import ObjectId
from .connection import ConnectionManager
from botocore.exceptions import ClientError
from fastapi.staticfiles import StaticFiles
from extensions.api.util import build_parameters
from fastapi.middleware.cors import CORSMiddleware
from modules.chat import generate_chat_reply, save_character
from fastapi import FastAPI, Response, WebSocket, Request, HTTPException, WebSocketDisconnect
from db_config import character_table, chat_history, user_table, tts_voices_table, rooms_table, generated_images_table

redis_client = redis.Redis(host='localhost', port=6379, db=0)
socket_manager = ConnectionManager()

PLAYHT_ACCESS_KEY = "1kQQPnypwRRaJskfyRj5QxpdztP2"
PLAYHT_SECRET_KEY = "bb8fe6b739264432a61af3346c1603f5"

PLAYHT_AUDIO_STATUS_URL = "https://play.ht/api/v1/articleStatus?transcriptionId="
PLAYHT_CONVERT_TTS = "https://play.ht/api/v1/convert"

sd_model = "majicmixv6"

OUTPUT_BASE_DIR = "/home/ubuntu/"
RVC_FOLDER = "Retrieval-based-Voice-Conversion-WebUI/"

aws_access_key = "AKIA3XH6443UJS5O3LNJ"
aws_secret_key = "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"
aws_region = "ap-northeast-1"
aws_bucket_name = "onlyfantasy"

app = FastAPI()
app.mount('/media/characters', StaticFiles(directory='characters'), name='characters')
app.mount('/media/voices', StaticFiles(directory='voices'), name='voices')
app.mount('/media/images', StaticFiles(directory='generated_images'), name='generated_images')
app.mount('/media/image-training', StaticFiles(directory='image-training'), name='images')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def remove_text_between_sign(input_string):
    modified_string = re.sub(r'\*.*?\*', '', input_string)
    modified_string = re.sub(r'\<.*?\>', '', modified_string)
    modified_string = re.sub(r'\(.*?\)', '', modified_string)
    modified_string = modified_string.replace("\\", "")
    return modified_string


def verify_token(req: Request):
    try:
        token = req.headers["Authorization"]
        user = user_table.find_one({"auth_token":token})
        if not user:
            raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )
        return user
    except:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

@app.post('/api/v1/create-character')
async def save_chat_character(response: Response, request:Request):
    user = verify_token(request)
    form = await request.form()
    picture = form['picture']
    name = form.get("name", "")
    greeting = form.get("greeting", "")
    context = form.get("context", "")
    description = form.get("description", "")
    tts_voice = form.get("tts_voice", None)
    voices = form.getlist("voice")
    filename = picture.filename.lower() 
    filename_ext = filename.split(".")[-1]
    allowed_exts = ['png', 'jpg', 'jpeg']
    if filename_ext not in allowed_exts:
        response.status_code = 400
        return {'message':"Please upload Character Image with PNG, JPG, JPEG extension."}
    datetime_str = str(datetime.datetime.now().timestamp()).split(".")[0]
    filename = datetime_str
    img = Image.open(io.BytesIO(picture.file.read()))
    save_character(name, greeting, context, img, filename)
    char = {"user":user["_id"],"name":name, "greeting":greeting, "context":context, "description":description, \
             "image":filename+'.png', "tts_voice":tts_voice, "filename":filename}
    if not tts_voice:
        del char["tts_voice"]
    if character_table.find_one({"user":user}):
        del char["user"]
        character_table.update_one({"user":user}, {"$set":char})
    else:
        character_table.insert_one(char)
    if tts_voice and len(voices):
        print(voices, " .>>>>>>>>>>>>.. ")
        email = user['email'].split("@")[0]
        voice_path = Path("/home/ubuntu/onlyfantasy-python/voice-training/"+email)
        if not voice_path.exists():
            voice_path.mkdir(parents=True, exist_ok=True)
        for i, voice in enumerate(voices):
            print(voice, " .>>>>>>>>>>>>.. ")
            voice_content = voice.file.read()
            with open(f"/home/ubuntu/onlyfantasy-python/voice-training/{email}/voice.mp3", "ab") as file:
                file.write(voice_content)
                file.close()
        redis_client.rpush("queue_rvc_train", email)
    return {'message':'Character saved successfully.'}

@app.post('/api/v1/add-voice-to-character')
async def add_voice_to_character(request:Request, response:Response):
    user = verify_token(request)
    form = await request.form()
    voices = form.getlist("voice")
    email = user['email'].split("@")[0]
    tts_voice = form.get("tts_voice", "")
    char_id = form.get("char_id", "")
    character_table.update_one({"_id":ObjectId(char_id)}, {"$set":{"tts_voice":tts_voice}})
    voice_path = Path("/home/ubuntu/onlyfantasy-python/voice-training/"+email)
    if not voice_path.exists():
        voice_path.mkdir(parents=True, exist_ok=True)
    for i, voice in enumerate(voices):
        voice_content = voice.file.read()
        with open(f"/home/ubuntu/onlyfantasy-python/voice-training/{email}/voice.mp3", "ab") as file:
            file.write(voice_content)
            file.close()
    redis_client.rpush("queue_rvc_train", email)
    return {"message":"Voice added to the character.", "status":1}


def train_voice(user):
    train_dir = "/home/ubuntu/onlyfantasy-python/voice-training/"+user
    python_cmd = "/home/ubuntu/rvc/bin/python train_rvc.py --trainset_dir %s --exp_dir %s" % (train_dir, user)
    Popen(python_cmd, shell=True, cwd=f"{OUTPUT_BASE_DIR}{RVC_FOLDER}")
    # run_cmd = os.system(python_cmd)
    return_proj_dir()
    # if run_cmd != 0:
    #     redis_client.rpush("queue_rvc_train", user)


@app.get("/api/v1/get-available-characters")
async def get_characters(request:Request):
    user = verify_token(request)
    data = character_table.find()
    characters = []
    for c in data:
        ch_data = {}
        ch_data["id"] = str(c['_id'])
        ch_data["name"] = c.get('name', '')
        ch_data["image"] = c.get('image', '')
        ch_data["image_bot"] = True if c.get("sd_keyword") else False
        ch_data["chat_bot"] = True if c.get("context") else False
        ch_data["voice"] = True if c.get("tts_voice") else False
        characters.append(ch_data)
    characters.reverse()
    return characters


@app.post("/api/v1/get-chat-history")
async def get_chat_history(request: Request):
    user = verify_token(request)
    body = await request.body()
    body = json.loads(body)
    character = body.get('character', '')
    history = chat_history.find({"character":ObjectId(character), "user":ObjectId(user["_id"])})
    user_history = []
    room_id = ""
    if len(list(history)) == 0:
        room_id = str(uuid.uuid1())
        rooms_table.insert_one({"room":room_id, "character":ObjectId(character), "user":user["_id"]})
        character_data = character_table.find_one({"_id":ObjectId(character)})
        if character_data:
            chat_internal = [
                "<|BEGIN-VISIBLE-CHAT|>",
                character_data['greeting']
            ]
            chat_visible = [
                "",
                character_data['greeting']
            ]
            chat_history.insert_one({"user":user["_id"], "internal":chat_internal, "visible":chat_visible, "created_at":datetime.datetime.utcnow(), \
                                    "character":ObjectId(character), "room":room_id})
        else:
            raise HTTPException(status_code=400, detail="Character not found.")
    room_id = rooms_table.find_one({"character":ObjectId(character), "user":user["_id"]})["room"]
    history = chat_history.find({"character":ObjectId(character), "user":ObjectId(user["_id"])})
    for chat in history:
        user_history.append({"chat":chat['visible'], "datetime":chat["_id"].generation_time.strftime("%Y-%m-%d %H:%M:%S"), "_id":str(chat['_id']), "audio_url": chat.get("audio_url", None)})
    return {"room":room_id, "history": user_history}

@app.get("/api/v1/tts-voices")
async def get_tts_voices(request: Request):
    voices = tts_voices_table.find({})
    return json.loads(json_util.dumps(voices))

def get_playht_voice(transcription_id, count=0):
    time.sleep(2)
    if count==5:
        return None
    playable_tts_url = PLAYHT_AUDIO_STATUS_URL+transcription_id
    tts_headers = {
        "accept": "application/json",
        "AUTHORIZATION": PLAYHT_SECRET_KEY,
        "X-USER-ID": PLAYHT_ACCESS_KEY
        }
    play_tts_response = requests.get(playable_tts_url, headers=tts_headers)
    response = play_tts_response.json()
    if response['converted']:
        return response['audioUrl']
    else:
        print("<<<< \t get_playht_voice recalled \t >>>> ".upper())
        return get_playht_voice(transcription_id, count=count+1)
    
def return_proj_dir():
    os.chdir("/home/ubuntu/onlyfantasy-python")

def download_file(url, target_directory, new_filename):
    response = requests.get(url)

    if response.status_code == 200:
        file_path = os.path.join(target_directory, new_filename)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded and saved to {file_path}")
    else:
        print("Failed to download the file.")

def get_added_index_filename(dir):
    for i in os.scandir(dir):
        if i.name.startswith("added"):
            return i.path

async def save_images(email, images, gender, repetations):
    training_folder_path = f"{OUTPUT_BASE_DIR}onlyfantasy-python/image-training/{email}/images/{repetations}_{email} {gender}"  # path of images folder for lora training for SD.
    path = Path(training_folder_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    for image in images:
        res = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        file_path = training_folder_path+f"/{email}_{res}.jpg"
        with open(file_path, "wb") as f:
            f.write(image.file.read())
            f.close()
    return "images saved"

@app.post("/api/v1/image-captions")
async def create_image_captions(request:Request, response:Response):
    user = verify_token(request)
    email = user['email'].split("@")[0]
    form = await request.form()
    images = form.getlist("images")
    gender = form.get("gender", "woman")
    repetations = str(form.get("repetations", "10"))
    training_folder_path = f"{OUTPUT_BASE_DIR}onlyfantasy-python/image-training/{email}/images/{repetations}_{email} {gender}"
    captioning_path = f"{OUTPUT_BASE_DIR}onlyfantasy-python/image-training/{email}/images/{repetations}_{email}\ {gender}"
    if len(images):
        save_images_func = await save_images(email, images, gender, repetations)
    character_obj = character_table.find_one({"user":user["_id"]})
    if character_obj:
        update_dict = {"sd_keyword":email, "sd_model":sd_model, "name":email, "repetations":form['repetations'], "gender":gender}
        if character_obj.get("name", ""):
            update_dict["name"] = character_obj["name"]
        character_table.update_one({"_id":character_obj["_id"]}, {"$set":update_dict})
    else:
        character_table.insert_one({"user":user["_id"],"sd_keyword":email, "sd_model":sd_model, "repetations":form['repetations'], "gender":gender})
    
    print("TRAINING IMAGES SAVED...")
    os.chdir("/home/ubuntu/kohya_ss")
    python_cmd = "/home/ubuntu/kohya_ss/venv/bin/python /home/ubuntu/kohya_ss/finetune/make_captions.py --batch_size=2 --num_beams=1 --top_p=0.9 --max_length=200 \
        --min_length=7 --beam_search --caption_extension=.txt %s \
        --caption_weights=https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth --prefix=%s" % (captioning_path, " ")
    run_cmd = os.system(python_cmd)
    print("CMD OUTPUT CAPTIONS ... "+email+"\t", run_cmd)
    return_proj_dir()
    if run_cmd == 0:
        img_array = []
        for i in os.listdir(training_folder_path):
            if i.endswith(".jpg"):
                f_name = i.split(".")[0]
                img_array.append(f_name)
        return {"folder_path":f"{email}/images/{repetations}_{email} {gender}", "images":img_array}
    else:
        response.status_code == 400
        return {"message":"An error occured while captioning images. Please try again later.", "status":run_cmd}

@app.post("/api/v1/image-training")
async def proceed_image_training(request:Request, response:Response):
    user = verify_token(request)
    email = user['email'].split("@")[0]
    form_data = await request.body()
    form_data = json.loads(form_data)
    gender = form_data.get("gender", "woman")
    form_data['epochs'] = form_data.get("epochs", 12)
    form_data['gender'] = form_data.get("gender", "woman")
    form_data['repetations'] = form_data.get("repetations", "10")
    training_folder_path = f"{OUTPUT_BASE_DIR}onlyfantasy-python/image-training/{email}/images/{form_data['repetations']}_{email} {gender}"  # path of images folder for lora training for SD.
    images = form_data.get("images")
    for img in images:
        for k,v in img.items():
            with open(training_folder_path+"/%s.txt" % k, "w+") as file:
                file.write(v)
                file.close()
    character_obj = character_table.find_one({"user":user["_id"]})
    if character_obj:
        update_dict = {"sd_keyword":email, "sd_model":sd_model, "name":email, "repetations":form_data['repetations'], "gender":gender}
        if character_obj.get("name", ""):
            update_dict["name"] = character_obj["name"]
        character_table.update_one({"_id":character_obj["_id"]}, {"$set":update_dict})
    else:
        character_table.insert_one({"user":user["_id"],"sd_keyword":email, "sd_model":sd_model, "repetations":form_data['repetations'], "gender":gender})
    del form_data['images']
    form_data['email'] = email
    form_data['user_id'] = str(user["_id"])
    redis_client.rpush("queue_sd", json.dumps(form_data))
    return {"message":"Image captions saved successfully. Your images are being used for training."}

@app.get("/api/v1/get-training-images")
async def get_training_images(request:Request, response:Response):
    user = verify_token(request)
    email = user['email'].split("@")[0]
    cwd = os.getcwd()
    character_obj = character_table.find_one({"user":user["_id"]})
    imagebot = False
    if character_obj and character_obj.get("sd_keyword", None):
        imagebot = True
    img_array = []
    repetations = "10"
    gender = "woman"
    if character_obj:
        repetations = character_obj.get("repetations")
        gender = character_obj.get("gender")
    if Path(f"{cwd}/image-training/{email}/images/{repetations}_{email} {gender}").exists():
        for i in os.listdir(f"{cwd}/image-training/{email}/images/{repetations}_{email} {gender}"):
            if i.endswith(".jpg") or i.endswith(".jpeg"):
                f_name = i.split(".")[0]
                img_array.append(f_name)
        return {"folder_path":f"{email}/images/{repetations}_{email} {gender}", "images":img_array, "imagebot":imagebot, "repetations":repetations, "gender":gender}    
    else:
        response.status_code == 404
        return {"message":"No images found."}
    
@app.delete("/api/v1/delete-image/{image}")
async def delete_image(request:Request, response:Response, image:str):
    user = verify_token(request)
    email = user['email'].split("@")[0]
    cwd = os.getcwd()
    character_obj = character_table.find_one({"user":user["_id"]})
    repetations = character_obj.get("repetations", "10")
    gender = character_obj.get("gender", "woman")
    path_str1 = f"{cwd}/image-training/{email}/images/{repetations}_{email} {gender}"
    path_str2 = f"{cwd}/image-training/{email}/images/{repetations}_{email} {gender}"
    if os.path.exists(f"{path_str1}/{image}.jpg"):
        os.remove(f"{path_str1}/{image}.jpg")
        os.remove(f"{path_str1}/{image}.txt")
        return {"message":"Image deleted."}
    elif os.path.exists(f"{path_str2}/{image}.jpg"):
        os.remove(f"{path_str2}/{image}.jpg")
        os.remove(f"{path_str2}/{image}.txt")
        return {"message":"Image deleted."}
    else:
        response.status_code = 400
        return {"message":"Image not found."}
    
@app.delete("/api/v1/delete-image-bot")
async def delete_imagebot(request:Request, response:Response):
    user = verify_token(request)
    email = user['email'].split("@")[0]
    cwd = os.getcwd()
    path_str = f"{cwd}/image-training/{email}"
    if os.path.exists(f"{path_str}"):
        shutil.rmtree(f"{path_str}")
        character_obj = character_table.find_one({"user":user["_id"]})
        if not character_obj:
            return {"message":"Image bot not found."}
        character_table.update_one({"_id":character_obj["_id"]}, {"$set":{"sd_keyword":"", "sd_model":""}})
        return {"message":"Image Bot deleted."}
    else:
        response.status_code = 400
        return {"message":"Image Bot not found."}


def process_sd_lora(form_data):
    form_data = json.loads(form_data)
    user = form_data['email']
    user_id = form_data['user_id']
    epochs = form_data['epochs']
    logs_path = Path("/home/ubuntu/onlyfantasy-python/image-training/%s/logs" % user)
    if not logs_path.exists():
        logs_path.mkdir(parents=True, exist_ok=True)
    # python_cmd = f". /home/ubuntu/kohya_ss/venv/bin/activate && accelerate launch --num_processes=1 /home/ubuntu/kohya_ss/train_network.py --enable_bucket \
    # --min_bucket_reso=256 --max_bucket_reso=1024 --pretrained_model_name_or_path=/home/ubuntu/stable-diffusion-webui/models/Stable-diffusion/{sd_model}.safetensors \
    # --train_data_dir=/home/ubuntu/onlyfantasy-python/image-training/{user}/images --resolution=512,512 \
    # --output_dir=/home/ubuntu/stable-diffusion-webui/models/Lora --logging_dir=/home/ubuntu/onlyfantasy-python/image-training/{user}/logs \
    # --network_alpha=64 --save_model_as=safetensors --network_module=networks.lora --text_encoder_lr=0.00005 --unet_lr=0.0001 --network_dim=128 \
    # --output_name={user} --lr_scheduler_num_cycles=1 --no_half_vae --learning_rate=0.0001 --lr_scheduler=cosine --lr_warmup_steps=500 \
    # --train_batch_size=1 --save_every_n_epochs={epochs} --mixed_precision=fp16 --save_precision=fp16 --max_train_epochs={epochs} --caption_extension=.txt --clip_skip=1"

    python_cmd = f". /home/ubuntu/kohya_ss/venv/bin/activate && accelerate launch --num_processes=1 /home/ubuntu/kohya_ss/train_network.py --enable_bucket --min_bucket_reso=256 --max_bucket_reso=1024 --pretrained_model_name_or_path=/home/ubuntu/stable-diffusion-webui/models/Stable-diffusion/brav5.safetensors --train_data_dir=/home/ubuntu/onlyfantasy-python/image-training/{user}/images --resolution=512,512 --output_dir=/home/ubuntu/stable-diffusion-webui/models/Lora --network_alpha=64 --save_model_as=safetensors --network_module=networks.lora --network_dim=128 --output_name={user} --lr_scheduler_num_cycles={epochs} --no_half_vae --learning_rate=1.0 --lr_scheduler=cosine --lr_warmup_steps=138 --train_batch_size=1 --max_train_steps=1380 --save_every_n_epochs={epochs} --mixed_precision=fp16 --save_precision=fp16 --caption_extension=.txt --cache_latents --optimizer_type=Prodigy --max_data_loader_n_workers=0 --bucket_reso_steps=64 --xformers --bucket_no_upscale --noise_offset=0.0"


    output_cmd = Popen(python_cmd, shell=True, cwd=f"{OUTPUT_BASE_DIR}kohya_ss")
    print(python_cmd)
    # if output_cmd.returncode != 0:
    #     redis_client.rpush("queue_sd", user)
    return_proj_dir()

def save_base64_image(base64_string, file_name, char):
    import base64
    from io import BytesIO

    try:
        image_bytes = base64.b64decode(base64_string)
        img_bytes = BytesIO(image_bytes)

        s3_client = boto3.client('s3',aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)
        s3_client.put_object(Body=img_bytes, Bucket=aws_bucket_name, Key=f"generated_images/{char}/{file_name}.png", ACL="public-read")

        return f"https://onlyfantasy.s3.ap-northeast-1.amazonaws.com/generated_images/{char}/{file_name}.png"

        # SAVE IMAGE TO LOCAL DIR..
        # image = Image.open(img_bytes)
        # image.save(f"/home/ubuntu/onlyfantasy-python/generated_images/{file_name}.png")
        # print(f"Image saved as {file_name}")
    except Exception as e:
        print("An error occurred:", e)
        return None

@app.post("/api/v1/generate-image")
async def generate_image(request:Request, response:Response):
    user = verify_token(request)
    body = await request.body()
    body = json.loads(body)
    character = body.get("character")
    prompt = body.get("prompt")
    steps = body.get("steps", "50")
    scale = int(body.get("scale", 7))
    width = int(body.get("width", 512))
    height = int(body.get("height", 512))
    # strength = body.get("strength", "1")
    character_obj = character_table.find_one({"_id":ObjectId(character)})
    file_name = str(datetime.datetime.now().timestamp()).split(".")[0]
    sd_keyword = character_obj.get("sd_keyword", None)
    sd_model = character_obj.get("sd_model", None)
    if not sd_keyword:
        response.status_code = 400
        return {"message":"Image bot does not exists."}
    elif not os.path.exists(f"/home/ubuntu/stable-diffusion-webui/models/Lora/{sd_keyword}.safetensors"):
        response.status_code = 400
        return {"message":"Please wait for few minutes as image bot is in process of training."}
        
    negative_prompt = "(sketch,cartoon,anime),(worst quality,low quality,normal quality:2), (lowres),monochrome,greyscale, disfigured,ugly,child, childish,extra legs,long neck, (deformed iris, deformed pupils), robot eyes, (asymmetrical eyes), closed eye, (imperfect eyes), huge breasts, non-asian woman"
    sd_request = {
        "prompt": f"{prompt}, <lora:{sd_keyword}:1>",
        "negative_prompt": negative_prompt,
        "steps":steps,
        "sampler_name":"DPM++ SDE Karras",
        "sampler_index":"DPM++ SDE Karras",
        "cfg_scale": scale,
        "seed":-1,
        "face_restoration":"CodeFormer",
        "face_restoration_model":"CodeFormer",
        "restore_faces":True,
        "width":width,
        "height":height,
        "denoising_strength":0.7,
        "token_merging_ratio":0.5,
        "hr_token_merging_ratio":0.5,
        "hires_token_merging_ratio":0.5,
        "hr_scale":1,
        "hires_scale":1,
        "hr_upscaler":None,
        "sd_model_checkpoint":sd_model
    }
    sd_resp = requests.post("http://3.113.146.134:7861/sdapi/v1/txt2img",
                            headers={"Content-Type": "application/json", 'accept': 'application/json'},
                            json=sd_request
                            )
    if sd_resp.status_code == 200:
        sd_resp = sd_resp.json()
        img = sd_resp.get('images', [])[0]
        generated_img = save_base64_image(img, file_name, character)
        if generated_img:
            generated_images_table.insert_one({"user":user["_id"], "image":f"{file_name}.png", "character":character_obj["_id"], "url":generated_img})
            return {"image":generated_img}
    else:
        response.status_code = sd_resp.status_code
        return {"message":"An error occured.", "error":sd_resp.text}


def upload_voice_to_s3(file, file_name, char):
    s3_client = boto3.client('s3',aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)
    if os.path.exists(file):
        s3_client.upload_file(file, aws_bucket_name, f"generated_voices/{char}/{file_name}.mp3", ExtraArgs={'ACL': 'public-read'})
        os.remove(file)


def process_tts_redis(msg_id):
    message = chat_history.find_one({"_id":ObjectId(msg_id)})
    character = message["character"]
    obj_character = character_table.find_one({"_id":character})
    char_user = user_table.find_one({"_id":obj_character["user"]})
    RVC_PATH = char_user['email'].split("@")[0]
    # RVC_PATH = obj_character["rvc_voice"] # folder name in rvc logs folder where model files are stored.
    script_path = OUTPUT_BASE_DIR + RVC_FOLDER + "infer_cli.py "
    text_to_convert = message['visible'][1]
    text_content = remove_text_between_sign(text_to_convert)
    tts_paylod = {
        "content":[text_content],
        "voice":obj_character["tts_voice"]
    }
    tts_headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "AUTHORIZATION": PLAYHT_SECRET_KEY,
        "X-USER-ID": PLAYHT_ACCESS_KEY
    }
    if text_content:
        tts_response = requests.post(PLAYHT_CONVERT_TTS, json=tts_paylod, headers=tts_headers)
        if tts_response.status_code == 400:
            # handle error of play.ht and store them in db to be processed later.
            chat_history.update_one({"_id":message["_id"]}, { "$set": { "rvc":0 } })
        tts_response_json = tts_response.text
        tts_response_json = json.loads(tts_response_json)
        print("tts_response_jsontts_response_jsontts_response_json \t ", tts_response_json)
        transcription_id = tts_response_json.get('transcriptionId', None)
        if transcription_id:
            audioUrl = get_playht_voice(transcription_id)
            if not audioUrl:
                return True
            PLAYHT_FILENAME = str(datetime.datetime.now().timestamp()).split(".")[0]
            TEMP_DIR = OUTPUT_BASE_DIR+"voices/"
            download_file_temp = download_file(audioUrl, TEMP_DIR, PLAYHT_FILENAME+".mp3")
            PYTHON_PATH = OUTPUT_BASE_DIR + "rvc/bin/python3 "
            TRANSPOSE_VALUE = "0 "
            WEIGHT_PATH = OUTPUT_BASE_DIR + RVC_FOLDER+"weights/" + RVC_PATH+".pth "
            RVC_USER_FOLDER = OUTPUT_BASE_DIR + RVC_FOLDER + "logs/" + RVC_PATH
            INDEX_FILE = get_added_index_filename(RVC_USER_FOLDER)
            print('INDEX_FILE \t %s' % INDEX_FILE )
            VOICE_OUTPUT_FILE = OUTPUT_BASE_DIR + "onlyfantasy-python/voices/" + PLAYHT_FILENAME+ ".mp3 "
            PYTHON_CMD = PYTHON_PATH + script_path + TRANSPOSE_VALUE + TEMP_DIR + PLAYHT_FILENAME+".mp3 " + VOICE_OUTPUT_FILE+ WEIGHT_PATH+ INDEX_FILE +" cuda:2 " + "crepe"
            print('running command \t %s' % PYTHON_CMD )
            os.chdir(f"{OUTPUT_BASE_DIR}{RVC_FOLDER}")
            run_cmd = os.system(PYTHON_CMD)
            print("run_cmd output \t :: %s" % run_cmd)
            if run_cmd == 0:
                upload_voice_to_s3(VOICE_OUTPUT_FILE, PLAYHT_FILENAME, str(character))
            return_dir = return_proj_dir()
            print("FILE >>>>>>>>>>>> ", upload_voice_to_s3(VOICE_OUTPUT_FILE, PLAYHT_FILENAME, str(character)))
            chat_history.update_one({"_id":message["_id"]}, { "$set": { "transcription_id": transcription_id, "audio_url":PLAYHT_FILENAME+".mp3", "rvc":"1" } })
            return True
        else:
            chat_history.update_one({"_id":message["_id"]}, { "$set": { "error":"1" } })
    return True

@app.post("/api/v1/get-tts-status")
async def get_tts_status(request:Request):
    user = verify_token(request)
    body = await request.body()
    body = json.loads(body)
    msg_id = body['message_id']
    message = chat_history.find_one({"_id":ObjectId(msg_id)})
    if message.get("rvc", 0):
        return {"status":message.get("rvc", 0), "audio_url":message.get("audio_url", "")}
    else:
        return {"status":message.get("rvc", 0), "message":"An error occured."}
    

@app.post("/api/v1/message-tts")
async def message_tts(request: Request, resp:Response):
    user = verify_token(request)
    body = await request.body()
    body = json.loads(body)
    msg_id = body['message_id']
    message = chat_history.find_one({"_id":ObjectId(msg_id), "user":ObjectId(user["_id"])})
    obj_character = character_table.find_one({"_id":message["character"]})
    char_user = user_table.find_one({"_id":obj_character["user"]})
    char_user_name = char_user['email'].split(".")[0]
    if message:
        if message.get("rvc", 0):
            audioUrl = message['audio_url']
            return {"audio_url":audioUrl, "status":1}
        else:
            redis_client.rpush("queue_rvc", str(message["_id"]))
            return {"message":"TTS added to queue.", "status":0}
        
        #     return {"audio_url":PLAYHT_FILENAME+".mp3"}
        # else:
        #     response.status_code == 400
        #     return {"message":"Expression messages cannot be played."}
    else:
        raise HTTPException(status_code=404, detail="Message not found.")

# WEBSOCKETS FROM HERE.
@app.websocket("/api/v1/chat-stream/{room_id}")
async def handle_chat_stream_message(websocket:WebSocket, room_id:str):
    await socket_manager.connect(websocket)
    try:
        while True:
            body = await websocket.receive_json()
            last_msg_internal = ""
            last_msg_visible = ""
            user_input = body['user_input']
            user_auth = body['token']
            character = body.get("character", "")
            db_user = user_table.find_one({"auth_token":user_auth})
            if not db_user:
                await websocket.send_json({"error":"Unauthorized."})
            else:
                body['max_new_tokens'] = 1200
                body['min_length'] = 150
                generate_params = build_parameters(body, chat=True, is_api=True)
                generate_params['stream'] = True
                regenerate = False
                _continue = False
                old_history = {'internal': [], 'visible': []}
                db_history = chat_history.find({"user":db_user["_id"], "character":ObjectId(character)})
                if len(list(db_history)):
                    for h in db_history:
                        old_history['internal'].append(h['internal'])
                        old_history['visible'].append(h['visible'])
                    generate_params['history'] = old_history
                    

                generator = generate_chat_reply(user_input, generate_params, False, False, loading_message=False)
                message_num = 0
                for a in generator:
                    await websocket.send_json({
                        'event': 'text_stream',
                        'message_num': message_num,
                        'message': a['visible'][-1][-1]
                    })
                    last_msg_internal = a['internal'][-1]
                    last_msg_visible = a['visible'][-1]


                    await asyncio.sleep(0)
                    message_num += 1


                last_msg = chat_history.insert_one({"user":db_user["_id"], "internal":last_msg_internal, "visible":last_msg_visible, \
                                                     "created_at":datetime.datetime.utcnow(), "character":ObjectId(character), "room":room_id})

                await websocket.send_text(json.dumps({
                    'event': 'stream_end',
                    'message_num': message_num,
                    'message_id':str(last_msg.inserted_id)
                }))
    except WebSocketDisconnect:
        pass

# queues
def queue_train_rvc_model(t=None):
    print("thread started queue RVC TRAINING >>>>>")
    while True:
        task = redis_client.lpop("queue_rvc_train")
        if task:
            t = Thread(target=train_voice, args=(task.decode("utf-8"),))
            t.setDaemon(True)
            t.start()
        else:
            time.sleep(1)


def queue_thread_diffusion(t=None):
    print("thread started queue DIFFUSION >>>>>>>>>>>>>>>>>>>>>")

    while True:
        task = redis_client.lpop("queue_sd")
        if task:
            t = Thread(target=process_sd_lora, args=(task.decode("utf-8"),))
            t.setDaemon(True)
            t.start()
        else:
            time.sleep(1)

def queue_thread_rvc(t=None):
    print("thread started queue RVC >>>>>>>>>>>>>>>>>>>>>")
    while True:
        task = redis_client.lpop("queue_rvc")
        if task:
            t = Thread(target=process_tts_redis, args=(task.decode("utf-8"),))
            t.setDaemon(True)
            t.start()
        else:
            time.sleep(1)

def start_server():
    t = Thread(target=uvicorn.run, kwargs={'app':app, 'host':'0.0.0.0', 'port':5000})
    t.setDaemon(True)
    t.start()
    
    t1 = Thread(target=queue_thread_rvc, args=(1,))
    t1.setDaemon(True)
    t1.start()

    t2 = Thread(target=queue_thread_diffusion, args=(1,))
    t2.setDaemon(True)
    t2.start()

    t3 = Thread(target=queue_train_rvc_model, args=(1,))
    t3.setDaemon(True)
    t3.start()

