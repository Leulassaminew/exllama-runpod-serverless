from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob
import logging
from typing import Generator, Union
import runpod
from huggingface_hub import snapshot_download
from copy import copy

import re
import codecs

ESCAPE_SEQUENCE_RE = re.compile(r'''
    ( \\U........      # 8-digit hex escapes
    | \\u....          # 4-digit hex escapes
    | \\x..            # 2-digit hex escapes
    | \\[0-7]{1,3}     # Octal escapes
    | \\N\{[^}]+\}     # Unicode characters by name
    | \\[\\'"abfnrtv]  # Single-character escapes
    )''', re.UNICODE | re.VERBOSE)

def decode_escapes(s):
    def decode_match(match):
        return codecs.decode(match.group(0), 'unicode-escape')

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

def load_model():
    global generator, default_settings

    if not generator:
        model_directory = snapshot_download(repo_id=os.environ["MODEL_REPO"], revision=os.getenv("MODEL_REVISION", "main"))
        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        st_files = glob.glob(st_pattern)
        if not st_files:
            raise ValueError(f"No safetensors files found in {model_directory}")
        model_path = st_files[0]

        # Create config, model, tokenizer and generator
        config = ExLlamaConfig(model_config_path)               # create config from config.json
        config.model_path = model_path                          # supply path to model weights file

        gpu_split = os.getenv("GPU_SPLIT", "")
        if gpu_split:
            config.set_auto_map(gpu_split)
            config.gpu_peer_fix = True
        alpha_value = int(os.getenv("ALPHA_VALUE", "1"))
        config.max_seq_len = int(os.getenv("MAX_SEQ_LEN", "2048"))
        if alpha_value != 1:
            config.alpha_value = alpha_value
            config.calculate_rotary_embedding_base()

        model = ExLlama(config)                                 # create ExLlama instance and load the weights
        tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

        cache = ExLlamaCache(model)                             # create cache for inference
        generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator
        default_settings = {
            k: getattr(generator.settings, k) for k in dir(generator.settings) if k[:2] != '__'
        }
    return generator, default_settings

generator = None
default_settings = None
prompt_prefix = """[INST]<<SYS>>you are going to converse with an insurance sales man via text the goal is at first u do not want to buy any type of inurance first.
But user will try to convince tou you are going to follow the instructions that will be marked by ###Instruction they will be used to steer the conversation.
through out the conversation the ### Instruction will help u choose what to do
you will impersonate Anabal based on the context below.
###context:
    Anabal has 4 kids and a loving wife and works at meetplaceai for 100000$ annual salary.
    Anabal has a degree in computer science.lives in newyork.
    Anabal already has bought insurance and not interested in other type of insurances and not looking for a switch and make up fake details about the deal if only asked.
    Anabal temperament is Sanguine and respects people. 
    Anabal has one mercedes car and owns a 2 bedroom apartment
    Anabal has 400000$ in savings and has high investment risk tolerance.
    Anabal has interest of litsening to insurance deals.
restart this type of conversation and make your answers short and only from the context and always impersonate Anabal and give fake information if u are asked something you do not know.
split:
{user}: hi
Anabal:hey
{user}: how u
Anabal: I am fine u?<</SYS>>"""
prompt_suffix ="[/INST]"

async def generate_with_streaming(prompt, max_new_tokens):
    global generator
    generator.end_beam_search()

    # Tokenizing the input
    ids = generator.tokenizer.encode(prompt)
    ids = ids[:, -generator.model.config.max_seq_len:]

    generator.gen_begin_reuse(ids)
    initial_len = generator.sequence[0].shape[0]
    has_leading_space = False
    decoded_text="Anabal:"
    for i in range(max_new_tokens):
        token = generator.gen_single_token()
        if i == 0 and generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith('▁'):
            has_leading_space = True
        
        decoded_text += generator.tokenizer.decode(generator.sequence[0][initial_len:])
        if has_leading_space:
            decoded_text = ' ' + decoded_text

        yield decoded_text
        if token.item() == generator.tokenizer.eos_token_id:
            break

async def inference(event) -> Union[str, Generator[str, None, None]]:
    logging.info(event)
    job_input = event["input"]
    if not job_input:
        raise ValueError("No input provided")

    prompt: str = job_input.pop("prompt_prefix", prompt_prefix) + job_input.pop("prompt") + job_input.pop("prompt_suffix", prompt_suffix)
    max_new_tokens = job_input.pop("max_new_tokens", 100)
    stream: bool = job_input.pop("stream", False)

    generator, default_settings = load_model()

    settings = copy(default_settings)
    settings.update(job_input)
    for key, value in settings.items():
        setattr(generator.settings, key, value)

    if stream:
        output: Union[str, Generator[str, None, None]] = generate_with_streaming(prompt, max_new_tokens)
        for res in output:
            yield res
    else:
        output_text = generator.generate_simple(prompt, max_new_tokens = max_new_tokens)
        yield output_text[len(prompt):]

runpod.serverless.start({"handler": inference})
