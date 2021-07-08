import os
import numpy as np
import torch

import sys
sys.path.append('../models')

from kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer

# root_path='drive/My Drive/Colab Notebooks/dialogLM'
root_path = './'
data_path = f"{root_path}/chatbot_dialog.txt"
checkpoint_path =f"{root_path}/checkpoint"
save_ckpt_path = f"{root_path}/kogpt2-chatbot-dialog.pth"
#save_ckpt_path = f"{checkpoint_path}/kogpt2-chatbot-wellness.pth"

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# 저장한 Checkpoint 불러오기
checkpoint = torch.load(save_ckpt_path, map_location=device)

model = DialogKoGPT2()
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

tokenizer = get_kogpt2_tokenizer()

count = 0
output_size = 200 # 출력하고자 하는 토큰 갯수

questions = ["안녕하세요",
             "날씨가 어때요?",
             "사랑해요",
             "오늘은 웬지 우울해요",
             "여자친구랑 헤어졌어요"]

#while 1:
# for i in range(5):
for sent in questions:
  #sent = input('Question: ')  # '요즘 기분이 우울한 느낌이에요'
  tokenized_indexs = tokenizer.encode(sent)

  input_ids = torch.tensor([tokenizer.bos_token_id,]  + tokenized_indexs +[tokenizer.eos_token_id]).unsqueeze(0)
  # set top_k to 50
  sample_output = model.generate(input_ids=input_ids)


  print("Answer: " + tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs)+1:],skip_special_tokens=True))
  print(100 * '-')