import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import os

parser = argparse.ArgumentParser()
    # required arguments
parser.add_argument('--data', default = 'youtube', type =str)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--embedding_path', default = './embedding/', type = str)
args = parser.parse_args()




tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", return_dict=True, output_hidden_states=True)
print("Loaded!")
model.eval()
model.parallelize()
print("Moved model to GPU!")
if args.data.lower() == 'youtube':
    prompts_list = [
        "Does the following comment talk about a song?",
        "Is the following comment fewer than 5 words?",
        "Does the following comment mention a person’s name?",
        "Does the following comment express a very strong sentiment?",
        "Does the following comment express a subjective opinion?",
        "Does the following comment reference the speaker’s channel or video?",
        "Does the following comment ask you to subscribe to a channel?",
        "Does the following comment have a URL?",
        "Does the following comment ask the reader to do something?",
        "Does the following comment contain the words \"check out\"?"
    ]
    prompt_augmented = [
        "Is the following comment talking about a song?",
        "Is the following comment less than 5 words?",
        "Does the following comment mention a person by name?",
        "Are the following comments expressing very strong emotions?",
        "Are the following comments expressing subjective opinions?",
        "Does the following comment reference the speaker's channel or video?",
        "Does the following comment require you to subscribe to the channel?",
        "Does the following comment have a URL?",
        "Does the following comment ask the reader to do something?",
        "Does the following comment contain the word \"check out\"?"
        ]
elif args.data.lower() == 'sms':
    prompts_list =[
'??1.50',
'??500' , 
'??5000', 
'call for offer', 
'cash prize', 
'chat date', 
'chat to', 
'childporn', 
'credits',
'dating call', 
'direct', 
'expires now', 
'fantasies call', 
'free phones', 
'free price', 
'free ringtones', 
'free sex', 
'free tone', 
'guaranteed free', 
'guaranteed gift', 
'hard live girl', 
'important lucky', 
'inviting friends', 
'latest', 
'latest offer', 
'message call', 
'new mobiles', 
'no extra', 
'password', 
'please call', 
'sms reply', 
'unlimited calls', 
'urgent award guaranteed', 
'urgent prize', 
'voucher claim', 
'welcome reply', 
'win shopping', 
'winner reward', 
'won call', 
'won cash', 
'won cash prize', 
'won claim',
'I', 
'I can did', 
'I it', 
'I miss', 
'I used to', 
'adventuring', 
'amrita', 
"can't talk", 
'did u got', 
'do you', 
'fb', 
'goodo', 
'hee hee', 
"i'll", 
'jus', 
'link', 
'maggi', 
'mine', 
'my kids', 
'noisy', 
'praying', 
'shit', 
'should I', 
'thanks', 
"that's fine", 
'thats nice', 
'u how 2',
'we will', 
'where are', 
'wtf', 
'your I'
]

    prompt_augmented =[
    '??1.50',
'??500' , 
'??5000', 
"request a quote",
"cash prize",
"chat date",
"chat with",
"child pornography",
"credit",
"dating phone",
"direct",
"Expires now",
"fantasy phone",
"toll free",
"free price",
"free ringtones",
"free sex",
"free tones",
"Guaranteed free",
"guaranteed gift",
"hard life girl",
"important luck",
"Invite friends",
"Newest",
"latest offer",
"Voicemail",
"new phone",
"no extra",
"password",
"please call",
"SMS reply",
"Unlimited calls",
"Emergency Reward Guarantee",
"emergency bonus",
"Voucher claim",
"Welcome to reply",
"win shopping",
"Winner's Reward",
"win the call",
"win cash",
"win cash prizes",
"win a claim",
"A generation",
"I can do it",
"i it",
"I think",
"I used to",
"adventure",
"honeydew",
"can't speak",
"do you have",
"you",
"Facebook",
"many",
"whee",
"Sick",
"Law",
"association",
"fried noodles",
"mine",
"my child",
"noisy",
"pray",
"shit",
"should i",
"thanks",
"It's ok",
"That's fine",
"how do you 2",
"We will",
"where",
"wtf",
"your me"
]

elif args.data.lower() == 'spouse':
    prompts_list =  ["Are [PERSON1] and [PERSON2] family members?",
    "Is [PERSON1] said to be a family member?",
    "Is [PERSON2] said to be a family member?",
    "Are [PERSON1] and [PERSON2] dating?",
    "Are [PERSON1] and [PERSON2] co-workers?",
    "Is there any mention of 'spouse' between the entities [PERSON1] and [PERSON2]?",
    "Is there any mention of 'spouse' before the entity [PERSON1]? ",
    "Is there any mention of 'spouse' before the entity [PERSON2]",
    "Do [PERSON1] and [PERSON2] have the same last name?",
    "Did [PERSON1] and [PERSON2] get married?",
    "Are [PERSON1] and [PERSON2] married?"
    ]
    prompt_augmented = [
    "Are [PERSON1] and [PERSON2] family members?",
    "Is [PERSON1] called a family member?",
    "Is [PERSON2] called a family member?",
    "Are [PERSON1] and [PERSON2] dating?",
    "Are [PERSON1] and [PERSON2] colleagues?",
    "Is there any mention of 'spouse' between entities [PERSON1] and [PERSON2]?",
    "Is there any mention of 'spouse before entity [PERSON1]?",
    "Is there any mention of 'spouse' before entity [PERSON2]",
    "Do [PERSON1] and [PERSON2] have the same last name?",
    "Are [PERSON1] and [PERSON2] married?",
    "Are [PERSON1] and [PERSON2] married?"
    ]

if args.augment:
    prompts_list = prompts_list + prompt_augmented

with torch.no_grad():
    # Tokenizes the prompts
    input_mask = tokenizer(prompts_list, padding='longest',truncation=True, return_tensors="pt").to("cpu").attention_mask
    tokenized_prompt = tokenizer(prompts_list, padding='longest',truncation=True, return_tensors="pt").to("cuda:0").input_ids

    # Gets hidden states
    encoder_hidden_states = model.base_model.encoder(tokenized_prompt, return_dict=True).last_hidden_state
    encoder_hidden_states = encoder_hidden_states.cpu()
    active_sequence_output = torch.einsum("ijk,ij->ijk",[encoder_hidden_states, input_mask])
    encoder_hidden_states = active_sequence_output.sum(1) / input_mask.sum(dim=1).view(input_mask.size(0),1)
    normed_encoder_hidden_states = encoder_hidden_states/ encoder_hidden_states.norm(
        dim=-1, keepdim=True
    )
    name = ['Ori','Aug']


if not os.path.exists(args.embedding_path):
    os.makedirs(args.embedding_path)
    # Save the hidden states and creates similarity matrix
torch.save(normed_encoder_hidden_states.cpu(), os.path.join(args.embedding_path, f'{name[args.augment]}_{args.data.lower()}.pt'))




