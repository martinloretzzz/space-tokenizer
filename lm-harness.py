import json

import lm_eval
import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import GPT, GPTConfig
from tokenizer import HfTokenizerWrapper, SpaceTokenizer, unpack_token
from tokenizers import Tokenizer


class LMWrapper(LM):
    def __init__(self, tokenizer, model, device='cuda', hf_model=False, TRAIN_SPACE=True):
        super().__init__()
        self.device = device
        self.hf_model = hf_model
        self.TRAIN_SPACE = TRAIN_SPACE
        self.tokenizer = tokenizer
        self.model = model.to(self.device)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []

        for req in tqdm(requests, disable=disable_tqdm):
            context, continuation = req.arguments
            loss = self.calculate_loglikelihood(context, continuation)
            res.append((loss, False))

        return res

    def calculate_loglikelihood(self, context, continuation):
        input_text = context + continuation
        inputs = self.tokenizer.encode(input_text)
        inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0).to(self.device)

        continuation_ids = self.tokenizer.encode(continuation)
        continuation_ids = torch.tensor(continuation_ids, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if not self.hf_model:
                logits, _, __, ___ = self.model(inputs)
            else:
                logits = self.model(inputs).logits

            shift_logits = (logits[..., :-1, :]).contiguous()
            shift_labels = (inputs[..., 1:]).contiguous()
            if self.TRAIN_SPACE:
                shift_labels = unpack_token(shift_labels)[0]
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
            loss = loss.view(shift_labels.size())

            mask = torch.zeros_like(shift_labels)

            continuation_length = continuation_ids.size(-1)
            mask[:, -continuation_length:] = 1

            continuation_loss = (loss * mask).sum() / mask.sum()
            return -continuation_loss.item()

    def generate_until(self, requests):
        raise NotImplementedError()
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()
    
# lm_eval --model hf --model_args pretrained=openai-community/gpt2 --tasks hellaswag --device cuda:0 --batch_size 8

TRAIN_SPACE = True

vocab_size = 20000 + 257
model = GPT(GPTConfig(vocab_size=vocab_size), TRAIN_SPACE=TRAIN_SPACE)

checkpoint_path = "./model_space_20k.pt" if TRAIN_SPACE else "./model_ref_20k.pt"
model_checkpoint = torch.load(checkpoint_path, weights_only=False)["model"]
model.load_state_dict(model_checkpoint)

with open('./tokenizer-space-20k-rs.json', 'r', encoding='utf-8') as f: tokenizer_config = json.load(f)
tokenizer = SpaceTokenizer(tokenizer_config) if TRAIN_SPACE else HfTokenizerWrapper(Tokenizer.from_file("./tokenizer-ref-20k.json"))

# tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
# model = AutoModelForCausalLM.from_pretrained('gpt2-medium')

lm_obj = LMWrapper(tokenizer, model, device="cuda", hf_model=False, TRAIN_SPACE=TRAIN_SPACE)

task_manager = lm_eval.tasks.TaskManager()

results = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["arc_easy"], # "hellaswag", "piqa"
    task_manager=task_manager,
    num_fewshot=None,
    limit=None,
)

print(results["results"])

with open("eval.json", "w") as json_file:
    json.dump(results["results"], json_file, indent=4)

# eval-hf-gpt2: acc:0.2892, acc_norm: 0.3114
# eval-wrapper-gpt2-medium: acc: 39.29, acc_norm: 32.41
# eval-wrapper-gpt2: acc: 0.309, acc_norm: 0.2724
# space-20k: acc: 31.965, acc_norm: 28.30
# ref-20k: acc: 30.36, acc_norm: 27.81

# piqa:
# space-20k: acc: 0.6333, acc_norm: 0.6034
# ref-20k: acc: 0.6219, acc_norm: 0.5930

# arc_easy
# space-20k: acc: 0.5063 acc_norm: 0.4301
# ref-20k: acc: 0.4171, acc_norm: 0.3699
