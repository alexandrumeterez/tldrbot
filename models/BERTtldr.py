from transformers import (BertConfig, BertTokenizer, BertForQuestionAnswering)
from utils.utils import *
from utils.convert import squad_examples_to_features
from utils.answer import *
import torch


def to_list(tensor):
    return tensor.detach().cpu().tolist()


class BERTtldr:
    def __init__(self, model_path):
        self.model, self.tokenizer = self.load_model(model_path)

    def load_model(self, model_path: str, do_lower_case=False):
        config = BertConfig.from_pretrained(model_path + "/bert_config.json")
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
        model = BertForQuestionAnswering.from_pretrained(model_path, from_tf=False, config=config)
        return model, tokenizer

    def predict(self, context, question):
        example = input_to_squad_example(context, question)
        features = squad_examples_to_features(example, self.tokenizer, 400, 128, 64)

        input_example = {'input_ids': torch.Tensor([features[0].input_ids]).long(),
                         'attention_mask': torch.Tensor([features[0].input_mask]).long(),
                         'token_type_ids': torch.Tensor([features[0].segment_ids]).long()}
        outputs = self.model(**input_example)
        result = RawResult(unique_id=1000000000, start_logits=to_list(outputs[0][0]), end_logits=to_list(outputs[1][0]))

        answer = get_answer(example, features, [result], 20, 30, False)
        return answer
