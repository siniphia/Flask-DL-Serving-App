# -*- coding: utf-8 -*-
import torch
import transformers


MODEL_TYPE = 'AutoModelForQuestionAnswering'
MODEL_DIR = 'ainize/klue-bert-base-mrc'
TOKENIZER_TYPE = 'AutoTokenizer'
TOKENIZER_DIR = 'ainize/klue-bert-base-mrc'


def get_model(model_type=MODEL_TYPE, model_dir=MODEL_DIR):
    return getattr(transformers, model_type).from_pretrained(model_dir)


def get_tknzr(tknzr_type=TOKENIZER_TYPE, tknzr_dir=TOKENIZER_DIR):
    return getattr(transformers, tknzr_type).from_pretrained(tknzr_dir)


def run_model_qa(context, question):

    # instantiate model and tokenizer
    tknzr = get_tknzr()
    model = get_model()

    # tokenize given strings and infer logits
    inputs = tknzr(question, context, return_tensors='pt')
    input_ids = inputs['input_ids'].tolist()[0]
    outputs = model(**inputs)

    # get start and end index
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    # slice and convert original string
    answer = tknzr.convert_tokens_to_string(tknzr.convert_ids_to_tokens(input_ids[start_idx:end_idx]))

    return answer


# sqaud examples
# c = '여우는 갯과에 속하는 소형 포식동물이다. 크기의 범위는 몸길이 24~140cm, 어깨높이 15~55cm, 몸무게 0.7~17kg이다. 붉은여우는 여우 중에서 가장 크고 가장 흔한 여우이며 몸길이 90cm, 꼬리길이 60cm, 어깨높이 55cm, 체중 10kg으로 고양이보다 조금 크고 살쾡이, 중소형견[2] 등과 비슷한 크기이다. 지역에 따라 차이가 큰데, 중부 유럽에 서식하는 개체들은 8kg 내외이며 미국이나 일본에 사는 종은 5~6kg이다. 또한 흔히 여우라고 하면 사람들이 지칭하는 녀석이 바로 이 붉은여우이며 수컷이 암컷보다 미세하게 크다.'
# q = '여우의 어깨높이는?'
# a = run_squad_model(MODEL_TYPE, MODEL_DIR, TOKENIZER_TYPE, TOKENIZER_DIR, c, q)
# print(a)

# pipeline usage
# from transformers import pipeline
# qa = pipeline('question-answering')
# result = qa(question=question, context=context)
# print(result)
