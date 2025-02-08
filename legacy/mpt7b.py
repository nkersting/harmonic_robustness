import transformers
from transformers import pipeline
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')


model = transformers.AutoModelForCausalLM.from_pretrained(
  'mosaicml/mpt-7b-chat',
  trust_remote_code=True
)

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

print(
    pipe('Here is a recipe for vegan banana bread:\n',
         max_new_tokens=100,
         do_sample=True,
         use_cache=True))
