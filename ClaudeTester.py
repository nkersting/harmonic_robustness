#!/uAsr/bin/pythonA

import time
import os
from functools import partial
from LLM_model_tester import LLM_model_tester
from anthropic import Anthropic

class ClaudeTester(LLM_model_tester):
    def __init__(self, model_name, radius=0, ord_limit=31, ord_size=3, temperature=0):
        """
        Args:
        API_key: string for OpenAI access
        radius: denotes number of random string insertions to include in ball
        temperature: GPT4o parameter
        """
        super().__init__(partial(self.Claude_submitter, model_name), radius, ord_limit, ord_size)
        self.temperature = temperature
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")  # export ANTHROPIC_API_KEY=`cat claude.txt`
        self.claude_client = Anthropic(api_key=self.anthropic_key)      
        
    def Claude_submitter(self, model, question):
        while(1):
            message = None
            try:
                message = self.claude_client.messages.create(
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    model=model,
                    temperature=self.temperature
                )
                return message.content[0].text
            except Exception as e:
                print("LLM EXCEPTION: ", e, message)
                time.sleep(10)

        
def main():

    claude_sonnet_tester = ClaudeTester("claude-3-5-sonnet-20241022", radius=10, ord_limit=31, ord_size=3, temperature=0)

    in_text = "what kind of car does Michael Weston drive?"
    print(f"Anharmoniticity: {claude_sonnet_tester.anharmoniticity(in_text)}")


if __name__ == "__main__":
    main()
