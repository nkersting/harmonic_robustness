import os
from anthropic import Anthropic


client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

message = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "what is product of 123432 and 4234",
        }
    ],
    model="claude-2.1",
    temperature=0
)
print(message.content[0].text)
