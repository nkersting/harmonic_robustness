# Gamma Hallucination Metric Local API

Herein is code to power a UI for measuring the gamma of an LLM response, as described in our paper "Harmonic LLMs are Trustworthy" (https://arxiv.org/abs/2404.19708).
After you clone this repository, you can launch a local gamma-testing UI (running via Flask) for various models.
You can add more OpenAI or Anthropic models to the list by editing the radio selectors in templates/index.html in the obvious way; for other models, you may need to define a new module interface similar to ../GPT4Tester.py and reference that in the app.py . 

## Setup

1. python3 -m venv venv
2. source venv/bin/activate
3. pip install flask requests numpy anthropic openai
4. python app.py
5. Access UI at http://127.0.0.1:5000/

You may then input your API keys for the desired models in the UI entry forms.

## Usage

After Setup you should see the following UI:
<p align="center">
  <picture>
    <img alt="Hallucination UI" src="https://www.quantumrepoire.com/ui-sample.png" width="2000" height="924" style="max-width: 100%;">
  </picture>
