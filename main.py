from flask import Flask

import os
from dotenv import load_dotenv, find_dotenv

from gpt import GPT, Example, set_openai_key


app = Flask(__name__)

load_dotenv(find_dotenv(), override=True)
set_openai_key(os.getenv("GPT_SECRET_KEY", ""))

gpt = GPT(engine="davinci", temperature=0.5, max_tokens=100)


gpt.add_example(Example("Two plus two equals four", "2 + 2 = 4"))
gpt.add_example(Example("The integral from zero to infinity", "\\int_0^{\\infty}"))
gpt.add_example(
    Example(
        "The gradient of x squared plus two times x with respect to x",
        "\\nabla_x x^2 + 2x",
    )
)
gpt.add_example(Example("The log of two times x", "\\log{2x}"))
gpt.add_example(
    Example("x squared plus y squared plus equals z squared", "x^2 + y^2 = z^2")
)


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/gpt3")
def gpt3():
    prompt = "integral from a to b of f of x"
    return gpt.get_top_reply(prompt)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
