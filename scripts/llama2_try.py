from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="ggml-model-q4_0.gguf",
    temperature=0.0,
    top_p=1,
    n_ctx=6000,
    callback_manager=callback_manager,
    verbose=True,
)

prompt = PromptTemplate.from_template(
    "who wrote {book}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
answer = chain.run("innovator's dilemma")
print(answer)

prompt = PromptTemplate.from_template(
    "What is {what}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
answer = chain.run("llama2")
print(answer)