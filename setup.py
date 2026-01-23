import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ellmer",
    version="0.0.1",
    description="Explain Large Language Models for Entity Resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['ellmer', 'ellmer.prompts', 'ellmer.post_hoc'],
    install_requires=[
          'pandas',
          'numpy',
          'certa',
          'langchain',
          'llama-cpp-python',
          'bitsandbytes',
          'transformers',
          'peft',
          'accelerate',
          'einops',
          'safetensors',
          'torch',
          'openai',
          'nltk',
          'langchain',
          'langchain_huggingface'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)
