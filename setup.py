import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ellmer",
    version="0.0.1",
    author="Tommaso Teofili",
    author_email="tommaso.teofili@gmail.com",
    description="Explain Large Language Models for Entity Resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url= 'https://github.com/tteofili/ellmer.git',
    packages=['ellmer', 'ellmer.prompts'],
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
          'xformers',
          'openai'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)
