# Introduction Tutorial for Large Language Model (Natural Language Processing)
A large language model (LLM) is a type of computational model designed for natural language processing tasks such as language generation. As language models, LLMs acquire these abilities by learning statistical relationships from vast amounts of text during a self-supervised and semi-supervised training process. The largest and most capable LLMs are artificial neural networks built with a decoder-only transformer-based architecture, enabling efficient processing and generation of large-scale text data. Modern models can be fine-tuned for specific tasks, or be guided by prompt engineering. These models acquire predictive power regarding syntax, semantics, and ontologies inherent in human language corpora, but they also inherit inaccuracies and biases present in the data on which they are trained.
**Read more on on Large Language Model [here](https://en.wikipedia.org/wiki/Large_language_model)**

## Steps
* **1. Clone or download this repo**
* **2. download the models [LLM .pth FILES](https://www.kaggle.com/models/njadnissi/llm_pth_models) and paste into `4_training_gpt` folder** as shown below:
```
.
├── 0_installation
│   └── 1_Installation_Guide_for_LLM_Training.pdf
├── 1_environment_setup
│   └── testing_env.ipynb
├── 2_text_processing
│   ├── images
│   ├── text_processing.ipynb
│   └── the-verdict.txt
├── 3_model
│   ├── images
│   ├── llm_implementation.ipynb
│   ├── __pycache__
│   └── utils.py
├── 4_training_gpt
│   ├── gpt2    <---------------------------------\ 
│   ├── images                                     \ Unizpped gpt2.zip, model.zip,
│   ├── model_and_optimizer.pth <------------------/ and model_and_optimizer.zip 
│   ├── model.pth   <-----------------------------/
│   └── training_gpt
├── 5_notebooks
│   ├── 1_testing_env.ipynb
│   ├── 2_text_processing.ipynb
│   ├── 3_llm_implementation.ipynb
│   ├── 4_train_model.ipynb
│   ├── 5_model_loading.ipynb
│   └── llm_temperature.py
├── environment.yml
├── PDF_QA_System
│   ├── artifacts   <------------------------------\ 
│   │   ├── pdf1_qa_index.pkl   <--- model built    \
│   │   ├── ...                      from 1 pdf only \ Unziped file:
│   │   ├── pdf18_qa_index.pkl  <--- ... with 18 pdf / 32_PDA_QA.zip 
│   │   ├── pdf32_qa_index.pkl  <--- ... with 32 pdf/   
│   ├── knowledge   <------------------------------/
│   │   ├── CSharpNotesForProfessionals.pdf <------\
│   │   ├── CSSNotesForProfessionals.pdf            \ Computer-Science knowledge base
│   │   ├── ...                                     /
│   │   ├── HTML5NotesForProfessionals.pdf         /
│   │   ├── JavaNotesForProfessionals.pdf   <-----/
│   ├── model_cache    <-------------------------- Sentence downloaded Model
│   └── ...
│   └── main.py <--------------------------------- Run the application CLI/Web
├── README.md
└── requirements.txt
```
* **3. Install an Integrated Development Environment** [PyCharm](https://www.jetbrains.com/help/pycharm/installation-guide.html), [VSCode](https://code.visualstudio.com/docs) or [any other](https://www.google.com/search?client=ubuntu-sn&channel=fs&q=python+IDEs) you prefer, then launch the IDE.
* **4.Install Virtual Environments Manager**  [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), [MiniConda](https://docs.anaconda.com/miniconda/install), [Anaconda](https://docs.anaconda.com/anaconda/install/) or you can just use the python inbuilt [venv](https://docs.python.org/3/tutorial/venv.html).
* **5. Move into project directory In your Terminal(inside or outside the IDE) and Run:**
	
	With **Conda**:
	```bash
	### Create Virtual Environment and Install Packages
	conda env create -f environment.yml
	### Activate:
	conda activate llm_env
	```
	
	With **venv (pip)**:
	```bash
	### Create Virtual Environment
	python3 -m venv llm_env

	### Activate:
	for linux/macos: source <path-to-llm_env>/bin/activate
	for windows Bash: cd <path-to-llm_env>\Scripts && activate

	### Install pip
	python install --upgrade pip

	###Confirm installation and pip path
	for linux/mac: which pip
	for windows: where pip
	
	### Install packages
	make sure you are in the project directory: cd <path-to-llm_env>
	pip install -r requirements.txt
	```

- For the Command Line Interface (CLI), use:
	```
		python main cli
	```
- For the web-app, `web` is optional run:
	```
	streamlit run main.py
	```

