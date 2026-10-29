# AI-Based Document Metadata Extraction System

An AI-powered document processing system that extracts important metadata from rental agreements and other documents using OCR and Large Language Models (LLMs). The application supports Word documents and scanned images, provides a web interface, and includes evaluation scripts for measuring extraction performance.

## 1. Project Overview

Manually extracting information from agreements can be time-consuming and error-prone. This project automates the process by accepting a document, extracting its text, and identifying important metadata using an LLM.

### Key Features

- Supports `.docx`, `.png`, `.jpg`, and `.jpeg` files.
- Extracts text from Word documents using `python-docx`.
- Uses EasyOCR to extract text from scanned images.
- Uses Ollama with the Qwen2.5 7B language model for metadata extraction.
- Extracts six metadata fields:
  - Agreement Value
  - Agreement Start Date
  - Agreement End Date
  - Renewal Notice Period (Days)
  - Party One
  - Party Two
- Provides a Streamlit interface for uploading and processing documents.
- Provides a FastAPI backend with API documentation.
- Includes dataset inspection, evaluation, and automated tests.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI |
| Language Model | Ollama — Qwen2.5 7B |
| OCR | EasyOCR |
| Word Document Processing | python-docx |
| Data Processing | Pandas |
| API Validation | Pydantic |
| Evaluation | Python scripts and CSV datasets |

## 3. Project Structure

```text
ai-document-metadata-extraction/
├── app/                  # Backend and extraction logic
├── frontend/             # Streamlit user interface
├── data/                 # Training and testing datasets
│   ├── train.csv
│   └── test.csv
├── scripts/              # Evaluation and dataset inspection
├── tests/                # Automated tests
├── outputs/               # Generated outputs, if configured
├── .env.example           # Example environment configuration
├── .gitignore
├── requirements.txt
├── pytest.ini
├── assignment-details.pdf
└── README.md
```

## 4. Prerequisites

Install the following before running the project:

1. **Python 3.10 or later:** https://www.python.org/downloads/
2. **Git:** https://git-scm.com/downloads
3. **Ollama:** https://ollama.com/

During Python installation, enable **Add Python to PATH** if the installer offers that option.

Verify the installations in PowerShell:

```powershell
python --version
git --version
ollama --version
```

## 5. Installation and Setup (Windows)

### Step 1: Clone the repository

Open PowerShell and run:

```powershell
git clone https://github.com/keerthanakkki/ai-document-metadata-extraction.git
cd ai-document-metadata-extraction
```

All remaining commands should be executed from this project directory unless stated otherwise.

### Step 2: Create a virtual environment

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation because of the execution policy, use Command Prompt and run:

```cmd
venv\Scripts\activate.bat
```

### Step 3: Install Python dependencies

With the virtual environment activated, run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Download the language model

Download the default model:

```powershell
ollama pull qwen2.5:7b
```

Verify that the model is available:

```powershell
ollama list
```

**Note:** The Qwen2.5 7B model requires sufficient RAM and disk space. Processing speed depends on the computer's hardware. The first run may take longer while the model loads.

If you use a different model, make sure its name matches the model configured in the application. Alternative models may produce different extraction results.

### Step 5: Configure environment variables

The project includes an example configuration file named `.env.example`.

To create a local environment file, run:

```powershell
Copy-Item .env.example .env
```

Edit `.env` only if you need to change the model or other supported configuration settings.

If the project works with the default configuration, no additional changes may be necessary. Keep local secrets and machine-specific settings out of GitHub.

## 6. Running the Application

Keep the virtual environment activated in each terminal where you run the project.

### Terminal 1: Start Ollama

Ollama normally runs in the background after installation. If it is already running, skip this step.

Otherwise, open a terminal and run:

```powershell
ollama serve
```

Leave the terminal running.

### Terminal 2: Start the FastAPI backend

Open a second PowerShell terminal:

```powershell
cd ai-document-metadata-extraction
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If the repository was cloned into a different directory, navigate to that directory instead.

The backend should be available at:

- API: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs

### Terminal 3: Start the Streamlit frontend

Open a third PowerShell terminal:

```powershell
cd ai-document-metadata-extraction
.\venv\Scripts\Activate.ps1
python -m streamlit run frontend/streamlit_app.py
```

Open the URL displayed in the terminal, usually:

**http://localhost:8501**

You can now use the web interface to upload a supported document and view the extracted metadata.

## 7. API Endpoints

The FastAPI backend exposes the following endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check backend health |
| `POST /extract` | Extract metadata from a document |
| `POST /extract/batch` | Process multiple documents |

Visit http://127.0.0.1:8000/docs to inspect the available request formats, parameters, and responses.

## 8. Dataset and Evaluation

The project includes CSV datasets for training examples and evaluation.

- `data/train.csv` contains training examples that can be used as extraction context.
- `data/test.csv` contains examples used for evaluation.

The test dataset should not be used as input context during prediction, as this could make evaluation results misleading.

### Inspect the dataset

From the project root, run:

```powershell
python scripts/inspect_dataset.py
```

### Run evaluation

```powershell
python scripts/evaluate.py
```

The evaluation script uses the project's configured dataset and extraction pipeline. Check the generated output and terminal results to review the predictions and evaluation metrics.

**Evaluation note:** Training and test data should contain separate documents. If the same document appears in both datasets, the evaluation may overestimate generalization. Check for overlapping document names before interpreting the results.

## 9. Run Automated Tests

From the project root, with the virtual environment activated:

```powershell
python -m pytest tests/ -v
```

Review the test output to confirm which tests passed or failed. Test results may depend on the current code, installed dependencies, and test configuration.

## 10. Troubleshooting

### Error: `ModuleNotFoundError: No module named 'app'`

Make sure you are in the repository root, where the `app` directory is located, before starting the backend.

### Error: Ollama connection refused

Check that Ollama is installed and running. Start it with `ollama serve` if necessary, and verify that the configured model is available using `ollama list`.

### Error: Model not found

Pull the configured model:

```powershell
ollama pull qwen2.5:7b
```

If your application uses a different model name, download that model instead.

### Error: Python module not found

Activate the virtual environment and reinstall the dependencies:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Extraction is slow

OCR and local LLM inference can take time, particularly on computers without a suitable GPU. Allow the model to load and finish processing before retrying. Performance depends on document size, hardware, and model configuration.

### Frontend cannot connect to the backend

Make sure the FastAPI backend is running and that the frontend is configured to use the correct backend address.

## 11. Limitations

- Extraction accuracy depends on document quality, OCR quality, model behavior, and document formatting.
- Scanned images may take longer to process than text-based Word documents.
- Local LLM inference requires sufficient system resources.
- Extracted metadata should be reviewed before being used for legal, financial, or other important decisions.
- Evaluation results depend on the size and quality of the available dataset.

## 12. Future Enhancements

- Support for additional document formats, including PDF.
- Improved confidence scoring and evidence highlighting.
- Better handling of conflicting values and ambiguous dates.
- Larger and more diverse evaluation datasets.
- Performance optimization for local inference.

## 13. Author

**Keerthana K.**



