# MultiModal RAG

A Multimodal Retrieval-Augmented Generation (RAG) system that enables intelligent querying of PDF documents containing text, images, and tables. This application extracts, processes, and indexes multimodal content to provide context-aware answers to user queries.

## Features

- **PDF Processing**: Extract and process text, images, and tables from PDF documents
- **Multimodal Content Extraction**: Automatically identifies and extracts different content types from documents
- **Chat Interface**: Interactive chat interface for querying processed documents
- **Persistent Memory**: Chat history stored in SQLite database for conversation continuity
- **Deployment Ready**: Configured for deployment on Render.com with build scripts

## Project Structure

```
MultiModal_RAG/
├── main.py                   # Main application entry point
├── chattingh.py             # Chat interface implementation
├── requirements.txt         # Python dependencies
├── chat_memory.db          # SQLite database for chat history
├── extracted_images/       # Directory for extracted images from PDFs
├── __pycache__/           # Python cache files
├── render.yaml            # Render.com deployment configuration
├── build.sh              # Build script for deployment
├── runtime.txt           # Python runtime version specification
└── aptfile              # System dependencies for deployment
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/vardan201/MultiModal_RAG.git
cd MultiModal_RAG
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add necessary API keys:
```
OPENAI_API_KEY=your_api_key_here
# Add other required API keys
```

## Usage

### Running Locally

1. Start the application:
```bash
python main.py
```

2. For the chat interface:
```bash
python chattingh.py
```

3. Upload PDF documents through the interface and start querying

### Querying Documents

- Upload a PDF document containing text, images, and tables
- The system will automatically extract and process the content
- Ask questions about the document content
- Get answers grounded in the extracted information with relevant context

## Deployment

This project is configured for deployment on Render.com:

1. The `render.yaml` file contains deployment configuration
2. The `build.sh` script handles build process
3. System dependencies are specified in `aptfile`

To deploy:
1. Push your code to GitHub
2. Connect your repository to Render.com
3. Render will automatically use the configuration files to deploy

## Technologies Used

- **Python**: Core programming language
- **LangChain**: Framework for building RAG applications
- **PDF Processing**: Libraries for extracting content from PDFs
- **Vector Database**: For storing and retrieving embeddings
- **Large Language Models**: For generating contextual responses
- **SQLite**: For storing chat history

## Features in Detail

### PDF Content Extraction
- Extracts text blocks from PDF documents
- Identifies and extracts images
- Parses tables and structured data
- Maintains document structure and relationships

### Multimodal RAG Pipeline
- Generates embeddings for text and visual content
- Stores multimodal embeddings in vector database
- Retrieves relevant context based on user queries
- Synthesizes answers using retrieved multimodal information

### Chat Memory
- Maintains conversation history in SQLite database
- Provides context-aware responses based on chat history
- Supports multi-turn conversations

## Sample Use Cases

- **Research Papers**: Query academic papers with complex diagrams and tables
- **Technical Documentation**: Extract information from manuals with images and schematics
- **Reports**: Analyze business reports with charts and data tables
- **Educational Materials**: Study from textbooks with illustrations and examples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source. Please check the repository for license details.

## Acknowledgments

- Built with modern RAG techniques and multimodal processing
- Inspired by research in multimodal document understanding
- Thanks to the open-source community for amazing tools and libraries

---

**Note**: Make sure to set up your API keys and environment variables before running the application. Refer to the `.env.example` file (if available) for required configurations.
