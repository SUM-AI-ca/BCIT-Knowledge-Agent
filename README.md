# BCIT Academic Advisor Chatbot

RAG-based chatbot for BCIT CST students.

## Requirements

- Python 3.10+
- Node.js 20+
- CUDA 12.1+ compatible GPU (minimum 4GB VRAM)
- Linux environment (WSL2 supported)

> ⚠️ The vectorstore uses `faiss-gpu==1.7.2` which is **Linux-only**.

## Installation

### 1. Backend Setup

```bash
cd server/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd server/frontend
npm install
npm run build
```

## Running the Application

### Terminal 1 - Backend

```bash
cd server/backend
source venv/bin/activate
python3 server.py
```

### Terminal 2 - Frontend

```bash
cd server/frontend
npm run dev
```

## Access

- URL: http://localhost:5174/
- Password: `bcitcstaiml`

## WSL Installation (Windows Users)

1. Open PowerShell as Administrator:
```powershell
wsl --install
```

2. Reboot your computer

3. Open Ubuntu and set up username/password

4. Install dependencies:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv -y

# Node.js via nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 20
```

5. Navigate to project:
```bash
cd /mnt/c/Users/YOUR_USERNAME/path/to/server
```