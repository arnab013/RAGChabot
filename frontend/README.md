# RAGBot - Frontend

This is a modern React frontend for the RAGBot, providing a user-friendly interface to interact with the Retrieval-Augmented Generation (RAG) system.

## Features

- Clean, responsive UI built with Material UI
- Real-time chat interface
- Markdown support for bot responses
- Easy to use and visually appealing

## Getting Started

### Prerequisites

- Node.js (v14+)
- npm or yarn
- Running RAGBot backend API (Python Flask)

### Installation

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```
   or
   ```
   yarn install
   ```

3. Start the development server:
   ```
   npm start
   ```
   or
   ```
   yarn start
   ```

4. The React app will open in your browser at [http://localhost:3000](http://localhost:3000)

## Connecting to the Backend

Make sure the Python Flask backend is running at http://localhost:5000 before using the frontend.

To start the backend:

```
cd ..
python src/api.py
```

## Build for Production

To build the app for production:

```
npm run build
```
or
```
yarn build
```

This creates an optimized build in the `build` folder that can be deployed to a web server.
