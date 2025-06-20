# Contributing to Patent Research Platform

We welcome contributions to the Patent Research Platform! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork the Repository**
   ```bash
   git clone https://github.com/arnab013/RAGChabot.git
   cd RAGChabot
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Copy `.env.example` to `.env` and fill in your API keys and configuration.

4. **Install Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   ```

## Development Guidelines

### Code Style

- **Python**: Follow PEP 8 style guidelines
- **JavaScript/React**: Use consistent formatting with Prettier
- **Documentation**: Write clear, concise docstrings and comments
- **Testing**: Include tests for new functionality

### Commit Messages

Use clear, descriptive commit messages:
```
feat: add new patent search functionality
fix: resolve query classification issue
docs: update API documentation
refactor: improve search performance
```

### Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   python -m pytest tests/
   cd frontend && npm test
   ```

4. **Submit Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure all tests pass

## Types of Contributions

### Bug Reports
- Use the issue template
- Include steps to reproduce
- Provide system information
- Include error messages and logs

### Feature Requests
- Clearly describe the proposed feature
- Explain the use case and benefits
- Consider implementation challenges

### Code Contributions
- **Backend**: API endpoints, query processing, data analysis
- **Frontend**: UI components, user experience improvements
- **Documentation**: README updates, API documentation
- **Testing**: Unit tests, integration tests

### Documentation
- API documentation improvements
- Installation and setup guides
- User guides and tutorials
- Code comments and docstrings

## Project Structure

```
├── src/                    # Backend Python code
│   ├── api.py             # Main Flask application
│   ├── queries/           # Query processing modules
│   └── ...
├── frontend/              # React frontend
│   ├── src/components/    # React components
│   └── ...
├── database/              # Database models and config
├── tests/                 # Test files
└── docs/                  # Documentation
```

## API Guidelines

### Adding New Endpoints
- Follow RESTful conventions
- Use appropriate HTTP methods
- Include proper error handling
- Document with clear examples

### Query Processing
- Extend the query classification system
- Add appropriate response handlers
- Include comprehensive error handling
- Write tests for new query types

## Frontend Guidelines

### Components
- Create reusable, modular components
- Follow React best practices
- Include PropTypes or TypeScript definitions
- Write component tests

### State Management
- Use React hooks appropriately
- Manage state efficiently
- Consider performance implications

## Testing

### Backend Testing
```bash
python -m pytest tests/ -v
```

### Frontend Testing
```bash
cd frontend
npm test
```

### Integration Testing
Test the complete workflow from frontend to backend.

## Code Review Process

1. **Self-Review**: Review your own code before submitting
2. **Peer Review**: At least one other contributor reviews the code
3. **Testing**: Ensure all tests pass
4. **Documentation**: Verify documentation is updated

## Getting Help

- **Issues**: Check existing issues for similar problems
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Refer to project documentation
- **Contact**: Reach out to maintainers for guidance

## Recognition

Contributors will be recognized in:
- README.md contributor section
- Release notes for significant contributions
- Project documentation

Thank you for contributing to the Patent Research Platform!
