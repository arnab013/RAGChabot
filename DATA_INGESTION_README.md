# Data Ingestion System

This system automatically detects and imports new Excel/CSV files from the data folder into the RAG chatbot database.

## Features

- **Automatic File Detection**: Scans the data folder for Excel (.xlsx, .xls) and CSV (.csv) files
- **Duplicate Prevention**: Tracks imported files using SHA256 hashes to avoid re-importing unchanged files
- **Incremental Updates**: Only imports new or modified files
- **Data Validation**: Normalizes column names and validates data before insertion
- **Text Chunking**: Automatically creates searchable text chunks from patent data
- **Progress Tracking**: Shows detailed progress during import process

## Quick Start

### 1. Test Current Setup
```bash
python test_data_import.py
```

### 2. Import New Files
```bash
python data_ingestion.py
```

### 3. View Database Summary
```bash
python data_ingestion.py --summary
```

### 4. List Imported Files
```bash
python data_ingestion.py --list-files
```

### 5. Force Re-import All Files
```bash
python data_ingestion.py --force-reimport
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--data-folder PATH` | Specify custom data folder path (default: `data`) |
| `--force-reimport` | Re-import all files, even if already imported |
| `--list-files` | Show all previously imported files with metadata |
| `--summary` | Display current database statistics |

## File Format Requirements

The system supports Excel and CSV files with patent data. Column names are automatically normalized to match the database schema.

### Required Columns
- `publication_number` (or variations like `pub_number`, `publication_no`)

### Supported Columns
- `publication_country`, `publication_kind`, `publication_date`
- `title_en` (or `title`, `patent_title`)
- `abstract_text` (or `abstract`, `summary`)
- `inventor_names` (or `inventors`, `inventor`)
- `inventor_countries` (or `inventor_country`)
- `applicant_names` (or `applicants`, `applicant`)
- `applicant_countries` (or `applicant_country`)
- `sdg_number` (or `sdg`, `sdg_numbers`)
- `ipc` (or `ipc_classification`, `ipc_class`)
- And many more...

### Data Formats

**Array Fields**: Can be provided as:
- JSON arrays: `["value1", "value2", "value3"]`
- Semicolon-separated: `value1; value2; value3`
- Single values: `value1`

**Date Fields**: Automatically parsed from various formats

## File Tracking

The system maintains a `data_source_files` table that tracks:
- File name and path
- SHA256 hash for change detection
- Import date and time
- Number of records imported
- File size

## Database Schema

### Patents Table
Stores main patent information with fields matching the data columns.

### Patent Chunks Table
Stores searchable text chunks created from patent data for RAG functionality.

### Data Source Files Table
Tracks imported files to prevent duplicates and enable incremental updates.

## Example Usage

```python
from data_ingestion import DataIngestionManager

# Initialize manager
manager = DataIngestionManager("data")

# Check for new files
new_files = manager.identify_new_files()
print(f"Found {len(new_files)} new files")

# Import all new files
results = manager.run_import()
print(f"Imported {results['total_records_imported']} records")

# Get database summary
summary = manager.get_database_summary()
print(f"Database now contains {summary['total_patents']} patents")
```

## Troubleshooting

### Common Issues

1. **"No files found"**: Ensure Excel/CSV files are in the data folder
2. **"publication_number required"**: Check that your data has a publication number column
3. **"Permission denied"**: Make sure the data folder and database file are writable
4. **Import errors**: Check the console output for specific error messages

### Log Output

The system provides detailed logging during import:
- File detection and validation
- Column normalization
- Data cleaning steps
- Import progress
- Error details

### Data Validation

Before import, the system:
- Removes rows with invalid publication numbers
- Removes duplicate publication numbers
- Converts date formats
- Normalizes array fields to JSON
- Creates searchable text chunks

## Integration

The data ingestion system integrates seamlessly with the RAG chatbot:
- Imported patents are immediately available for search
- Text chunks enable semantic search functionality
- Database schema matches the query system expectations

## Future Enhancements

- **Automatic file monitoring**: Watch folder for new files
- **Data quality reporting**: Detailed validation reports
- **Backup and restore**: Database backup before major imports
- **Custom column mapping**: User-defined column mappings
- **Batch processing**: Parallel processing for large files
