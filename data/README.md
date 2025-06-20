# Patent Data Directory

This directory contains patent data files used by the Patent Research Platform.

## Files

### `sample_patent_data.xlsx`
A sample dataset containing 100 fictional patent records for demonstration and testing purposes. This file includes all the necessary columns and data structure expected by the system.

**Important**: This is sample data with fictional patent information created for public repository purposes. It demonstrates the data structure and format but does not contain real patent data.

### Data Structure

The Excel file contains the following columns:

- `publication_number`: Patent publication number (e.g., US1234567A1)
- `publication_country`: Country code (US, EP, JP, etc.)
- `publication_kind`: Publication kind (A1, B1, B2)
- `publication_date`: Publication date (YYYY-MM-DD format)
- `ipc`: International Patent Classification codes (JSON array)
- `title_en`: Patent title in English
- `abstract_text`: Patent abstract text
- `sdg_number`: Sustainable Development Goal numbers (JSON array)
- `analysis_explanation`: Analysis and explanation (JSON object)
- `applicant_names`: Names of patent applicants (JSON array)
- `applicant_countries`: Countries of applicants (JSON array)
- `applicant_count`: Number of applicants
- `inventor_names`: Names of inventors (JSON array)
- `inventor_countries`: Countries of inventors (JSON array)
- `inventor_count`: Number of inventors
- `ipc_tech_field`: Technology fields (JSON array)
- `ipc_technologies`: Technologies (JSON array)
- `sdg_technology_fields`: SDG technology fields (JSON array)
- `analysis_potential_beneficiaries`: Potential beneficiaries (JSON array)
- `designated_states_*`: Patent designation information (JSON arrays)
- `prior_art`: Prior art references (JSON array)
- `reference`: References (JSON array)
- `parent`: Parent application information (JSON array)
- `pct_publication_number`: PCT publication number
- `parent_publication_number`: Parent publication number

## Usage

### For Development and Testing
Use the sample dataset to:
- Test the application functionality
- Develop new features
- Demonstrate the system capabilities
- Validate data processing pipelines

### For Production
Replace the sample data with your actual patent dataset following the same structure and format.

## Data Format Notes

- JSON fields: Many columns contain JSON-formatted data to support multiple values
- Date format: All dates should be in YYYY-MM-DD format
- Country codes: Use standard 2-letter country codes (ISO 3166-1 alpha-2)
- IPC codes: Follow standard International Patent Classification format

## Database Integration

The data is imported into the SQLite database using the models defined in `database/models.py`. The system expects:

1. Patent records in the main patents table
2. Text chunks for semantic search in the patent_chunks table
3. Proper relationships between patents and chunks

## Generating New Sample Data

To generate a new sample dataset, run:

```bash
python generate_sample_data.py
```

This will create a new `sample_patent_data.xlsx` file with 100 fictional patent records.

## Security Note

**Never commit real patent data to public repositories.** Always use sample or anonymized data for public code repositories to protect intellectual property and sensitive information.