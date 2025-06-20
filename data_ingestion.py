#!/usr/bin/env python3
"""
Enhanced Data Ingestion Script for RAG Chatbot
Automatically detects and imports new Excel/CSV files from the data folder.
Tracks imported files to avoid duplicates and supports incremental updates.
"""

import os
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import json
import logging
from tqdm import tqdm

# Import database components
from database.config import get_db_session, create_tables
from database.models import Patent, PatentChunk, DataSourceFile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIngestionManager:
    """Manages the ingestion of patent data from Excel/CSV files into the database"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = Path(data_folder)
        self.supported_extensions = {'.xlsx', '.xls', '.csv'}
        
        # Create tables if they don't exist
        create_tables()
        
        logger.info(f"Initialized DataIngestionManager for folder: {self.data_folder}")
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file information including hash and size"""
        return {
            'path': file_path,
            'filename': file_path.name,
            'hash': self.calculate_file_hash(file_path),
            'size': file_path.stat().st_size,
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
        }
    
    def find_data_files(self) -> List[Dict[str, Any]]:
        """Find all Excel and CSV files in the data folder"""
        files = []
        
        if not self.data_folder.exists():
            logger.error(f"Data folder does not exist: {self.data_folder}")
            return files
        
        for file_path in self.data_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    file_info = self.get_file_info(file_path)
                    files.append(file_info)
                    logger.info(f"Found data file: {file_path.name} ({file_info['size']} bytes)")
                except Exception as e:
                    logger.warning(f"Could not process file {file_path}: {e}")
        
        return files
    
    def get_imported_files(self) -> Dict[str, DataSourceFile]:
        """Get a dictionary of already imported files keyed by filename"""
        imported = {}
        
        with get_db_session() as session:
            imported_files = session.query(DataSourceFile).all()
            for file_record in imported_files:
                imported[file_record.filename] = file_record
        
        return imported
    
    def identify_new_files(self) -> List[Dict[str, Any]]:
        """Identify files that need to be imported (new or modified)"""
        all_files = self.find_data_files()
        imported_files = self.get_imported_files()
        new_files = []
        
        for file_info in all_files:
            filename = file_info['filename']
            
            if filename not in imported_files:
                # Completely new file
                logger.info(f"New file detected: {filename}")
                new_files.append(file_info)
            else:
                # Check if file has been modified
                imported_record = imported_files[filename]
                if file_info['hash'] != imported_record.file_hash:
                    logger.info(f"Modified file detected: {filename}")
                    new_files.append(file_info)
                else:
                    logger.info(f"File already imported (unchanged): {filename}")
        
        return new_files
    
    def read_data_file(self, file_path: Path) -> pd.DataFrame:
        """Read a data file (Excel or CSV) into a pandas DataFrame"""
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"Read CSV file: {file_path.name} ({len(df)} rows)")
            else:  # Excel file
                df = pd.read_excel(file_path)
                logger.info(f"Read Excel file: {file_path.name} ({len(df)} rows)")
            
            return df
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to match the database schema"""
        # Create a mapping of possible column name variations
        column_mapping = {
            # Publication information
            'publication_number': ['publication_number', 'pub_number', 'publication_no', 'pubno'],
            'publication_country': ['publication_country', 'pub_country', 'country'],
            'publication_kind': ['publication_kind', 'pub_kind', 'kind'],
            'publication_date': ['publication_date', 'pub_date', 'date'],
            
            # Technical information
            'ipc': ['ipc', 'ipc_classification', 'ipc_class'],
            'title_en': ['title_en', 'title', 'patent_title'],
            'abstract_text': ['abstract_text', 'abstract', 'summary'],
            
            # Inventors and applicants
            'inventor_names': ['inventor_names', 'inventors', 'inventor'],
            'inventor_countries': ['inventor_countries', 'inventor_country'],
            'applicant_names': ['applicant_names', 'applicants', 'applicant'],
            'applicant_countries': ['applicant_countries', 'applicant_country'],
            
            # SDG and analysis
            'sdg_number': ['sdg_number', 'sdg', 'sdg_numbers'],
            'analysis_explanation': ['analysis_explanation', 'analysis'],
            'sdg_technology_fields': ['sdg_technology_fields', 'tech_fields'],
            'analysis_potential_beneficiaries': ['analysis_potential_beneficiaries', 'beneficiaries'],
            
            # IPC technology
            'ipc_tech_field': ['ipc_tech_field', 'tech_field'],
            'ipc_technologies': ['ipc_technologies', 'technologies'],
            
            # Other fields
            'prior_art': ['prior_art', 'prior_references'],
            'reference': ['reference', 'references'],
            'parent': ['parent', 'parent_application'],
            'pct_publication_number': ['pct_publication_number', 'pct_number'],
        }
        
        # Normalize column names to lowercase for comparison
        df_columns_lower = {col.lower(): col for col in df.columns}
        new_column_names = {}
        
        for target_col, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name.lower() in df_columns_lower:
                    original_col = df_columns_lower[possible_name.lower()]
                    new_column_names[original_col] = target_col
                    break
        
        # Rename columns
        if new_column_names:
            df = df.rename(columns=new_column_names)
            logger.info(f"Renamed columns: {new_column_names}")
        
        return df
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the data before insertion"""
        # Ensure publication_number is present and unique
        if 'publication_number' not in df.columns:
            raise ValueError("publication_number column is required but not found")
        
        # Remove rows with null publication_number
        initial_count = len(df)
        df = df.dropna(subset=['publication_number'])
        df = df[df['publication_number'].str.strip() != '']
        
        if len(df) != initial_count:
            logger.warning(f"Removed {initial_count - len(df)} rows with invalid publication_number")
        
        # Remove duplicates based on publication_number
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['publication_number'], keep='first')
        
        if len(df) != before_dedup:
            logger.warning(f"Removed {before_dedup - len(df)} duplicate publication_numbers")
        
        # Convert date columns
        date_columns = ['publication_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Handle array fields (convert to JSON strings)
        array_fields = [
            'ipc', 'prior_art', 'reference', 'parent',
            'designated_states_contracting', 'designated_states_extension', 'designated_states_validation',
            'sdg_number', 'analysis_explanation', 'sdg_technology_fields', 'analysis_potential_beneficiaries',
            'ipc_tech_field', 'ipc_technologies', 'applicant_names', 'applicant_countries',
            'inventor_names', 'inventor_countries'
        ]
        
        for field in array_fields:
            if field in df.columns:
                df[field] = df[field].apply(self._convert_to_json_array)
        
        return df
    
    def _convert_to_json_array(self, value) -> Optional[str]:
        """Convert a value to a JSON array string"""
        if pd.isna(value) or value == '' or value is None:
            return None
        
        if isinstance(value, str):
            # Try to parse as JSON first
            try:
                parsed = json.loads(value)
                return json.dumps(parsed)
            except:
                # If not JSON, treat as semicolon-separated list
                if ';' in value:
                    items = [item.strip() for item in value.split(';') if item.strip()]
                    return json.dumps(items)
                else:
                    return json.dumps([value.strip()])
        elif isinstance(value, (list, tuple)):
            return json.dumps(list(value))
        else:
            return json.dumps([str(value)])
    
    def chunk_text(self, text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or len(text) <= max_length:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_length
            
            # Try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence ending
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + max_length // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + max_length // 2:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def create_patent_chunks(self, patent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create text chunks from patent data"""
        chunks = []
        
        # Define which fields to chunk
        text_fields = ['title_en', 'abstract_text', 'prior_art']
        
        for field in text_fields:
            if field in patent_data and patent_data[field]:
                text = str(patent_data[field])
                if len(text.strip()) > 50:  # Only chunk substantial text
                    field_chunks = self.chunk_text(text)
                    
                    for i, chunk_text in enumerate(field_chunks):
                        chunk_data = {
                            'publication_number': patent_data['publication_number'],
                            'chunk_text': chunk_text,
                            'chunk_index': len(chunks),
                            'embedding': None,  # Will be populated later
                            
                            # Copy relevant metadata
                            'sdg_number': patent_data.get('sdg_number'),
                            'publication_date': patent_data.get('publication_date'),
                            'ipc': patent_data.get('ipc'),
                            'ipc_tech_field': patent_data.get('ipc_tech_field'),
                            'analysis_explanation': patent_data.get('analysis_explanation'),
                            'sdg_technology_fields': patent_data.get('sdg_technology_fields'),
                            'analysis_potential_beneficiaries': patent_data.get('analysis_potential_beneficiaries'),
                            'ipc_technologies': patent_data.get('ipc_technologies'),
                        }
                        chunks.append(chunk_data)
        
        return chunks
    
    def import_dataframe(self, df: pd.DataFrame, file_info: Dict[str, Any]) -> int:
        """Import a DataFrame into the database"""
        logger.info(f"Starting import of {len(df)} records from {file_info['filename']}")
        
        imported_count = 0
        
        with get_db_session() as session:
            try:
                # Check if this file was already imported and remove old data if needed
                existing_file = session.query(DataSourceFile).filter_by(
                    filename=file_info['filename']
                ).first()
                
                if existing_file:
                    logger.info(f"Updating existing file import: {file_info['filename']}")
                    # You might want to implement logic to remove old data here
                    # For now, we'll just update the record
                
                for index, row in tqdm(df.iterrows(), total=len(df), desc="Importing patents"):
                    try:
                        # Convert row to dictionary
                        patent_data = row.to_dict()
                        
                        # Create or update patent record
                        patent = session.query(Patent).filter_by(
                            publication_number=patent_data['publication_number']
                        ).first()
                        
                        if not patent:
                            patent = Patent()
                            session.add(patent)
                        
                        # Update patent fields
                        for key, value in patent_data.items():
                            if hasattr(patent, key) and pd.notna(value):
                                setattr(patent, key, value)
                        
                        # Create chunks for this patent
                        chunks = self.create_patent_chunks(patent_data)
                        
                        # Remove existing chunks for this patent (if updating)
                        session.query(PatentChunk).filter_by(
                            publication_number=patent_data['publication_number']
                        ).delete()
                        
                        # Add new chunks
                        for chunk_data in chunks:
                            chunk = PatentChunk(**chunk_data)
                            session.add(chunk)
                        
                        imported_count += 1
                        
                        # Commit every 100 records to avoid memory issues
                        if imported_count % 100 == 0:
                            session.commit()
                            logger.info(f"Committed {imported_count} records")
                    
                    except Exception as e:
                        logger.error(f"Error importing record {index}: {e}")
                        session.rollback()
                        continue
                
                # Update or create file tracking record
                if existing_file:
                    existing_file.file_hash = file_info['hash']
                    existing_file.import_date = datetime.utcnow()
                    existing_file.records_imported = imported_count
                    existing_file.file_size = file_info['size']
                else:
                    file_record = DataSourceFile(
                        filename=file_info['filename'],
                        file_path=str(file_info['path']),
                        file_hash=file_info['hash'],
                        records_imported=imported_count,
                        file_size=file_info['size']
                    )
                    session.add(file_record)
                
                session.commit()
                logger.info(f"Successfully imported {imported_count} records from {file_info['filename']}")
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error during import: {e}")
                raise
        
        return imported_count
    
    def run_import(self, force_reimport: bool = False) -> Dict[str, Any]:
        """Run the complete import process"""
        logger.info("Starting data import process...")
        
        results = {
            'files_processed': 0,
            'total_records_imported': 0,
            'files_skipped': 0,
            'errors': []
        }
        
        try:
            # Find files that need to be imported
            if force_reimport:
                files_to_import = self.find_data_files()
                logger.info("Force reimport enabled - processing all files")
            else:
                files_to_import = self.identify_new_files()
            
            if not files_to_import:
                logger.info("No new files to import")
                return results
            
            logger.info(f"Found {len(files_to_import)} files to import")
            
            for file_info in files_to_import:
                try:
                    logger.info(f"Processing file: {file_info['filename']}")
                    
                    # Read the file
                    df = self.read_data_file(file_info['path'])
                    
                    # Normalize column names
                    df = self.normalize_column_names(df)
                    
                    # Clean and validate data
                    df = self.clean_and_validate_data(df)
                    
                    # Import into database
                    imported_count = self.import_dataframe(df, file_info)
                    
                    results['files_processed'] += 1
                    results['total_records_imported'] += imported_count
                    
                    logger.info(f"Completed processing {file_info['filename']}: {imported_count} records imported")
                    
                except Exception as e:
                    error_msg = f"Error processing {file_info['filename']}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                    continue
            
            logger.info(f"Import process completed. Files processed: {results['files_processed']}, Total records: {results['total_records_imported']}")
            
        except Exception as e:
            error_msg = f"Fatal error during import process: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def list_imported_files(self) -> List[Dict[str, Any]]:
        """List all imported files with their metadata"""
        imported_files = []
        
        with get_db_session() as session:
            files = session.query(DataSourceFile).order_by(DataSourceFile.import_date.desc()).all()
            
            for file_record in files:
                imported_files.append({
                    'filename': file_record.filename,
                    'file_path': file_record.file_path,
                    'import_date': file_record.import_date,
                    'records_imported': file_record.records_imported,
                    'file_size': file_record.file_size,
                    'file_hash': file_record.file_hash[:8] + '...'  # Show first 8 chars of hash
                })
        
        return imported_files
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get a summary of the current database state"""
        with get_db_session() as session:
            patent_count = session.query(Patent).count()
            chunk_count = session.query(PatentChunk).count()
            file_count = session.query(DataSourceFile).count()
            
            return {
                'total_patents': patent_count,
                'total_chunks': chunk_count,
                'imported_files': file_count
            }


def main():
    """Main function for running the data ingestion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Ingestion Manager for RAG Chatbot")
    parser.add_argument('--data-folder', default='data', help='Path to data folder (default: data)')
    parser.add_argument('--force-reimport', action='store_true', help='Force reimport of all files')
    parser.add_argument('--list-files', action='store_true', help='List imported files')
    parser.add_argument('--summary', action='store_true', help='Show database summary')
    
    args = parser.parse_args()
    
    ingestion_manager = DataIngestionManager(args.data_folder)
    
    if args.list_files:
        print("\n📁 Imported Files:")
        print("-" * 80)
        files = ingestion_manager.list_imported_files()
        for file_info in files:
            print(f"📄 {file_info['filename']}")
            print(f"   📅 Imported: {file_info['import_date']}")
            print(f"   📊 Records: {file_info['records_imported']:,}")
            print(f"   💾 Size: {file_info['file_size']:,} bytes")
            print(f"   🔑 Hash: {file_info['file_hash']}")
            print()
    
    elif args.summary:
        print("\n📊 Database Summary:")
        print("-" * 40)
        summary = ingestion_manager.get_database_summary()
        print(f"🔬 Total Patents: {summary['total_patents']:,}")
        print(f"📝 Total Chunks: {summary['total_chunks']:,}")
        print(f"📁 Imported Files: {summary['imported_files']}")
        print()
    
    else:
        print("\n🚀 Starting Data Import Process...")
        print("=" * 50)
        
        results = ingestion_manager.run_import(force_reimport=args.force_reimport)
        
        print("\n📋 Import Results:")
        print("-" * 30)
        print(f"✅ Files Processed: {results['files_processed']}")
        print(f"📊 Records Imported: {results['total_records_imported']:,}")
        print(f"⏭️ Files Skipped: {results['files_skipped']}")
        
        if results['errors']:
            print(f"\n❌ Errors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"   • {error}")
        
        print("\n🎉 Import process completed!")


if __name__ == "__main__":
    main()
