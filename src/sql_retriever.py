import numpy as np
import os
import sys
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_, extract, func

# Add the project root to the Python path if the module isn't found
try:
    from database.config import get_db_session, get_db_session_simple, close_session
    from database.models import Patent, PatentChunk
except ModuleNotFoundError:
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from database.config import get_db_session, get_db_session_simple, close_session
    from database.models import Patent, PatentChunk
import requests
import json
from datetime import datetime

class SQLRetriever:
    """
    SQL-based retriever for patent search using vector similarity.
    
    Architecture:
    - Database embeddings: Created using local model (already done during data ingestion)
    - Query embeddings: Generated using remote API with the same model for consistency
    - Search: Compares query embeddings with stored database embeddings using cosine similarity
    
    This ensures maximum consistency as both data and queries use the same embedding model.
    """
    def __init__(self, remote_embedding_url=None, remote_api_key=None):
        """
        Initialize SQL-based retriever with remote embedding API for queries
        
        Args:
            remote_embedding_url: URL for remote embedding API (same model used for data ingestion)
            remote_api_key: API key for remote embedding service
        """
        # Import config here to get remote settings
        from config import REMOTE_EMBEDDING_URL, REMOTE_EMBEDDING_API_KEY
          # Use provided URLs or default from config
        self.remote_embedding_url = remote_embedding_url or REMOTE_EMBEDDING_URL
        self.remote_api_key = remote_api_key or REMOTE_EMBEDDING_API_KEY
        
        if not self.remote_embedding_url:
            raise Exception("Remote embedding API URL is required. Please configure REMOTE_EMBEDDING_URL in your .env file.")
        
        print(f"SQLRetriever initialized with remote embedding API: {self.remote_embedding_url}")
        print("Query embeddings will use remote API (same model that embedded the database)")
        print("Database contains pre-embedded patent chunks using the same model")
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for user query using remote API only"""
        try:
            # Use remote API with the correct payload format
            headers = {'Content-Type': 'application/json'}
            if self.remote_api_key:
                headers['Authorization'] = f'Bearer {self.remote_api_key}'
            
            response = requests.post(
                self.remote_embedding_url,
                json={'query_to_embed': query},
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                # Extract embedding from response
                result = response.json()
                if 'query_embedded' in result:
                    return np.array(result['query_embedded'])
                elif 'embedding' in result:
                    return np.array(result['embedding'])
                elif isinstance(result, list):
                    return np.array(result)
                else:
                    raise Exception(f"Unexpected response format: {result}")
            else:
                raise Exception(f"Remote embedding API failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Failed to get embedding from remote API: {e}. Please check your internet connection and API configuration.")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, 
               query: str, 
               max_passages: int = 400,
               filters: List[Dict[str, Any]] = None,
               column_order: List[str] = None,
               top_k_return: int = 60) -> List[Dict[str, Any]]:
        """
        Search for relevant patent chunks using vector similarity
        
        Args:
            query: Search query
            max_passages: Maximum number of passages to consider
            filters: List of filter dictionaries
            column_order: Preferred column order (not used in SQL version)
            top_k_return: Number of top results to return
        
        Returns:
            List of relevant passages with metadata
        """
        # Use get_db_session_simple instead of the context manager version
        session = get_db_session_simple()
        
        try:
            # Get query embedding from remote API
            query_embedding = self.get_query_embedding(query)
            
            # Start with base query
            sql_query = session.query(PatentChunk)
            
            # Apply filters
            if filters:
                for filter_dict in filters:
                    column = filter_dict.get('column')
                    op = filter_dict.get('op')
                    value = filter_dict.get('value')
                    
                    if column and op and value is not None:
                        sql_query = self._apply_filter(sql_query, column, op, value)
            
            # Get all matching chunks
            chunks = sql_query.limit(max_passages).all()
            
            # Calculate similarities
            similarities = []
            for chunk in chunks:
                chunk_embedding = chunk.get_embedding()
                if chunk_embedding is not None:
                    similarity = self.cosine_similarity(query_embedding, chunk_embedding)
                    similarities.append((chunk, similarity))
            
            # Sort by similarity and take top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_chunks = similarities[:top_k_return]
              # Format results
            results = []
            for chunk, similarity in top_chunks:
                result = {
                    'publication_number': chunk.publication_number,
                    'publication_country': chunk.patent.publication_country if chunk.patent else None,
                    'publication_kind': chunk.patent.publication_kind if chunk.patent else None,
                    'title': chunk.patent.title_en if chunk.patent else '',
                    'text': chunk.chunk_text,
                    'similarity': similarity,
                    'chunk_index': chunk.chunk_index,
                    'publication_date': chunk.publication_date.isoformat() if chunk.publication_date else None,
                    'sdg_number': json.loads(chunk.sdg_number) if chunk.sdg_number else [],
                    'ipc': json.loads(chunk.ipc) if chunk.ipc else [],
                    'applicant_countries': json.loads(chunk.patent.applicant_countries) if chunk.patent and chunk.patent.applicant_countries else []
                }
                results.append(result)
            
            return results
            
        finally:
            session.close()
    
    def _apply_filter(self, query, column: str, op: str, value: Any):
        """Apply a filter to the SQL query"""
        if column == 'publication_date':
            if op == 'gte':
                return query.filter(PatentChunk.publication_date >= value)
            elif op == 'lte':
                return query.filter(PatentChunk.publication_date <= value)
            elif op == 'eq':
                return query.filter(PatentChunk.publication_date == value)
        elif column == 'publication_country':
            if op == 'eq':
                return query.filter(PatentChunk.publication_country == value)
        elif column == 'sdg_number':
            if op == 'eq':
                # Check if SDG number is in the JSON array
                return query.filter(PatentChunk.sdg_number.like(f'%"{value}"%'))
        elif column == 'ipc':
            if op == 'eq':
                return query.filter(PatentChunk.ipc.like(f'%{value}%'))
        elif column == 'applicant_countries':
            if op == 'eq':
                return query.join(Patent).filter(Patent.applicant_countries.like(f'%"{value}"%'))
        
        return query
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        session = get_db_session()
        
        try:
            # Basic counts
            total_patents = session.query(Patent).count()
            total_chunks = session.query(PatentChunk).count()
            
            # Country distribution
            country_stats = session.query(
                PatentChunk.publication_country,
                func.count(PatentChunk.publication_country).label('count')
            ).group_by(PatentChunk.publication_country).all()
            
            # Year distribution
            year_stats = session.query(
                extract('year', PatentChunk.publication_date).label('year'),
                func.count(PatentChunk.id).label('count')
            ).group_by(extract('year', PatentChunk.publication_date)).all()
            
            # SDG distribution
            sdg_stats = {}
            chunks_with_sdgs = session.query(PatentChunk.sdg_number).filter(
                PatentChunk.sdg_number.isnot(None)
            ).all()
            
            for chunk in chunks_with_sdgs:
                if chunk.sdg_number:
                    try:
                        sdgs = json.loads(chunk.sdg_number)
                        for sdg in sdgs:
                            sdg_key = str(sdg)
                            sdg_stats[sdg_key] = sdg_stats.get(sdg_key, 0) + 1
                    except:
                        pass
            
            return {
                'total_patents': total_patents,
                'total_chunks': total_chunks,
                'by_country': {country: count for country, count in country_stats},
                'by_year': {int(year): count for year, count in year_stats if year is not None},
                'by_sdg': sdg_stats
            }
            
        finally:
            session.close()
    
    # For compatibility with old code
    @property
    def df(self):
        """Compatibility property - not used in SQL version"""
        return None
