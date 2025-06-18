from sqlalchemy import create_engine, Column, Integer, String, Text, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import json
import numpy as np

Base = declarative_base()

class Patent(Base):
    __tablename__ = 'patents'
    
    publication_number = Column(String, primary_key=True)
    publication_country = Column(String)
    publication_kind = Column(String)
    publication_date = Column(Date)
    ipc = Column(Text)  # JSON string for SQLite compatibility
    title_en = Column(Text)
    abstract_text = Column(Text)
    prior_art = Column(Text)  # JSON string
    reference = Column(Text)  # JSON string
    parent = Column(Text)  # JSON string
    pct_publication_number = Column(String)
    designated_states_contracting = Column(Text)  # JSON string
    designated_states_extension = Column(Text)  # JSON string
    designated_states_validation = Column(Text)  # JSON string
    sdg_number = Column(Text)  # JSON string for array of integers
    analysis_explanation = Column(Text)  # JSON string
    sdg_technology_fields = Column(Text)  # JSON string
    analysis_potential_beneficiaries = Column(Text)  # JSON string
    ipc_tech_field = Column(Text)  # JSON string
    ipc_technologies = Column(Text)  # JSON string
    applicant_names = Column(Text)  # JSON string
    applicant_countries = Column(Text)  # JSON string
    applicant_count = Column(Integer)
    inventor_names = Column(Text)  # JSON string
    inventor_countries = Column(Text)  # JSON string
    inventor_count = Column(Integer)
    parent_publication_number = Column(String)
    
    # Relationship to chunks
    chunks = relationship("PatentChunk", back_populates="patent", cascade="all, delete-orphan")
    
    def set_array_field(self, field_name, value):
        """Helper method to set array fields as JSON strings"""
        if value and isinstance(value, (list, tuple)):
            setattr(self, field_name, json.dumps(value))
        elif isinstance(value, str) and value.strip():
            # Try to parse as list if it's semicolon separated
            try:
                if ';' in value:
                    setattr(self, field_name, json.dumps([v.strip() for v in value.split(';') if v.strip()]))
                else:
                    setattr(self, field_name, json.dumps([value.strip()]))
            except:
                setattr(self, field_name, json.dumps([str(value)]))
        else:
            setattr(self, field_name, None)
    
    def get_array_field(self, field_name):
        """Helper method to get array fields from JSON strings"""
        value = getattr(self, field_name)
        if value:
            try:
                return json.loads(value)
            except:
                return [value]
        return []

class PatentChunk(Base):
    __tablename__ = 'patent_chunks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    publication_number = Column(String, ForeignKey('patents.publication_number', ondelete='CASCADE'))
    chunk_text = Column(Text)
    chunk_index = Column(Integer)
    embedding = Column(Text)  # JSON string of embedding vector for SQLite
    
    # Duplicate essential metadata for efficient filtering
    sdg_number = Column(Text)  # JSON string
    publication_date = Column(Date)
    designated_states_contracting = Column(Text)  # JSON string
    designated_states_extension = Column(Text)  # JSON string
    designated_states_validation = Column(Text)  # JSON string
    ipc = Column(Text)  # JSON string
    ipc_tech_field = Column(Text)  # JSON string
    analysis_explanation = Column(Text)  # JSON string
    sdg_technology_fields = Column(Text)  # JSON string
    analysis_potential_beneficiaries = Column(Text)  # JSON string
    ipc_technologies = Column(Text)  # JSON string
    
    # Relationship to patent
    patent = relationship("Patent", back_populates="chunks")
    
    def set_embedding(self, embedding_vector):
        """Set embedding as JSON string"""
        if embedding_vector is not None:
            self.embedding = json.dumps(embedding_vector.tolist() if hasattr(embedding_vector, 'tolist') else list(embedding_vector))
    
    def get_embedding(self):
        """Get embedding as numpy array"""
        if self.embedding:
            try:
                return np.array(json.loads(self.embedding))
            except:
                return None
        return None
