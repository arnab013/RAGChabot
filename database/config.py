import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base

# Database configuration
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'patents.db')
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all tables"""
    # Ensure data directory exists
    data_dir = os.path.dirname(DATABASE_PATH)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    Base.metadata.create_all(bind=engine)
    print(f"Database created at: {DATABASE_PATH}")

@contextmanager
def get_db_session():
    """Get database session as context manager"""
    session = SessionLocal()
    try:
        yield session
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_db_session_simple():
    """Get database session (non-context manager)"""
    return SessionLocal()

def close_session(session):
    """Close database session"""
    session.close()
