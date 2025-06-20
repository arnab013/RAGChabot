import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from database.config import get_db_session_simple, close_session
from database.models import Patent

session = get_db_session_simple()

# Simple country check
countries_with_data = session.query(Patent).filter(
    Patent.applicant_countries.isnot(None)
).limit(5).all()

print("Sample country data:")
for p in countries_with_data:
    print(f"{p.publication_number}: {p.applicant_countries}")

close_session(session)
