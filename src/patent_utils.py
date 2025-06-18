"""
Patent utility functions for formatting and text processing
"""

import re


def format_patent_number(publication_country, publication_number=None, publication_kind=None):
    """
    Format patent number for display
    
    Args:
        publication_country (str): Country code
        publication_number (str, optional): Publication number. If None, assumes publication_country contains the full number.
        publication_kind (str, optional): Publication kind code (A1, B1, etc.)
        
    Returns:
        str: Formatted patent number as: [country][zero-padded 7-digit number][kind]
        Example: EP0000006B1
    """
    if publication_number is None:
        # If only one parameter is provided, assume it's the full patent number
        if not publication_country:
            return ""
        # Remove any whitespace and convert to uppercase
        return str(publication_country).strip().upper()
    
    # If both parameters are provided
    if not publication_country or not publication_number:
        return ""
    
    # Format as country code + zero-padded number + kind
    country = str(publication_country).strip().upper()
    
    # Convert the publication number to an integer if possible and format with leading zeros
    try:
        number_int = int(str(publication_number).strip())
        number = f"{number_int:07d}"  # Zero-padding to 7 digits
    except (ValueError, TypeError):
        # If not a simple number (contains letters or special chars), just use as is
        number = str(publication_number).strip()
    
    # Add the publication kind if available
    kind_suffix = ""
    if publication_kind:
        kind_suffix = str(publication_kind).strip()
    
    # Return the formatted patent number
    return f"{country}{number}{kind_suffix}"


def remove_similarity_from_text(text):
    """
    Remove similarity scores or redundant text from patent text
    
    Args:
        text (str): Input text that may contain similarity scores
        
    Returns:
        str: Cleaned text with similarity information removed
    """
    if not text:
        return ""
    
    # Remove similarity scores like "(similarity: 0.85)" or "[similarity: 0.85]"
    text = re.sub(r'\(similarity:\s*[\d.]+\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[similarity:\s*[\d.]+\]', '', text, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
