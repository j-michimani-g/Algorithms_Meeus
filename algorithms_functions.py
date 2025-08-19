###### PI value
DPI = 3.141592653589793



def parse_ra_string(ra_string):
    """
    Parse right ascension string into component values.
    
    Takes a right ascension string in format "HH MM SS.SS" and extracts
    the hours, minutes, and seconds as separate float values.
    
    Args:
        ra_string (str): Right ascension in format "HH MM SS.SS"
                        (e.g., "14 23 45.67")
    
    Returns:
        tuple: A tuple of three floats (hours, minutes, seconds)
               e.g., (14.0, 23.0, 45.67)
    
    Example:
        >>> parse_ra_string("14 23 45.67")
        (14.0, 23.0, 45.67)
    """

    try:
        parts = ra_string.strip().split()
        if len(parts) != 3:
            raise ValueError("RA string must be in the format 'HH MM SS.SS'")
        
        ra_hours = float(parts[0])
        ra_minutes = float(parts[1])
        ra_seconds = float(parts[2])
        
        return (ra_hours, ra_minutes, ra_seconds)
    
    except ValueError as e:
        print(f"Error parsing RA string: {e}")
        return None
    

def convert_ra_components(hours, minutes, seconds):
    """
    Convert right ascension components to decimal, degrees, and radians.
    
    Takes individual hours, minutes, and seconds components of right ascension
    and converts them to decimal hours, decimal degrees, and radians.
    
    Args:
        hours (float): Hours component of right ascension
        minutes (float): Minutes component of right ascension  
        seconds (float): Seconds component of right ascension
    
    Returns:
        tuple: A tuple of three floats:
               - decimal_hours (float): RA in decimal hours
               - degrees (float): RA in decimal degrees
               - radians (float): RA in radians
    
    Example:
        >>> convert_ra_components(14.0, 23.0, 45.67)
        14.396019444444445, 215.9402916666667, 3.768...
    
    Note:
        Right ascension ranges from 0-24 hours, 0-360 degrees, or 0-2π radians.
    """
    ra_decimal_hours = hours + minutes/60 + seconds/3600 
    degrees = ra_decimal_hours * 15
    radians = degrees * (DPI/180)    

    return (ra_decimal_hours, degrees, radians)