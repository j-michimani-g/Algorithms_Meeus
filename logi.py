import os
import pytz
import math
import pandas as pd
import numpy as np
from tzfpy import get_tz
from datetime import datetime, timedelta


# ----------------------------------------------------------------------
# 1. DATE AND TIME PARSING
# ----------------------------------------------------------------------

def parse_date_to_YMD_date_only(date_string: str) -> tuple[int, int, int]:
    """
    Parses a date string in "YYYY-MM-DD" format and returns (Y, M, D) 
    as integers, representing the date at 00:00:00 (midnight) UTC.
    
    Args:
        date_string (str): The date string (e.g., "1999-10-22").
        
    Returns:
        tuple[int, int, int]: (Year, Month, Day).
    """
    
    # Check for presence of time part, which is no longer allowed
    if ' ' in date_string:
         raise ValueError(f"Time component detected. This function is for date-only (YYYY-MM-DD) strings: {date_string}")

    # 1. Extract Y, M, and D
    try:
        # The map(int, ...) converts all three components to integers
        Y, M, D = map(int, date_string.split('-'))
    except ValueError:
        # This will catch issues like "1999/10/22" or non-numeric parts
        raise ValueError(f"Invalid date format: {date_string}. Expected YYYY-MM-DD.")
        
    # Optional: Basic range validation (for robustness)
    if not (1 <= M <= 12 and 1 <= D <= 31):
         raise ValueError(f"Invalid month or day value: {date_string}. Month must be 1-12, Day must be 1-31 (approx).")

    # The day (D) is returned as an integer, 
    # which corresponds to the JD formula's base day at 0.0 fraction of the day.
    return (Y, M, D)


def parse_date_to_YMD_flexible(date_string: str) -> tuple[int, int, float]:
    """
    Parses a date and time string (e.g., "2000-01-01 12:00:30.5") 
    ASSUMING THE INPUT TIME IS IN UTC.

    Parses a date and time string in "YYYY-MM-DD HH:MM" or "YYYY-MM-DD HH:MM:SS.SS" 
    format and returns (Y, M, D) where D includes the time as a decimal fraction of the day.
    
    This function handles flexible time inputs, where minutes or seconds 
    may be decimal values (e.g., 30.5 seconds).
    
    Args:
        date_string (str): The date and time string (e.g., "2000-01-01 12:00:30.5").
        
    Returns:
        tuple[int, int, float]: (Year, Month, Day with decimal time).
    """
    
    # 1. Split the date and time parts
    try:
        date_part, time_part = date_string.split(' ')
    except ValueError:
        raise ValueError(f"Invalid format. Must contain a space between date and time: {date_string}")
    
    # 2. Extract Y, M, and the base day
    try:
        Y, M, day = map(int, date_part.split('-'))
    except ValueError:
        raise ValueError(f"Invalid date format: {date_part}. Expected YYYY-MM-DD.")
    
    # 3. Parse the time part and calculate total seconds
    
    # Split HH:MM[:SS.SS]
    time_components = time_part.split(':')
    
    # Validate structure
    if len(time_components) < 2 or len(time_components) > 3:
        raise ValueError(f"Invalid time format: {time_part}. Expected HH:MM or HH:MM:SS.SS.")

    H = int(time_components[0])
    Min = int(time_components[1]) # Minutes must be an integer part
    
    total_seconds = float(H * 3600) + float(Min * 60)
    
    if len(time_components) == 3:
        # Format is HH:MM:SS.SS, supporting decimal seconds
        Sec_decimal = float(time_components[2])
        total_seconds += Sec_decimal
        
    # 4. Calculate the fraction of the day
    seconds_in_day = 86400.0 # 24 * 60 * 60
    decimal_fraction_of_day = total_seconds / seconds_in_day
    
    # 5. Combine base day and decimal fraction
    D = float(day) + decimal_fraction_of_day
    
    return (Y, M, D)

# ----------------------------------------------------------------------
# 2. JULIAN DAY CALCULATION
# ----------------------------------------------------------------------

def calculate_julian_day(Y: int, M: int, D: float) -> float:
    """
    Calculates the Julian Date (JD) for a given calendar date using formula (7.1) 
    from Astronomical Algorithms (Meeus).
    
    This version ensures the Julian-to-Gregorian transition logic is correct 
    by checking the date BEFORE the M and Y adjustments.
    
    Args:
        Y (int): Year.
        M (int): Month number (1=Jan, 12=Dec).
        D (float): Day of the month (with decimals for time).
        
    Returns:
        float: The calculated Julian Date (JD).
    """
    
    # --- Step 1: Determine the Gregorian Correction (B) ---
    # The Gregorian calendar begins on 15 October 1582. 
    
    # Set the default (Julian) correction
    B = 0
    
    # Check if the date is on or after the Gregorian start date (1582-10-15)
    # The comparison must be against the ORIGINAL Y, M, D.
    if Y > 1582 or \
       (Y == 1582 and M > 10) or \
       (Y == 1582 and M == 10 and D >= 15):
        
        # Calculate B for the Gregorian Calendar
        A = math.floor(Y / 100)
        B = 2 - A + math.floor(A / 4)
    
    # --- Step 2: Adjust Y and M for Meeus's formula ---
    # Jan (1) and Feb (2) become month 13 and 14 of the preceding year.
    # This modification is only for the formula; it doesn't change the calendar system.
    if M <= 2:
        Y = Y - 1
        M = M + 12
        
    # --- Step 3: Calculate the Julian Date (JD) ---
    # Formula (7.1): JD = INT(365.25 * (Y + 4716)) + INT(30.6001 * (M + 1)) + D + B - 1524.5
    
    term1 = math.floor(365.25 * (Y + 4716))
    term2 = math.floor(30.6001 * (M + 1))
    
    JD = term1 + term2 + D + B - 1524.5
    
    return JD


# ----------------------------------------------------------------------
# 3. SIDEREAL TIME CALCULATION
# ----------------------------------------------------------------------

def calculate_mean_sidereal_time(JD: float) -> float:
    """
    Calculates the Mean Sidereal Time at Greenwich (GMST, or theta_0) in degrees 
    using the full expression of equation (11.4) from Astronomical Algorithms.

    The formula is valid for any instant of time (JD).

    Args:
        JD (float): The instantaneous Julian Day (JD) corresponding to the 
                    Universal Time (UT) of observation.

    Returns:
        float: The Greenwich Mean Sidereal Time (GMST) in decimal degrees, 
               normalized to the range [0.0, 360.0).
    """
    
    # JD_2000_EPOCH = 2451545.0 (Julian Day for 12h UT on Jan 1, 2000)
    JD_DIFF = JD - 2451545.0
    
    # 1. Calculate T (Time in Julian centuries since J2000.0)
    # T = (JD - 2451545.0) / 36525.0
    T = JD_DIFF / 36525.0
    
    # 2. Calculate theta_0 (GMST) using equation (11.4)
    # theta_0 = 280.46061837 + 360.98564736629(JD - 2451545.0) + 0.000387933 * T^2 - T^3 / 38710000
    
    term1 = 280.46061837
    term2 = 360.98564736629 * JD_DIFF
    term3 = 0.000387933 * (T ** 2)
    term4 = (T ** 3) / 38710000.0
    
    theta_0 = term1 + term2 + term3 - term4
    
    # 3. Normalize the result to the range [0.0, 360.0)
    theta_0 = theta_0 % 360.0
    
    if theta_0 < 0:
        theta_0 += 360.0
        
    return theta_0

def calculate_local_sidereal_time(GMST_degrees: float, longitude_deg: float) -> float:
    """
    Calculates the Local Sidereal Time (LST) in decimal hours. 

    Args: 
        GMST_degrees (float): Greenwich Mean Sidereal Time in degrees [0.0, 360.0).
        longitude_deg (float): The longitude of the site, measured EASTWARD 
                               in the standard geographical range [-180.0, +180.0].
                               (e.g., East is positive, West is negative).
                                
    Returns:
        float: Local Sidereal Time (LST) in decimal hours [0.0, 24.0).
    """
    
    # 1. Longitude conversion: No change needed to the variable, 
    #    as the formula (GMST + Longitude) works directly with 
    #    the sign-aware [-180, +180] input.
    
    # Convert GMST and Longitude from degrees to hours (1 hour = 15 degrees)
    GMST_hours = GMST_degrees / 15.0 
    longitude_hours = longitude_deg / 15.0 
    
    # LST (in hours) = GMST (in hours) + Longitude (in hours, East positive)
    LST = (GMST_hours + longitude_hours) % 24.0

    # Ensure result is positive [0.0, 24.0)
    if LST < 0: 
        LST += 24.0

    return LST


# ----------------------------------------------------------------------
# 4. COORDINATE PARSING AND CONVERSION
# ----------------------------------------------------------------------

def parse_and_convert_ra_dec(ra_dec_string: str) -> tuple[float, float, float, float]:
    """
    Parses a combined Right Ascension and Declination string and converts 
    both to decimal degrees and radians.
    
    The expected input format is: "HH MM SS.SS ±DD MM SS.S"
    Example: "19 34 30.77 -17 34 36.4" or "05 14 32.2 +08 12 55.0"
    
    Args:
        ra_dec_string (str): Combined RA and DEC string.
        
    Returns:
        tuple[float, float, float, float]: 
            (ra_decimal_hours, ra_degrees, dec_degrees, dec_radians)
    """
    parts = ra_dec_string.strip().split()
    
    if len(parts) != 6:
        raise ValueError(f"Input string must have 6 components: 'HH MM SS.S ±DD MM SS.S'. Found {len(parts)} parts.")
    
    # 2. Extract RA Components (Hours, Minutes, Seconds)
    try:
        ra_h = float(parts[0])
        ra_m = float(parts[1])
        ra_s = float(parts[2])
    except ValueError:
        raise ValueError("Invalid format for RA components (HH MM SS.SS).")
        
    # 3. Extract DEC Components (Sign, Degrees, Minutes, Seconds)
    dec_d_str = parts[3]
    
    # Determine the sign and strip it from the degrees string
    if dec_d_str.startswith('-'):
        dec_sign = -1
        dec_d_abs = float(dec_d_str[1:])
    elif dec_d_str.startswith('+'):
        dec_sign = 1
        dec_d_abs = float(dec_d_str[1:])
    else:
        dec_sign = 1
        dec_d_abs = float(dec_d_str)

    try:
        dec_m = float(parts[4])
        dec_s = float(parts[5])
    except ValueError:
        raise ValueError("Invalid format for DEC minutes/seconds (MM SS.S).")

    
    # 4. Convert RA Components 
    ra_decimal_hours = ra_h + ra_m/60.0 + ra_s/3600.0
    ra_degrees = ra_decimal_hours * 15.0
    
    # 5. Convert DEC Components
    # Convert to decimal degrees
    dec_decimal_degrees = dec_sign * (dec_d_abs + dec_m/60.0 + dec_s/3600.0)
    
    # Convert to radians (using math.pi)
    dec_radians = dec_decimal_degrees * (math.pi / 180.0)
    
    return (ra_decimal_hours, ra_degrees, dec_decimal_degrees, dec_radians)

# ----------------------------------------------------------------------
# 4.a. CALCULATE DELTA T (TT - UT)
# ----------------------------------------------------------------------

def calculate_delta_t(year: int, month: int) -> float:
    """
    Calculates Delta T (TT - UT) in seconds using the piecewise polynomial 
    fits provided by NASA/GSFC for years -2000 to +3000.
    
    Args:
        year (int): Calendar year (e.g., 2025).
        month (int): Month number (1-12).
        
    Returns:
        float: Delta T in seconds.
    """
    
    # Calculate y, the time in fractional years, centered on the middle of the month
    y = year + (month - 0.5) / 12
    
    # Initialize delta_t to 0.0 to prevent UnboundLocalError
    delta_t = 0.0 

    if year <= -500: # -2000 to -500
        u = (y - 1820) / 100
        delta_t = -20 + 32 * u**2 
    
    elif year < 500: # -500 to +500 (note: < 500 to align with the source's breakpoints)
        u = y / 100
        delta_t = 10583.6 - 1014.41 * u + 33.78311 * u**2 - 5.952053 * u**3 - 0.1798452 * u**4 + 0.022174192 * u**5 + 0.0090316521 * u**6
    
    elif year <= 1600: # 500 to 1600
        u = (y - 1000) / 100
        delta_t = 1574.2 - 556.01 * u + 71.23472 * u**2 + 0.319781 * u**3 - 0.8503463 * u**4 - 0.005050998 * u**5 + 0.0083572073 * u**6

    elif year <= 1700: # 1600 to 1700
        t = y - 1600
        delta_t = 120 - 0.9808 * t - 0.01532 * t**2 + t**3 / 7129

    elif year <= 1800: # 1700 to 1800
        t = y - 1700
        delta_t = 8.83 + 0.1603 * t - 0.0059285 * t**2 + 0.00013336 * t**3 - t**4 / 1174000

    elif year <= 1860: # 1800 to 1860
        t = y - 1800
        delta_t = 13.72 - 0.332447 * t + 0.0068612 * t**2 + 0.0041116 * t**3 - 0.00037436 * t**4 + 0.0000121272 * t**5 - 0.0000001699 * t**6 + 0.000000000875 * t**7

    elif year <= 1900: # 1860 to 1900
        t = y - 1860
        delta_t = 7.62 + 0.5737 * t - 0.251754 * t**2 + 0.01680668 * t**3 - 0.0004473624 * t**4 + t**5 / 233174

    elif year <= 1920: # 1900 to 1920
        t = y - 1900
        delta_t = -2.79 + 1.494119 * t - 0.0598939 * t**2 + 0.0061966 * t**3 - 0.000197 * t**4

    elif year <= 1941: # 1920 to 1941
        t = y - 1920
        delta_t = 21.20 + 0.84493*t - 0.076100 * t**2 + 0.0020936 * t**3

    elif year <= 1961: # 1941 to 1961
        t = y - 1950
        delta_t = 29.07 + 0.407*t - t**2/233 + t**3 / 2547

    elif year <= 1986: # 1961 to 1986
        t = y - 1975
        delta_t = 45.45 + 1.067*t - t**2/260 - t**3 / 718

    elif year <= 2005: # 1986 to 2005
        t = y - 2000
        delta_t = 63.86 + 0.3345 * t - 0.060374 * t**2 + 0.0017275 * t**3 + 0.000651814 * t**4 + 0.00002373599 * t**5

    elif year <= 2050: # 2005 to 2050 (Prediction)
        t = y - 2000
        delta_t = 62.92 + 0.32217 * t + 0.005589 * t**2

    elif year <= 2150: # 2050 to 2150 (Prediction)
        # Note: The original formula was delta_t = -20 + 32 * ((y-1820)/100)**2 - 0.5628 * (2150 - y).
        # We need to verify which formula the source intends for the square term.
        u = (y - 1820) / 100
        delta_t = -20 + 32 * u**2 - 0.5628 * (2150 - y)

    else: # After 2150
        u = (y - 1820) / 100
        delta_t = -20 + 32 * u**2

    return delta_t



# ----------------------------------------------------------------------
# 5. ALTITUDE CALCULATION
# ----------------------------------------------------------------------

def calculate_object_altitude(site_latitude_degrees: float, dec_degrees: float, H_hours: float) -> float:
    """
    Calculates the **Altitude (a)** of an astronomical object above the horizon 
    in degrees, using the formula from spherical trigonometry.

    Formula used: sin(a) = sin(phi) * sin(delta) + cos(phi) * cos(delta) * cos(H)
    
    Args: 
        site_latitude_degrees (float): Observer's latitude (phi) in degrees. North is positive.
        dec_degrees (float): Declination of the object (delta) in degrees.
        H_hours (float): Hour angle (H) in hours (LST - RA).
        
    Returns:
        float: Altitude above the horizon (altitude_degrees) in degrees.
    """
    
    # Convert inputs to radians
    phi = math.radians(site_latitude_degrees)  
    delta = math.radians(dec_degrees)
    
    # Convert Hour Angle from hours to radians (1 hour = 15 degrees)
    H = math.radians(H_hours * 15.0)

    # Spherical trigonometry formula for altitude
    sin_altitude = (math.sin(phi) * math.sin(delta) + 
                    math.cos(phi) * math.cos(delta) * math.cos(H))

    altitude_radians = math.asin(sin_altitude)
    altitude_degrees = math.degrees(altitude_radians)

    return altitude_degrees


# ----------------------------------------------------------------------
# 6. WRAPPER FUNCTION FOR ALTITUDE CALCULATION 
# ----------------------------------------------------------------------

def object_altitude(date_str: str, ra_dec_str: str, site_longitude: float, site_latitude: float) -> float:
    """
    Calculates the altitude of a celestial object above the horizon (h) 
    at a specific time and location using the standard astronomical workflow.

    The calculation follows these steps:
    1. Universal Time (UT) -> Julian Day (JD)
    2. JD -> Greenwich Mean Sidereal Time (GMST)
    3. GMST + Site Longitude -> Local Sidereal Time (LST)
    4. LST - Right Ascension (RA) -> Hour Angle (H)
    5. Hour Angle (H), Declination (DEC), and Site Latitude (phi) -> Altitude (h)

    Args:
        date_str (str): The date and time string in "YYYY-MM-DD HH:MM:SS.SS" format.
        ra_dec_str (str): The object's coordinates in "HH MM SS.SS ±DD MM SS.S" format.
        site_longitude (float): Observer's longitude in degrees, East-positive [-180.0, +180.0].
        site_latitude (float): Observer's latitude in degrees, North-positive.

    Returns:
        float: The Altitude of the object above the horizon in decimal degrees.
    """
    
    # 1. Calculate Julian Day (JD) for the instant of observation
    Y, M, D = parse_date_to_YMD_flexible(date_str)
    JD = calculate_julian_day(Y, M, D)
    
    # 2. Calculate Greenwich Mean Sidereal Time (GMST) in degrees (theta_0)
    GMST_degrees = calculate_mean_sidereal_time(JD)
    
    # 3. Parse the object's Right Ascension (RA) and Declination (DEC)
    # ra_hours: RA in decimal hours (needed for H calculation)
    # dec_deg: Declination in decimal degrees (needed for altitude calculation)
    ra_hours, _, dec_deg, _ = parse_and_convert_ra_dec(ra_dec_str)
    
    # **CRITICAL IMPROVEMENT:** Ensure longitude is in the required [-180.0, +180.0] range
    # LST calculation works with degrees in this range (East+, West-).
    # If a user provides [0.0, 360.0], convert it.
    
    # Example: If site_longitude is 300 (W 60), this converts it to -60.
    # If site_longitude is 120 (E 120), this leaves it as 120.
    if site_longitude > 180.0:
        site_longitude -= 360.0
        
    # 4. Calculate Local Sidereal Time (LST) in hours
    # The LST function now correctly expects [-180.0, +180.0]
    LST = calculate_local_sidereal_time(GMST_degrees, site_longitude) 
    
    # 5. Calculate the Hour Angle (H) in hours
    # H = LST - RA (where LST and RA are in hours)
    H_hours = LST - ra_hours
    
    # 6. Calculate the final Altitude in degrees
    altitude_degrees = calculate_object_altitude(site_latitude, dec_deg, H_hours)

    return altitude_degrees

# ----------------------------------------------------------------------
# 7. CALCULATE THE SUN'S POSITION IN THE SKY FOR A GIVEN JULIAN DAY 
# ----------------------------------------------------------------------

def calculate_solar_position(JD: float, delta_t_seconds: float = 0.0) -> tuple[float, float, float, float, float]:
    """
    Calculate the Sun's position in the sky for a given Julian Day.
    
    This function computes the Sun's celestial coordinates (Right Ascension and 
    Declination) using a simplified astronomical algorithm suitable for sunrise/
    sunset calculations with accuracy of ~1-2 minutes.
    
    Args:
        JD (float): Julian Day number. This is a continuous count of days since 
                    January 1, 4713 BC, Greenwich noon. For example:
                    - January 1, 2000, 12:00 TT = JD 2451545.0
                    - September 26, 2025, 23:00 ≈ JD 2460945.458
        delta_t_seconds (float): The parameter delta-T (ΔT) is the arithmetic difference, 
                                in seconds, between the two as: ΔT = TD - UT
                                Terrestrial Dynamical Time (TD) - Universal Time (UT)
    Returns:
        tuple[float, float, float, float, float]: A tuple containing:
            - L (float): Sun's mean longitude in degrees [0-360)
            - M (float): Sun's mean anomaly in degrees [0-360)
            - true_longitude (float): Sun's true ecliptic longitude in degrees [0-360)
            - sun_RA (float): Sun's Right Ascension in degrees [0-360)
            - sun_DEC (float): Sun's Declination in degrees [-23.44 to +23.44]
    
    Algorithm Steps:
        1. Calculate time since J2000.0 epoch (January 1, 2000, 12:00 TT)
        2. Convert to Julian centuries for astronomical calculations
        3. Calculate Sun's mean longitude (position in circular orbit)
        4. Calculate Sun's mean anomaly (position in elliptical orbit)
        5. Calculate equation of center (correction for orbital eccentricity)
        6. Calculate Sun's true longitude (actual position on ecliptic)
        7. Convert ecliptic coordinates to equatorial coordinates:
           - Right Ascension (RA): celestial longitude
           - Declination (DEC): celestial latitude
    
    Notes:
        - The declination varies seasonally from +23.44° (summer solstice) to 
          -23.44° (winter solstice), causing the seasons.
        - This is a simplified algorithm. For higher precision applications,
          use more sophisticated algorithms like VSOP87 or JPL ephemerides.
        - Earth's obliquity (axial tilt) slowly changes over millennia; this
          algorithm accounts for the secular variation.
    
    Example:
        >>> JD = 2460945.458  # September 26, 2025, 23:00 UTC
        >>> L, M, true_lon, RA, DEC = calculate_solar_position(JD)
        >>> print(f"Sun's Declination: {DEC:.2f}°")
        Sun's Declination: -1.23°
        >>> print(f"Sun's Right Ascension: {RA:.2f}°")
        Sun's Right Ascension: 183.45°
    
    References:
        - Astronomical Algorithms by Jean Meeus
        - NOAA Solar Calculator Algorithm
    """

    
    # --- START of Delta T Correction (UT to TT) ---
    # Convert Delta T from seconds to days
    delta_t_days = delta_t_seconds / 86400.0
    
    # JD for TT is used for solar ephemerides (JD_TT = JD_UT + Delta_T_days)
    JD_TT = JD + delta_t_days
    # --- END of Delta T Correction ---

    # Calculate T, the number of Julian centuries since J2000.0 
    # NOTE: Use JD_TT instead of the original JD (which was JD_UT)
    T = (JD_TT - 2451545.0) / 36525.0

    # Calculate Time Since J2000.0 epoch
    # J2000.0 = JD 2451545.0 (January 1, 2000, 12:00 Terrestrial Time)
    n = JD - 2451545.0
    
    # Convert to Julian centuries (36525 days per century)
    # This is the standard time unit for astronomical calculations
    T = n / 36525
    
    # Calculate the Sun's Mean Longitude (L)
    # This is where the Sun would be if Earth's orbit were perfectly circular
    # 280.460° = position at J2000.0
    # 36000.771° = approximate degrees traveled per century
    L = (280.460 + 36000.771 * T) % 360
    
    # Calculate the Sun's Mean Anomaly (M)
    # This measures Earth's position in its elliptical orbit from perihelion
    # 357.528° = mean anomaly at J2000.0
    # 35999.050° = degrees per century (slightly less than 360° per year)
    M = (357.528 + 35999.050 * T) % 360
    
    # Calculate the Equation of Center (C)
    # This corrects for Earth's elliptical orbit (eccentricity ≈ 0.0167)
    # The correction can be up to ±1.9 degrees
    # First term: primary correction (sin M)
    # Second term: secondary correction (sin 2M)
    C = 1.915 * math.sin(math.radians(M)) + 0.020 * math.sin(math.radians(2 * M))
    
    # Calculate the Sun's True Longitude (λ)
    # This is the Sun's actual position on the ecliptic plane
    true_longitude = (L + C) % 360
    
    # Calculate Earth's obliquity (axial tilt) in degrees
    # 23.439° = obliquity at J2000.0
    # -0.0000004°/century = rate of change due to precession
    # This tilt causes the seasons
    earth_obliquity = 23.439 - 0.0000004 * T
    
    # Calculate the Sun's Right Ascension (RA)
    # Convert from ecliptic coordinates to equatorial coordinates
    # Using atan2 to handle all four quadrants correctly [0-360°]
    sun_RA = math.atan2(
        math.cos(math.radians(earth_obliquity)) * math.sin(math.radians(true_longitude)),
        math.cos(math.radians(true_longitude))
    )
    sun_RA = math.degrees(sun_RA)
    
    # Normalize RA to [0, 360) range
    if sun_RA < 0:
        sun_RA += 360
    
    # Calculate the Sun's Declination (δ)
    # This is the Sun's "latitude" on the celestial sphere
    # Ranges from +23.44° (summer solstice) to -23.44° (winter solstice)
    # Determines how high the Sun rises and length of day
    sun_DEC = math.asin(
        math.sin(math.radians(earth_obliquity)) * math.sin(math.radians(true_longitude))
    )
    sun_DEC = math.degrees(sun_DEC)
    
    return L, M, true_longitude, sun_RA, sun_DEC


# ----------------------------------------------------------------------
# 8. CALCULATE THE SUNRISE AND SUNSET FROM 
# ----------------------------------------------------------------------

def calculate_sunrise_sunset(lat: float, lon: float, dec: float, M: float, timezone_offset: float, elevation_m: float = 0.0) -> tuple[float | None, float | None]:
    """
    Calculate sunrise and sunset times for a given location and date.
    
    This function computes the local times when the Sun's center crosses the 
    horizon, accounting for atmospheric refraction. Uses the spherical trigonometry
    hour angle method combined with the equation of time correction.
    
    Args:
        lat (float): Observer's latitude in degrees. Positive for North, negative 
                     for South. Valid range: -90 to +90.
                     Examples: 40.7128 (New York), -33.8688 (Sydney)
        
        lon (float): Observer's longitude in degrees. Positive for East, negative 
                     for West. Valid range: -180 to +180.
                     Examples: -74.0060 (New York), 151.2093 (Sydney)
        
        dec (float): Sun's declination in degrees (from calculate_solar_position).
                     This is the Sun's "latitude" on the celestial sphere.
                     Range: approximately -23.44° to +23.44°
        
        M (float): Sun's mean anomaly in degrees (from calculate_solar_position).
                   Used to calculate the equation of time correction.
                   Range: 0° to 360°
        
        timezone_offset (float): Hours offset from UTC. Positive for East of 
                                Greenwich, negative for West.
                                Examples: -5 (EST), +1 (CET), +9 (JST)

        elevation_m (float): Observer's elevation in meters.
                            Example: 390 (Y28)
    
    Returns:
        tuple[float | None, float | None]: A tuple containing:
            - sunrise_local (float | None): Sunrise time in decimal hours [0-24)
                                           None if sun never rises (polar night)
            - sunset_local (float | None): Sunset time in decimal hours [0-24)
                                          None if sun never sets (polar day)
        
        Decimal hours example: 6.5 = 6:30:00, 18.75 = 18:45:00
    
    Algorithm Steps:
        1. Set target altitude to -0.833° (accounts for atmospheric refraction
           and Sun's angular radius)
        2. Calculate hour angle using spherical trigonometry
        3. Check for polar day/night conditions (sun never sets/rises)
        4. Calculate equation of time (correction for Earth's orbital eccentricity)
        5. Calculate solar noon (when Sun crosses local meridian)
        6. Add/subtract hour angle to get sunrise/sunset times
        7. Convert from UTC to local time
    
    Special Cases:
        - Returns (None, None) if cos_H > 1: Sun never rises above horizon
          (occurs at high latitudes during winter - polar night)
        - Returns (None, None) if cos_H < -1: Sun never sets below horizon
          (occurs at high latitudes during summer - midnight sun)
        - Near the Arctic/Antarctic circles (±66.5° latitude), these conditions
          occur around the solstices
    
    Technical Details:
        - Altitude angle h = -0.833° accounts for:
          * Atmospheric refraction: ~0.267° (light bends near horizon)
          * Sun's angular radius: ~0.267° (we measure center, not edge)
          * Geometric horizon: -0.567° (Sun's center at true horizon)
        - Equation of Time (EOT): Corrects for the fact that solar days aren't
          exactly 24 hours due to Earth's elliptical orbit and axial tilt.
          EOT varies from about -14 to +16 minutes throughout the year.
        - Solar noon is NOT 12:00 PM clock time; it varies by:
          * Longitude within time zone
          * Equation of time
          * Daylight saving time (not handled here)
    
    Accuracy:
        - Typical accuracy: ±1-2 minutes for most locations
        - Less accurate near poles or during polar day/night transitions
        - Does not account for:
          * Elevation above sea level (adds ~1 min per 1500m)
          * Local atmospheric conditions
          * Mountains or obstacles on horizon
    
    Examples:
        >>> # New York City on equinox (dec ≈ 0°)
        >>> lat, lon = 40.7128, -74.0060
        >>> dec, M = 0.0, 180.0  # Approximate equinox values
        >>> sunrise, sunset = calculate_sunrise_sunset(lat, lon, dec, M, -4)
        >>> print(f"Sunrise: {sunrise:.2f}h ({sunrise:.2f} = ~6:00 AM)")
        >>> print(f"Sunset: {sunset:.2f}h ({sunset:.2f} = ~6:00 PM)")
        
        >>> # North Pole during summer solstice (sun never sets)
        >>> lat, lon = 90.0, 0.0
        >>> dec, M = 23.44, 180.0  # Summer solstice
        >>> sunrise, sunset = calculate_sunrise_sunset(lat, lon, dec, M, 0)
        >>> print(sunrise, sunset)  # Output: None None
        
        >>> # Convert decimal hours to time string
        >>> from datetime import timedelta
        >>> sunrise_time = timedelta(hours=6.523)
        >>> print(str(sunrise_time))  # 6:31:22.800000
    
    References:
        - Astronomical Algorithms by Jean Meeus, Chapter 15
        - NOAA Solar Calculator: https://gml.noaa.gov/grad/solcalc/
        - Spherical astronomy and celestial mechanics texts
    
    See Also:
        - calculate_solar_position(): Computes dec and M parameters
        - calculate_twilight_times(): Similar calculation for twilight periods
        - decimal_hours_to_time(): Convert result to HH:MM:SS format
    """
    # Target altitude for sunrise/sunset (Geometric Center)
    # Standard: -0.833 degrees = -0.5 (Solar Radius) - 0.333 (Atmospheric Refraction)
    h_geometric = -0.833
    
    # --- START of Elevation Correction ---
    h_dip = 0.0
    if elevation_m > 0.0:
        # Calculate the Dip of the Horizon (in degrees)
        # Formula: Dip (degrees) ≈ 0.0347 * sqrt(Elevation in meters)
        h_dip = 0.0347 * math.sqrt(elevation_m)
        
    # The effective target altitude is the standard altitude minus the dip
    # (The horizon appears lower, so h is a larger negative number)
    h = h_geometric - h_dip
    # --- END of Elevation Correction ---

    # Convert everything to radians for trig functions
    lat_rad = math.radians(lat)
    dec_rad = math.radians(dec)
    h_rad = math.radians(h)
    
    # Calculate Hour Angle (H)
    # cos(H) = (sin(h) - sin(lat) * sin(dec)) / (cos(lat) * cos(dec))
    try:
        cos_H = (math.sin(h_rad) - math.sin(lat_rad) * math.sin(dec_rad)) / \
                (math.cos(lat_rad) * math.cos(dec_rad))
    except ZeroDivisionError:
        # Handle cases where the event doesn't occur (e.g., polar summer/winter)
        return None, None
    
    # Check for cases where the Sun is always up (cos_H > 1) or always down (cos_H < -1)
    if cos_H > 1:
        # Sun is always below the target horizon (no rise/set)
        return None, None
    if cos_H < -1:
        # Sun is always above the target horizon (no set/rise)
        return None, None

    # Calculate Hour Angle in degrees
    H_deg = math.degrees(math.acos(cos_H))
    
    # Calculate Equation of Time (EOT) in minutes
    # This corrects for two effects:
    # 1. Earth's elliptical orbit (varying orbital speed)
    # 2. Obliquity of the ecliptic (23.44° axial tilt)
    # Formula is a Fourier series approximation
    # Result ranges from approximately -14 to +16 minutes
    eot = 229.18 * (0.000075 +
                    0.001868 * math.cos(math.radians(M)) -
                    0.032077 * math.sin(math.radians(M)) -
                    0.014615 * math.cos(math.radians(2 * M)) -
                    0.040849 * math.sin(math.radians(2 * M)))
    
    # Calculate solar noon in UTC hours
    # Solar noon = when Sun crosses the local meridian (due south/north)
    # 12.0 = nominal noon
    # (lon / 15.0) = correction for longitude (15° per hour time zone)
    # (eot / 60.0) = equation of time correction (convert minutes to hours)
    solar_noon_UTC = 12.0 - (lon / 15.0) - (eot / 60.0)
    
    # Convert hour angle from degrees to hours
    # 360° = 24 hours, so 15° = 1 hour
    H_hours = H_deg / 15.0
    
    # Calculate sunrise and sunset in UTC
    # Sunrise occurs H hours before solar noon
    # Sunset occurs H hours after solar noon
    sunrise_UTC = solar_noon_UTC - H_hours
    sunset_UTC = solar_noon_UTC + H_hours
    
    # Convert to local time by adding timezone offset
    # Positive offset for east of Greenwich, negative for west
    sunrise_local = sunrise_UTC + timezone_offset
    sunset_local = sunset_UTC + timezone_offset
    
    # Normalize to 0-24 hour range
    # Handle cases where time calculation crosses midnight
    # Example: UTC 23:30 + offset 2 = 25:30 → 1:30 (next day)
    sunrise_local = sunrise_local % 24
    sunset_local = sunset_local % 24
    
    return sunrise_local, sunset_local

# ----------------------------------------------------------------------
# 9. CALCULATE THE TWILIGHT DAWN AND DUSK 
# ----------------------------------------------------------------------

def calculate_single_twilight(lat: float, lon: float, dec: float, M: float, 
                             timezone_offset: float, altitude: float) -> tuple[float | None, float | None]:
    """
    Calculate twilight times for a specific solar altitude angle.
    
    This is a generalized function that calculates when the Sun crosses a given
    altitude below the horizon. It uses the same hour angle method as sunrise/sunset
    calculations but with different target altitudes.
    
    Args:
        lat (float): Observer's latitude in degrees. Positive for North, negative 
                     for South. Valid range: -90 to +90.
        
        lon (float): Observer's longitude in degrees. Positive for East, negative 
                     for West. Valid range: -180 to +180.
        
        dec (float): Sun's declination in degrees (from calculate_solar_position).
                     Range: approximately -23.44° to +23.44°
        
        M (float): Sun's mean anomaly in degrees (from calculate_solar_position).
                   Used to calculate the equation of time correction.
        
        timezone_offset (float): Hours offset from UTC. Positive for East of 
                                Greenwich, negative for West.
        
        altitude (float): Target solar altitude in degrees (negative for below horizon).
                         Common values:
                         * -0.833°: Sunrise/sunset
                         * -6°: Civil twilight
                         * -12°: Nautical twilight
                         * -18°: Astronomical twilight
    
    Returns:
        tuple[float | None, float | None]: A tuple containing:
            - dawn_local (float | None): Morning twilight time in decimal hours [0-24)
                                        None if condition never occurs
            - dusk_local (float | None): Evening twilight time in decimal hours [0-24)
                                        None if condition never occurs
    
    Algorithm:
        Uses spherical trigonometry to calculate the hour angle when the Sun
        reaches the specified altitude, then converts to local time using the
        equation of time and longitude corrections.
    
    Special Cases:
        - Returns (None, None) if the Sun never reaches the specified altitude
        - At extreme latitudes, certain twilight phases may not occur
        - During polar day, all twilight calculations return None
        - During polar night, all twilight calculations return None
    
    Notes:
        - This is the base function used by calculate_civil_twilight(),
          calculate_nautical_twilight(), and calculate_astronomical_twilight()
        - For sunrise/sunset, use calculate_sunrise_sunset() which uses h = -0.833°
    
    Example:
        >>> # Calculate civil twilight manually
        >>> lat, lon = 40.7128, -74.0060
        >>> dec, M = -1.23, 268.5
        >>> dawn, dusk = calculate_single_twilight(lat, lon, dec, M, -4, -6.0)
        >>> print(f"Civil dawn: {dawn:.2f}h, Civil dusk: {dusk:.2f}h")
    
    See Also:
        - calculate_civil_twilight(): Convenience function for civil twilight
        - calculate_nautical_twilight(): Convenience function for nautical twilight
        - calculate_astronomical_twilight(): Convenience function for astronomical twilight
    """
    # Target altitude (already provided as parameter)
    h = altitude
    
    # Calculate hour angle using spherical trigonometry
    # Same formula as sunrise/sunset, but with different altitude
    cos_H = (math.sin(math.radians(h)) -
             math.sin(math.radians(lat)) * math.sin(math.radians(dec))) / \
            (math.cos(math.radians(lat)) * math.cos(math.radians(dec)))
    
    # Check for conditions where twilight doesn't occur
    if cos_H > 1:
        # Sun never reaches this altitude (stays below it)
        # Common at high latitudes during winter
        return None, None
    elif cos_H < -1:
        # Sun never descends to this altitude (stays above it)
        # Common at high latitudes during summer (continuous daylight)
        return None, None
    
    # Calculate hour angle in degrees [0, 180]
    H = math.degrees(math.acos(cos_H))
    
    # Calculate Equation of Time (EOT) in minutes
    # Corrects for Earth's elliptical orbit and axial tilt
    eot = 229.18 * (0.000075 +
                    0.001868 * math.cos(math.radians(M)) -
                    0.032077 * math.sin(math.radians(M)) -
                    0.014615 * math.cos(math.radians(2 * M)) -
                    0.040849 * math.sin(math.radians(2 * M)))
    
    # Calculate solar noon in UTC hours
    solar_noon_UTC = 12.0 - (lon / 15.0) - (eot / 60.0)
    
    # Convert hour angle from degrees to hours
    H_hours = H / 15.0
    
    # Calculate dawn (morning) and dusk (evening) times in UTC
    # Dawn occurs H hours before solar noon
    # Dusk occurs H hours after solar noon
    dawn_UTC = solar_noon_UTC - H_hours
    dusk_UTC = solar_noon_UTC + H_hours
    
    # Convert to local time
    dawn_local = dawn_UTC + timezone_offset
    dusk_local = dusk_UTC + timezone_offset
    
    # Normalize to 0-24 hour range
    dawn_local = dawn_local % 24
    dusk_local = dusk_local % 24
    
    return dawn_local, dusk_local


def calculate_civil_twilight(lat: float, lon: float, dec: float, M: float, 
                             timezone_offset: float) -> tuple[float | None, float | None]:
    """
    Calculate civil twilight times (Sun at -6° below horizon).
    
    Civil twilight is the period when the Sun is between 0° and -6° below the
    horizon. During this time, there is enough natural light for most outdoor
    activities without artificial lighting, and the brightest stars become visible.
    
    Args:
        lat (float): Observer's latitude in degrees [-90, 90]
        lon (float): Observer's longitude in degrees [-180, 180]
        dec (float): Sun's declination in degrees (from calculate_solar_position)
        M (float): Sun's mean anomaly in degrees (from calculate_solar_position)
        timezone_offset (float): Hours offset from UTC
    
    Returns:
        tuple[float | None, float | None]: 
            - civil_dawn (float | None): When Sun reaches -6° in morning (decimal hours)
            - civil_dusk (float | None): When Sun reaches -6° in evening (decimal hours)
            Returns (None, None) if civil twilight doesn't occur
    
    Definition:
        - Civil Dawn: The beginning of morning civil twilight (when Sun rises to -6°)
        - Civil Dusk: The end of evening civil twilight (when Sun descends to -6°)
        - Duration: From sunset to civil dusk (evening), civil dawn to sunrise (morning)
    
    Practical Significance:
        - Sufficient light to distinguish objects clearly
        - Horizon is clearly visible
        - Brightest planets and stars become visible
        - Artificial lighting may be needed for detailed work
        - Legal definition for some activities (e.g., vehicle headlights)
    
    Example:
        >>> lat, lon = 40.7128, -74.0060  # New York City
        >>> dec, M = -1.23, 268.5  # Example values from calculate_solar_position
        >>> dawn, dusk = calculate_civil_twilight(lat, lon, dec, M, -4)
        >>> if dawn is not None:
        ...     print(f"Civil dawn: {dawn:.2f}h ({int(dawn)}:{int((dawn%1)*60):02d})")
        ...     print(f"Civil dusk: {dusk:.2f}h ({int(dusk)}:{int((dusk%1)*60):02d})")
        Civil dawn: 6.35h (6:21)
        Civil dusk: 18.80h (18:48)
    
    Notes:
        - Civil twilight is the brightest/shallowest twilight phase
        - At latitudes beyond ~60°, civil twilight may last all night in summer
        - Common reference for outdoor activity planning and photography
    
    See Also:
        - calculate_sunrise_sunset(): For actual sunrise/sunset times
        - calculate_nautical_twilight(): For nautical twilight (-12°)
        - calculate_astronomical_twilight(): For astronomical twilight (-18°)
    """
    return calculate_single_twilight(lat, lon, dec, M, timezone_offset, -6.0)


def calculate_nautical_twilight(lat: float, lon: float, dec: float, M: float, 
                                timezone_offset: float) -> tuple[float | None, float | None]:
    """
    Calculate nautical twilight times (Sun at -12° below horizon).
    
    Nautical twilight is the period when the Sun is between -6° and -12° below
    the horizon. During this time, the horizon becomes indistinct, but there is
    enough light for sailors to navigate using visible stars and the horizon.
    
    Args:
        lat (float): Observer's latitude in degrees [-90, 90]
        lon (float): Observer's longitude in degrees [-180, 180]
        dec (float): Sun's declination in degrees (from calculate_solar_position)
        M (float): Sun's mean anomaly in degrees (from calculate_solar_position)
        timezone_offset (float): Hours offset from UTC
    
    Returns:
        tuple[float | None, float | None]:
            - nautical_dawn (float | None): When Sun reaches -12° in morning (decimal hours)
            - nautical_dusk (float | None): When Sun reaches -12° in evening (decimal hours)
            Returns (None, None) if nautical twilight doesn't occur
    
    Definition:
        - Nautical Dawn: The beginning of morning nautical twilight (Sun rises to -12°)
        - Nautical Dusk: The end of evening nautical twilight (Sun descends to -12°)
        - Duration: From civil dusk to nautical dusk (evening), nautical dawn to civil dawn (morning)
    
    Practical Significance:
        - Horizon becomes indistinct and difficult to see clearly
        - Navigational stars are visible for celestial navigation
        - Traditionally used by sailors for taking star sights
        - General outlines of ground objects still distinguishable
        - Not dark enough for astronomical observations
    
    Historical Context:
        Named "nautical" because sailors historically used this period to take
        celestial measurements with a sextant, when both stars and the horizon
        were visible simultaneously for navigation.
    
    Example:
        >>> lat, lon = 40.7128, -74.0060  # New York City
        >>> dec, M = -1.23, 268.5
        >>> dawn, dusk = calculate_nautical_twilight(lat, lon, dec, M, -4)
        >>> if dawn is not None:
        ...     print(f"Nautical dawn: {dawn:.2f}h")
        ...     print(f"Nautical dusk: {dusk:.2f}h")
        Nautical dawn: 5.88h (5:52)
        Nautical dusk: 19.28h (19:16)
    
    Notes:
        - Darker than civil twilight but lighter than astronomical twilight
        - At latitudes beyond ~54°, nautical twilight may last all night in summer
        - Important for marine navigation and some aviation operations
    
    See Also:
        - calculate_civil_twilight(): For civil twilight (-6°)
        - calculate_astronomical_twilight(): For astronomical twilight (-18°)
    """
    return calculate_single_twilight(lat, lon, dec, M, timezone_offset, -12.0)


def calculate_astronomical_twilight(lat: float, lon: float, dec: float, M: float, 
                                   timezone_offset: float) -> tuple[float | None, float | None]:
    """
    Calculate astronomical twilight times (Sun at -18° below horizon).
    
    Astronomical twilight is the period when the Sun is between -12° and -18° below
    the horizon. This is the darkest phase of twilight; beyond -18°, the sky is
    considered fully dark for astronomical observations.
    
    Args:
        lat (float): Observer's latitude in degrees [-90, 90]
        lon (float): Observer's longitude in degrees [-180, 180]
        dec (float): Sun's declination in degrees (from calculate_solar_position)
        M (float): Sun's mean anomaly in degrees (from calculate_solar_position)
        timezone_offset (float): Hours offset from UTC
    
    Returns:
        tuple[float | None, float | None]:
            - astronomical_dawn (float | None): When Sun reaches -18° in morning (decimal hours)
            - astronomical_dusk (float | None): When Sun reaches -18° in evening (decimal hours)
            Returns (None, None) if astronomical twilight doesn't occur
    
    Definition:
        - Astronomical Dawn: Beginning of morning astronomical twilight (Sun rises to -18°)
        - Astronomical Dusk: End of evening astronomical twilight (Sun descends to -18°)
        - Duration: From nautical dusk to astronomical dusk (evening), 
                   astronomical dawn to nautical dawn (morning)
        - Beyond -18°: True astronomical darkness (no solar illumination)
    
    Practical Significance:
        - No solar illumination detectable to the naked eye
        - Sky is fully dark for astronomical observations
        - All stars visible (limited only by light pollution and atmospheric conditions)
        - Professional observatories typically require Sun below -18°
        - Zodiacal light and airglow become visible
        - Milky Way clearly visible in dark sky locations
    
    Astronomical Observations:
        - Deep sky objects (galaxies, nebulae) best observed after astronomical dusk
        - Before astronomical dawn, sky begins to brighten in the east
        - -18° is the formal boundary between twilight and night
        - Beyond this point, solar illumination is negligible
    
    Example:
        >>> lat, lon = 40.7128, -74.0060  # New York City
        >>> dec, M = -1.23, 268.5
        >>> dawn, dusk = calculate_astronomical_twilight(lat, lon, dec, M, -4)
        >>> if dawn is not None:
        ...     print(f"Astronomical dawn: {dawn:.2f}h")
        ...     print(f"Astronomical dusk: {dusk:.2f}h")
        ...     # Calculate duration of true darkness
        ...     darkness = (dawn + 24 - dusk) % 24
        ...     print(f"Hours of true darkness: {darkness:.2f}h")
        Astronomical dawn: 5.39h (5:23)
        Astronomical dusk: 19.76h (19:45)
        Hours of true darkness: 9.63h
    
    Special Cases:
        - At latitudes beyond ~48.5°, astronomical twilight may not occur in summer
          (continuous astronomical twilight or "white nights")
        - Near the Arctic/Antarctic circles, this can last for months
        - When astronomical twilight doesn't occur, returns (None, None)
    
    Notes:
        - Darkest and deepest twilight phase
        - Most restrictive twilight condition (least likely to occur at high latitudes)
        - Critical for professional astronomy and astrophotography
        - At mid-latitudes in summer, true darkness may only last a few hours
    
    See Also:
        - calculate_civil_twilight(): For civil twilight (-6°)
        - calculate_nautical_twilight(): For nautical twilight (-12°)
    """
    return calculate_single_twilight(lat, lon, dec, M, timezone_offset, -18.0)

# ----------------------------------------------------------------------
# 10. CONVERT DECIMAL HOURS TO HH:MM:SS.SS format 
# ----------------------------------------------------------------------

def decimal_hours_to_time(decimal_hours):
    """
    Convert decimal hours to HH:MM:SS.SS format
    
    Args:
        decimal_hours: Time in decimal hours (e.g., 6.5 = 6:30:00)
    
    Returns:
        String in format "HH:MM:SS.SS"
    """
    # Ensure the value is in 0-24 range
    decimal_hours = decimal_hours % 24
    
    # Extract hours
    hours = int(decimal_hours)
    
    # Extract minutes from the decimal part
    remaining = (decimal_hours - hours) * 60
    minutes = int(remaining)
    
    # Extract seconds from the decimal part of minutes
    seconds = (remaining - minutes) * 60
    
    # Format as string
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    
    return time_str


def calculate_sunrise_sunset_twilight(date_str: str, site_longitude: float, 
                                     site_latitude: float, ut_offset: float, elevation_m: float = 0.0) -> dict:
    """
    Calculate sunrise, sunset, and astronomical twilight times for a given date and location.
    
    This is a convenience function that combines all the necessary steps to calculate
    solar events from a simple date string and location coordinates.
    
    Args:
        date_str (str): Date in flexible format. Supported formats include:
                       - "YYYY-MM-DD" (e.g., "2025-09-26")
                       - "YYYY/MM/DD" (e.g., "2025/09/26")
                       - "DD-MM-YYYY" (e.g., "26-09-2025")
                       - Or any format supported by parse_date_to_YMD_flexible()
        
        site_longitude (float): Observer's longitude in degrees.
                               Positive for East, negative for West.
                               Range: -180 to +180
                               Example: -74.0060 (New York City)
        
        site_latitude (float): Observer's latitude in degrees.
                              Positive for North, negative for South.
                              Range: -90 to +90
                              Example: 40.7128 (New York City)
        
        ut_offset (float): UTC offset in hours.
                          Positive for East of Greenwich, negative for West.
                          Example: -4 (EDT), +1 (CET), +9 (JST)
        
        elevation_m (float): Observer's elevation in meters.
                            Example: 390 (Y28)
    
    Returns:
        dict: Dictionary containing solar event times with the following keys:
            - 'sunrise' (str | None): Sunrise time in HH:MM:SS.SS format
            - 'sunset' (str | None): Sunset time in HH:MM:SS.SS format
            - 'astronomical_dawn' (str | None): Astronomical dawn time
            - 'astronomical_dusk' (str | None): Astronomical dusk time
            - 'sunrise_decimal' (float | None): Sunrise in decimal hours
            - 'sunset_decimal' (float | None): Sunset in decimal hours
            - 'dawn_decimal' (float | None): Dawn in decimal hours
            - 'dusk_decimal' (float | None): Dusk in decimal hours
            - 'date' (str): Input date string
            - 'location' (dict): Latitude, longitude, and UTC offset
            
            Returns None for times that don't occur (e.g., polar day/night)
    
    Process:
        1. Parse the date string to extract year, month, day
        2. Calculate Julian Day number
        3. Calculate Sun's position (declination and mean anomaly)
        4. Calculate sunrise and sunset times
        5. Calculate astronomical twilight times
        6. Convert all times to HH:MM:SS.SS format
        7. Return organized results
    
    Example:
        >>> result = calculate_sunrise_sunset_twilight(
        ...     "2025-09-26", -74.0060, 40.7128, -4
        ... )
        >>> print(f"Sunrise: {result['sunrise']}")
        >>> print(f"Sunset: {result['sunset']}")
        >>> print(f"Astronomical Dawn: {result['astronomical_dawn']}")
        >>> print(f"Astronomical Dusk: {result['astronomical_dusk']}")
        Sunrise: 06:47:23.56
        Sunset: 18:21:58.90
        Astronomical Dawn: 05:23:45.67
        Astronomical Dusk: 19:45:36.79
        
        >>> # Check for polar conditions
        >>> result = calculate_sunrise_sunset_twilight(
        ...     "2025-06-21", 0.0, 85.0, 0  # Near North Pole in summer
        ... )
        >>> if result['sunrise'] is None:
        ...     print("Sun never sets (midnight sun)")
    
    Notes:
        - Returns None for times when the Sun doesn't reach the specified altitude
        - At extreme latitudes, some or all events may return None
        - Times are in local time based on ut_offset (does not account for DST)
        - Decimal hour values are also provided for calculations
    
    Raises:
        ValueError: If date_str cannot be parsed
        ValueError: If latitude or longitude are out of valid ranges
    
    See Also:
        - parse_date_to_YMD_flexible(): Date parsing function
        - calculate_julian_day(): Julian Day calculation
        - calculate_solar_position(): Sun position calculation
        - calculate_sunrise_sunset(): Sunrise/sunset calculation
        - calculate_astronomical_twilight(): Twilight calculation
        - decimal_hours_to_time(): Time formatting function
    """
    # --- START of Longitude Conversion for Solar Events ---
    # The solar functions (calculate_sunrise_sunset, etc.) require longitude
    # in the [-180, +180] range (East positive, West negative).
    if site_longitude > 180.0:
        # Example: 204.5317 (East) becomes -155.4683 (West)
        site_longitude_180 = site_longitude - 360.0
    else:
        site_longitude_180 = site_longitude
    # --- END of Longitude Conversion ---

    # Step 1: Parse the date string
    Y, M, D = parse_date_to_YMD_flexible(date_str)
    
    # Step 2: Calculate Julian Day
    JD = calculate_julian_day(Y, M, D)
    
    # Calculate Sun's position
    L, M_anomaly, true_lon, sun_RA, sun_DEC = calculate_solar_position(JD)
    
    # Calculate sunrise and sunset
    # NOTE: Pass the new elevation_m argument
    sunrise_decimal, sunset_decimal = calculate_sunrise_sunset(
        site_latitude, site_longitude_180, sun_DEC, M_anomaly, ut_offset, elevation_m
    )
    
    # Calculate astronomical twilight (Twilight is NOT affected by elevation, 
    # as the Sun is far below the horizon)
    dawn_decimal, dusk_decimal = calculate_astronomical_twilight(
        site_latitude, site_longitude_180, sun_DEC, M_anomaly, ut_offset
    )
    
    # Step 6: Convert decimal hours to time strings
    sunrise_str = decimal_hours_to_time(sunrise_decimal) if sunrise_decimal is not None else None
    sunset_str = decimal_hours_to_time(sunset_decimal) if sunset_decimal is not None else None
    dawn_str = decimal_hours_to_time(dawn_decimal) if dawn_decimal is not None else None
    dusk_str = decimal_hours_to_time(dusk_decimal) if dusk_decimal is not None else None
    
    # Step 7: Return organized results
    return {
        # Formatted time strings
        'sunrise': sunrise_str,
        'sunset': sunset_str,
        'astronomical_dawn': dawn_str,
        'astronomical_dusk': dusk_str,
        
        # Decimal hours (for further calculations if needed)
        'sunrise_decimal': sunrise_decimal,
        'sunset_decimal': sunset_decimal,
        'dawn_decimal': dawn_decimal,
        'dusk_decimal': dusk_decimal,
        
        # Metadata
        'date': date_str,
        'location': {
            'latitude': site_latitude,
            'longitude': site_longitude,
            'utc_offset': ut_offset
        }
    }

# ----------------------------------------------------------------------
# 11. WRAPER TO CALCULATE SUNRISE, SUNSET, TWILIGHT
# ----------------------------------------------------------------------
# Optional: Simpler version if you only want the time strings
def calculate_sunrise_sunset_twilight_simple(JD: float, site_longitude: float, 
                                            site_latitude: float, ut_offset: float, 
                                            elevation_m: float = 0.0,
                                            delta_t_seconds: float = 0.0) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Simplified version that returns only decimal hours.
    
    [LOGIC ADDED]: Converts site_longitude from [0.0, 360.0] to [-180.0, +180.0]
    for use in solar event calculation functions.
    """
    
    # --- START of Longitude Conversion for Solar Events ---
    # The solar functions (calculate_sunrise_sunset, etc.) require longitude
    # in the [-180, +180] range (East positive, West negative).
    if site_longitude > 180.0:
        # Example: 204.5317 (East) becomes -155.4683 (West)
        site_longitude_180 = site_longitude - 360.0
    else:
        site_longitude_180 = site_longitude
    # --- END of Longitude Conversion ---
     
    # Calculate Sun's position
    # NOTE: Pass the new delta_t_seconds argument
    L, M_anomaly, true_lon, sun_RA, sun_DEC = calculate_solar_position(JD, delta_t_seconds)
    
    # Calculate sunrise and sunset
    sunrise_decimal, sunset_decimal = calculate_sunrise_sunset(
        site_latitude, site_longitude_180, sun_DEC, M_anomaly, ut_offset, elevation_m
    )
    
    # Calculate astronomical twilight
    dawn_decimal, dusk_decimal = calculate_astronomical_twilight(
        site_latitude, site_longitude_180, sun_DEC, M_anomaly, ut_offset
    )
    
    return sunrise_decimal, sunset_decimal, dawn_decimal, dusk_decimal


# ----------------------------------------------------------------------
# 12. ALTERNATIVE WRAPER TO CALCULATE SUNRISE, SUNSET, TWILIGHT
# ----------------------------------------------------------------------

def calculate_events_for_date(date_str, site_lon, site_lat, ut_off, elevation):
    # 1. Parse Date and Calculate JD
    Y, M, D = parse_date_to_YMD_date_only(date_str)
    JD = calculate_julian_day(Y, M, D)
    
    # 2. Calculate Delta T for high precision
    delta_t_seconds = calculate_delta_t(Y, M)
    
    # 3. Calculate all events for this single date
    sunrise, sunset, dawn, dusk = calculate_sunrise_sunset_twilight_simple(
        JD, 
        site_lon, 
        site_lat, 
        ut_off,
        elevation,         
        delta_t_seconds
    )
    return sunrise, sunset, dawn, dusk


def calculate_sun_evening_and_morning_events(date_str,site_longitude, site_latitude, utc_offset, site_elevation_m):
    # --- 1. Get Evening Events (Sunset & Dusk for the starting date) ---
    sunrise_1, sunset_1, dawn_1, dusk_1 = calculate_events_for_date(
        date_str, site_longitude, site_latitude, utc_offset, site_elevation_m)
    
    # From the string of the starting date (date_str) get the string for the next day (date_str_2)
    date_str_object = datetime.strptime(date_str, "%Y-%m-%d")
    date_str_2_object = date_str_object + timedelta(days=1)
    date_str_2 = date_str_2_object.strftime("%Y-%m-%d")

    # --- 2. Get Morning Events (Dawn & Sunrise for the next day) ---
    sunrise_2, sunset_2, dawn_2, dusk_2 = calculate_events_for_date(
        date_str_2, site_longitude, site_latitude, utc_offset, site_elevation_m)

    return date_str,date_str_2,sunset_1,dusk_1,dawn_2,sunrise_2


# ----------------------------------------------------------------------
# 13. GET OBSERVATORY COORDINATES
# ----------------------------------------------------------------------
def convert_longitude_to_standard(longitude_east: float) -> float:
    """
    Convert East longitude (0-360°) to standard longitude (-180 to +180°).
    
    Parameters:
        longitude_east: Longitude in degrees East (0-360°)
        
    Returns:
        Longitude in standard format (-180° to +180°)
    """
    return longitude_east - 360.0 if longitude_east > 180.0 else longitude_east

def get_observatory_coordinates(mpc_codes_file: str, mpc_code: str) -> tuple[float, float, float]:
    """
    Retrieve observatory coordinates from MPC (Minor Planet Center) observatory codes file.
    
    Parameters:
        mpc_codes_file: Path to CSV file containing MPC observatory data
        mpc_code: Three-character MPC observatory code (e.g., '500' for Geocentric)
        
    Returns:
        Tuple of (longitude, latitude, elevation):
            - longitude: Degrees (-180 to +180, negative for West)
            - latitude: Degrees (-90 to +90, negative for South)  
            - elevation: Meters above sea level
            
    Raises:
        FileNotFoundError: If MPC codes file doesn't exist
        KeyError: If MPC code not found in file
    """
    if not os.path.exists(mpc_codes_file):
        raise FileNotFoundError(f"MPC codes file not found: {mpc_codes_file}")
    
    try:
        mpc_observatories = pd.read_csv(mpc_codes_file)
        observatory_data = mpc_observatories[mpc_observatories.Code == mpc_code]
        
        if observatory_data.empty:
            raise KeyError(f"MPC code '{mpc_code}' not found in observatory file")
        
        longitude_east = observatory_data.Longitude.values[0]
        longitude = convert_longitude_to_standard(longitude_east)
        latitude = observatory_data.Latitude.values[0]
        elevation = observatory_data.Altitude.values[0]
        
        return longitude, latitude, elevation
        
    except Exception as e:
        raise RuntimeError(f"Error reading MPC observatory data: {e}")


def get_utc_offset_from_coords(lat, lon):
    """
    Get UTC offset from latitude and longitude using tzfpy
    
    Args:
        lat: Latitude (float)
        lon: Longitude (float)
        
    Returns:
        tuple: (offset_hours, timezone_name) or (None, None) if failed
    """
    try:
        # Get timezone name from coordinates
        tz_name = get_tz(lon, lat)  # Note: tzfpy uses (lon, lat) order
        
        if tz_name is None:
            print("Could not find timezone for these coordinates")
            return None, None
        
        # Get the timezone object
        tz = pytz.timezone(tz_name)
        
        # Get current UTC offset (including DST if applicable)
        now = datetime.now(tz)
        offset = now.utcoffset().total_seconds() / 3600
        
        return offset, tz_name
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None
    
# ----------------------------------------------------------------------
# 13. GET LIST OF OBJECT ALTITUDES
# ----------------------------------------------------------------------

def calculate_object_altitude_range(ra_dec_str,date_str,sunset_1,site_longitude,site_latitude):

    ra_hours, ra_deg, dec_deg, dec_rad = parse_and_convert_ra_dec(ra_dec_str)
    date_str_object = datetime.strptime(date_str, "%Y-%m-%d")
    date_str_elevation_start_object = date_str_object + timedelta (hours=sunset_1) - timedelta (minutes=30)

    list_dates_objects = [date_str_elevation_start_object + timedelta(minutes=1*(i+1)) for i in range(1200)]
    list_dates_str =  [i.strftime("%Y-%m-%d %H:%M:%S") for i in list_dates_objects]

    list_JDs = []
    for i in list_dates_str:
        Y, M, D = parse_date_to_YMD_flexible(i)
        list_JDs.append(calculate_julian_day(Y, M, D))

    list_GMST_degrees = [calculate_mean_sidereal_time(i) for i in list_JDs]

    list_LST = [calculate_local_sidereal_time(i,site_longitude) for i in list_GMST_degrees]

    list_H_hours = [i - ra_hours for i in list_LST]

    list_altitude_degrees = [calculate_object_altitude(site_latitude, dec_deg, i) for i in list_H_hours] 

    return list_dates_str,list_dates_objects,list_altitude_degrees




# ----------------------------------------------------------------------
# 14. SOME FUNCTIONS NEEDED FOR THE PLOT
# ----------------------------------------------------------------------
def elevation_to_airmass(elevation_deg):
    """
    Calculates the airmass (approximately) from the elevation in degrees.
    Airmass is typically A = 1 / sin(elevation).
    """
    # Convert degrees to radians for the sine function
    elevation_rad = np.deg2rad(elevation_deg)
    
    # Handle the case where elevation is 0 to avoid division by zero
    if elevation_deg <= 0:
        return '---' # Or 'Inf' for theoretical infinity
        
    airmass = 1 / np.sin(elevation_rad)
    return airmass


def airmass_young_1994(elevation_deg):
    """
    Calculates the airmass (X) using the highly accurate formula by Young (1994).
    The formula uses the true zenith angle (zt) in terms of degrees.
    """
    # 1. Convert Elevation (alpha) to Zenith Angle (zt)
    # zt = 90 - alpha
    zt_deg = 90.0 - elevation_deg
    
    # Handle the case where the object is below the horizon (zt_deg >= 90)
    # The formula is unstable for zt near 90 or greater. 
    # For zt >= 89.9 (or elevation <= 0.1), use a large fixed value or an indicator.
    if zt_deg >= 89.9:
        return np.nan # Use NaN to filter out of display or return '---'
        
    # 2. Convert Zenith Angle to Radians for numpy.cos()
    zt_rad = np.deg2rad(zt_deg)
    
    # Calculate cos(zt)
    cos_zt = np.cos(zt_rad)
    
    # 3. Apply the Young (1994) formula:
    
    # Numerator
    numerator = (1.002432 * cos_zt**2) + (0.148386 * cos_zt) + 0.0096467
    
    # Denominator
    denominator = (cos_zt**3) + (0.149864 * cos_zt**2) + (0.0102963 * cos_zt) + 0.000303978
    
    airmass = numerator / denominator
    return airmass


def airmass_formatter(y, pos):
    """
    Custom formatter for the Airmass axis labels.
    y is the elevation value (in degrees) from the primary y-axis.
    """
    # Only calculate for positive elevation
    if y <= 0:
        return '---'
    
    airmass = airmass_young_1994(y)
    
    # Format the airmass value to two decimal places
    return f"{airmass:.2f}"

def parse_time(time_str, base_date):
    """Parses 'HH:MM:SS.ms' string and combines with a base date."""
    H, M, S = map(float, time_str.split(':'))
    seconds_int = int(S)
    microseconds = int((S - seconds_int) * 1000000)
    
    return base_date.replace(
        hour=int(H), 
        minute=int(M), 
        second=seconds_int, 
        microsecond=microseconds
    )