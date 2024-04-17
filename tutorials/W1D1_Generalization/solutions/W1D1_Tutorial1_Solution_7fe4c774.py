
def calculate_writing_time(total_words, words_per_day, days_per_week, weeks_per_year, average_human_lifespan):
    """
    Calculate the time required to write a given number of words in lifetimes.

    Inputs:
    - total_words (int): total number of words to be written.
    - words_per_day (int): number of words written per day.
    - days_per_week (int): number of days dedicated to writing per week.
    - weeks_per_year (int): number of weeks dedicated to writing per year.
    - average_human_lifespan (int): average lifespan of a human in years.

    Outpus:
    - time_to_write_lifetimes (float): time to write the given words in lifetimes.
    """

    words_per_year = words_per_day * days_per_week * weeks_per_year

    # Calculate the time to write in years
    time_to_write_years = total_words / words_per_year

    # Calculate the time to write in lifetimes
    time_to_write_lifetimes = time_to_write_years / average_human_lifespan

    return time_to_write_lifetimes