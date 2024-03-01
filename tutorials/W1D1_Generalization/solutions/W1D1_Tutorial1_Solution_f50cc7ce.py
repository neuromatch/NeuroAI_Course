
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

# Example values
total_words = 5e9
words_per_day = 1500
days_per_week = 6
weeks_per_year = 50
average_human_lifespan = 80

# Uncomment the code below to test your function

# Test the function
#time_to_write_lifetimes_roberta = calculate_writing_time(
    #total_words,
    #words_per_day,
    #days_per_week,
    #weeks_per_year,
    #average_human_lifespan
#)

# Print the result
#print(f"Time to write {total_words} words in lifetimes: {time_to_write_lifetimes_roberta} lifetimes")