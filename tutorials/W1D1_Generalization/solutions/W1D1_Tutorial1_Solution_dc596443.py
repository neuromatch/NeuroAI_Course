def calculate_mean_max_cer(df_results):
    """
    Calculate the mean character-error-rate across subjects as
    well as the maximum (that is, the OOD risk).

    Args:
        df_results: a dataframe containing results

    Returns:
        A tuple, (mean_cer, max_cer)
    """
    # Calculate the mean CER across test subjects.
    mean_subjects = df_results.cer.mean()

    # Calculate the max CER across test subjects.
    max_subjects = df_results.cer.max()
    return mean_subjects, max_subjects

mean_subjects, max_subjects = calculate_mean_max_cer(df_results)
mean_subjects, max_subjects