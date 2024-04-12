
def clean_string(input_string):
    """
    Clean string prior to comparison

    Args:
        input_string (str): the input string

    Returns:
        (str) a cleaned string, lowercase, alphabetical characters only, no double spaces
    """

    # Convert all characters to lowercase
    lowercase_string = input_string.lower()

    # Remove non-alphabetic characters
    alpha_string = re.sub(r'[^a-z\s]', '', lowercase_string)

    # Remove double spaces and start and end spaces
    return re.sub(r'\s+', ' ', alpha_string).strip()


def calculate_mismatch(estimated_text, reference_text):
    """
    Calculate mismatch (character and word error rates) between estimated and true text.

    Args:
        estimated_text: a list of strings
        reference_text: a list of strings

    Returns:
        A tuple, (CER and WER)
    """
    # Lowercase the text and remove special characters for the comparison
    estimated_text = [clean_string(x) for x in estimated_text]
    reference_text = [clean_string(x) for x in reference_text]

    # Calculate the character error rate and word error rates. They should be
    # raw floats, not tensors.
    cer = fm.char_error_rate(estimated_text, reference_text).item()
    wer = fm.word_error_rate(estimated_text, reference_text).item()
    return (cer, wer)

cer, wer = calculate_mismatch(transcribed_text, true_transcripts)
assert isinstance(cer, float)
cer, wer