def calculate_all_mismatch(df, model, processor):
    """
    Calculate CER and WER for all subjects in a dataset

    Args:
        df: a dataframe containing information about images and transcripts
        model: an image-to-text model
        processor: a processor object

    Returns:
        a list of dictionaries containing a per-subject breakdown of the
        results
    """
    subjects = df.subject.unique().tolist()

    results = []

    # Calculate CER and WER for all subjects
    for subject in tqdm.tqdm(subjects):
        # Load images and labels for a given subject
        images, true_transcripts = get_images_and_transcripts(df, subject)

        # Transcribe the images to text
        transcribed_text = transcribe_images(images, model, processor)

        # Calculate the CER and WER
        cer, wer = calculate_mismatch(transcribed_text, true_transcripts)

        results.append({
            'subject': subject,
            'cer': cer,
            'wer': wer,
        })
    return results