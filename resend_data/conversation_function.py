def detect_intent_audio(project_id, session_id, audio_file_path,
                        language_code, sample_rate):
    """Returns the result of detect intent with an audio file as input.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    import dialogflow_v2 as dialogflow

    session_client = dialogflow.SessionsClient()

    # Note: hard coding audio_encoding and sample_rate_hertz for simplicity.
    audio_encoding = dialogflow.enums.AudioEncoding.AUDIO_ENCODING_LINEAR_16
    sample_rate_hertz = sample_rate

    session = session_client.session_path(project_id, session_id)

    with open(audio_file_path, 'rb') as audio_file:
        input_audio = audio_file.read()

    audio_config = dialogflow.types.InputAudioConfig(
        audio_encoding=audio_encoding, language_code=language_code,
        sample_rate_hertz=sample_rate_hertz)

    query_input = dialogflow.types.QueryInput(audio_config=audio_config)

    response = session_client.detect_intent(
        session=session, query_input=query_input,
        input_audio=input_audio)


    #print('Intent:', response.query_result.intent.display_name)
    #print('End:', response.queryResult)

    return response.query_result.fulfillment_text, response.query_result.query_text, response.query_result.intent.display_name, str(audio_file_path)




    #print(response.query_result.fulfillment_text)
    #print('=' * 20)
    #print('Query text: {}'.format(response.query_result.query_text))
    #print('Detected intent: {} (confidence: {})\n'.format(
    #    response.query_result.intent.display_name,
    #    response.query_result.intent_detection_confidence))
    #print('Fulfillment text: {}\n'.format(
    #    response.query_result.fulfillment_text))



