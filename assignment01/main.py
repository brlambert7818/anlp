@ author: Brian Lambert

import re


def preprocess_line(text):
    '''
    Takes line of text and returns string which removes all characters from line not
    in English alphabet, space, digits, or '.'. All characters are lowercased and all
    digits are converted to '0'.

    Parameters:
        text (str): single line of input text
    
    Returns:
        text_processed (str): single line of text with unwanted characters removed
    '''
    # lowercase all letters
    text = text.lower()
    # remove replace digits with 0
    text = re.sub('[1-9]', '0', text)
    # remove non-English alphabet, spaces, or .
    return re.sub('[^a-z0.\s]', '', text)
    

preprocess_line('This is a test 1 2 0 3 4 5!.')

    
