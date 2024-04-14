import pandas as pd
import re

'''
    Remove the common special characters
'''
def basicRemove(x: pd.Series, newline=True, tabs=True, links=True, specialCharacter=True, questionMark = False, exclamationMark = False, hyphen = False):
    res = pd.Series(dtype=str)

    # Crafting appropriate regex for basic removal
    regex = ""

    # Remove newline
    if newline:
        regex += "[\n]|"
    
    # Remove tabs
    if tabs:
        regex += "[ \t]|"

    # Remove special character
    if specialCharacter:
        exclusion = "[^a-zA-Z0-9"

        # Exclude ?
        if not questionMark:
            exclusion += "?"

        # Exclude !
        if not exclamationMark:
            exclusion += "!"
        
        if not hyphen:
            exclusion += "-"

        exclusion += "]"

        regex += exclusion
        
    res = x.apply(lambda line: re.sub(regex, " ", line))

    res = res.apply(lambda line: re.sub("[?]", " ?", line))
    res = res.apply(lambda line: re.sub("[!]", " !", line))

    # Replace single characters with a space
    res = res.apply(lambda line: re.sub(r'\b[a-zA-Z]\b', " ", line))

    # Replace all double spaces with one space
    res = res.apply(lambda line: re.sub(" +", " ", line))

    return res
