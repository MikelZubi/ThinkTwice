from difflib import SequenceMatcher


def maxCommStr(str1, str2):
    seqMatch = SequenceMatcher(None,str1,str2, autojunk=False)

    # find match of longest sub-string
    # output will be like Match(a=0, b=0, size=5)
    match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))

    # print longest substring
    if match.size == 0:
        return ""
    length = match.a + match.size
    return str1[match.a: length]
