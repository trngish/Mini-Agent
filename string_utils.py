"""
String utility functions for common string operations.

This module provides functions for string manipulation including
reversing, palindrome checking, vowel counting, and anagram grouping.
"""

from collections import defaultdict
from typing import List


def reverse_string(s: str) -> str:
    """
    Reverse a given string.

    Args:
        s: The input string to reverse.

    Returns:
        The reversed string.

    Examples:
        >>> reverse_string("hello")
        'olleh'
        >>> reverse_string("Python")
        'nohtyP'
        >>> reverse_string("")
        ''
        >>> reverse_string("a")
        'a'
    """
    return s[::-1]


def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome.

    A palindrome reads the same forwards and backwards.
    The comparison is case-sensitive and ignores non-alphanumeric characters.

    Args:
        s: The input string to check.

    Returns:
        True if the string is a palindrome, False otherwise.

    Examples:
        >>> is_palindrome("racecar")
        True
        >>> is_palindrome("hello")
        False
        >>> is_palindrome("A man a plan a canal Panama")
        True
        >>> is_palindrome("")
        True
        >>> is_palindrome("a")
        True
    """
    # Filter to keep only alphanumeric characters and convert to lowercase
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]


def count_vowels(s: str) -> int:
    """
    Count the number of vowel letters in a string.

    Vowels are defined as: a, e, i, o, u (both uppercase and lowercase).

    Args:
        s: The input string to count vowels in.

    Returns:
        The number of vowel characters in the string.

    Examples:
        >>> count_vowels("hello")
        2
        >>> count_vowels("Python")
        1
        >>> count_vowels("AEIOU")
        5
        >>> count_vowels("rhythm")
        0
        >>> count_vowels("")
        0
    """
    vowels = set('aeiouAEIOU')
    return sum(1 for char in s if char in vowels)


def anagram_groups(strings: List[str]) -> List[List[str]]:
    """
    Group strings into anagram groups.

    Anagrams are strings that contain the same characters with the same frequency.
    Strings are grouped together if they are anagrams of each other.

    Args:
        strings: A list of strings to group into anagram groups.

    Returns:
        A list of lists, where each inner list contains strings that are anagrams
        of each other. Each group contains at least one string.

    Examples:
        >>> anagram_groups(["eat", "tea", "tan", "ate", "nat", "bat"])
        [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
        >>> anagram_groups(["abc", "def", "ghi"])
        [['abc'], ['def'], ['ghi']]
        >>> anagram_groups([""])
        [['']]
        >>> anagram_groups(["a"])
        [['a']]
    """
    anagram_map: dict = defaultdict(list)
    
    for s in strings:
        # Sort characters to create a canonical key for anagram grouping
        key = ''.join(sorted(s))
        anagram_map[key].append(s)
    
    return list(anagram_map.values())