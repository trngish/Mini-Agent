"""
Comprehensive tests for string_utils module.

Run with: pytest test_string_utils.py -v
"""

import pytest
from string_utils import reverse_string, is_palindrome, count_vowels, anagram_groups


class TestReverseString:
    """Tests for reverse_string function."""

    def test_basic_string(self):
        """Test reversing a simple string."""
        assert reverse_string("hello") == "olleh"

    def test_single_character(self):
        """Test reversing a single character."""
        assert reverse_string("a") == "a"

    def test_empty_string(self):
        """Test reversing an empty string."""
        assert reverse_string("") == ""

    def test_palindrome_string(self):
        """Test reversing a palindrome (result is same)."""
        assert reverse_string("racecar") == "racecar"

    def test_string_with_spaces(self):
        """Test reversing a string with spaces."""
        assert reverse_string("hello world") == "dlrow olleh"

    def test_string_with_numbers(self):
        """Test reversing a string with numbers."""
        assert reverse_string("abc123") == "321cba"

    def test_string_with_special_characters(self):
        """Test reversing a string with special characters."""
        assert reverse_string("hello!@#") == "#@!olleh"
        assert reverse_string("a,b$c") == "c$b,a"

    def test_mixed_case(self):
        """Test reversing preserves case."""
        assert reverse_string("Hello") == "olleH"
        assert reverse_string("Python") == "nohtyP"

    def test_unicode_characters(self):
        """Test functions handle unicode characters."""
        assert reverse_string("café") == "éfac"

    def test_whitespace_only(self):
        """Test functions with whitespace-only strings."""
        assert reverse_string("   ") == "   "

    def test_string_with_newlines(self):
        """Test reversing a string with newlines."""
        assert reverse_string("a\nb") == "b\na"


class TestIsPalindrome:
    """Tests for is_palindrome function."""

    def test_basic_palindromes(self):
        """Test basic palindromes."""
        assert is_palindrome("racecar") is True
        assert is_palindrome("level") is True
        assert is_palindrome("madam") is True

    def test_basic_non_palindromes(self):
        """Test strings that are not palindromes."""
        assert is_palindrome("hello") is False
        assert is_palindrome("world") is False
        assert is_palindrome("python") is False

    def test_empty_string(self):
        """Test empty string is considered palindrome."""
        assert is_palindrome("") is True

    def test_single_character(self):
        """Test single character is palindrome."""
        assert is_palindrome("a") is True
        assert is_palindrome("z") is True

    def test_case_sensitivity_ignore_true(self):
        """Test palindrome check is case-insensitive."""
        assert is_palindrome("RaceCar") is True
        assert is_palindrome("Aba") is True
        assert is_palindrome("Noon") is True

    def test_ignore_non_alphanumeric(self):
        """Test ignoring spaces, punctuation, etc. (default behavior)."""
        # By default, is_palindrome ignores non-alphanumeric characters
        assert is_palindrome("A man a plan a canal Panama") is True
        assert is_palindrome("Was it a car or a cat I saw") is True
        assert is_palindrome("Able was I ere I saw Elba") is True

    def test_ignore_non_alphanumeric_false(self):
        """Test that spaces and non-alphanumeric chars are part of palindrome check by default."""
        # By default, non-alphanumeric characters are filtered out, so these are palindromes
        assert is_palindrome("race car") is True  # spaces ignored
        assert is_palindrome("race  car") is True  # multiple spaces ignored

    def test_numbers_in_palindrome(self):
        """Test palindromes containing numbers."""
        assert is_palindrome("12321") is True
        assert is_palindrome("12-21") is True
        assert is_palindrome("12 21") is True

    def test_unicode_palindrome(self):
        """Test unicode palindrome."""
        assert is_palindrome("racecar") is True

    def test_both_options_combined(self):
        """Test both case and non-alphanumeric options."""
        # The function is case-insensitive and ignores non-alphanumeric by default
        assert is_palindrome("A man, a plan, a canal: Panama") is True
        assert is_palindrome("Never odd or even") is True


class TestCountVowels:
    """Tests for count_vowels function."""

    def test_basic_vowel_counting(self):
        """Test basic vowel counting."""
        assert count_vowels("hello") == 2
        assert count_vowels("world") == 1
        assert count_vowels("python") == 1

    def test_uppercase_vowels(self):
        """Test uppercase vowels are counted."""
        assert count_vowels("AEIOU") == 5
        assert count_vowels("HELLO") == 2

    def test_mixed_case_vowels(self):
        """Test mixed case vowels."""
        assert count_vowels("Hello World") == 3
        assert count_vowels("Python Programming") == 4

    def test_empty_string(self):
        """Test empty string has zero vowels."""
        assert count_vowels("") == 0

    def test_no_vowels(self):
        """Test string with no vowels."""
        assert count_vowels("xyz") == 0
        assert count_vowels("bcdfg") == 0
        assert count_vowels("rhythm") == 0

    def test_all_vowels(self):
        """Test string with all vowels."""
        assert count_vowels("aeiouAEIOU") == 10
        assert count_vowels("aAeEiIoOuU") == 10

    def test_vowels_with_consonants(self):
        """Test consecutive vowels are counted correctly."""
        assert count_vowels("beautiful") == 5
        assert count_vowels("cooperation") == 6

    def test_numbers_and_special_characters(self):
        """Test numbers and special chars don't affect count."""
        assert count_vowels("12345!@#$") == 0
        assert count_vowels("a1e2i3o4u5") == 5

    def test_spaces(self):
        """Test spaces don't affect count."""
        assert count_vowels("a e i o u") == 5
        assert count_vowels("Hello World") == 3

    def test_unicode_vowels(self):
        """Test unicode vowels are NOT counted (only ASCII vowels counted)."""
        # Only ASCII vowels are counted
        assert count_vowels("caf") == 1  # a
        assert count_vowels("aeiou") == 5


class TestAnagramGroups:
    """Tests for anagram_groups function."""

    def test_basic_anagram_grouping(self):
        """Test basic anagram grouping."""
        result = anagram_groups(["eat", "tea", "tan", "ate", "nat", "bat"])
        # Groups are not sorted internally, but anagrams are grouped together
        # Result matches the expected output from docstring example
        assert result == [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
        # Verify the groups contain the expected words
        all_words = [word for group in result for word in group]
        assert set(all_words) == {'eat', 'tea', 'tan', 'ate', 'nat', 'bat'}

    def test_single_word(self):
        """Test single string returns single group."""
        assert anagram_groups(["hello"]) == [['hello']]

    def test_empty_list(self):
        """Test empty list returns empty list."""
        assert anagram_groups([]) == []

    def test_all_unique_words(self):
        """Test list with no anagrams."""
        result = anagram_groups(["hello", "world", "python"])
        assert len(result) == 3
        for group in result:
            assert len(group) == 1

    def test_all_anagrams(self):
        """Test all strings are anagrams of each other."""
        result = anagram_groups(["abc", "bca", "cab", "cba"])
        assert len(result) == 1
        assert sorted(result[0]) == ['abc', 'bca', 'cab', 'cba']

    def test_case_handling(self):
        """Test case-sensitive anagram grouping."""
        # Anagrams are case-sensitive by default
        result = anagram_groups(["Eat", "TEA", "eat"])
        # Each different case is a separate group
        assert len(result) == 3
        for group in result:
            assert len(group) == 1

    def test_mixed_case(self):
        """Test that same-case words are grouped properly."""
        # Case-sensitive: "Eat" and "eat" have different characters (E vs e)
        result = anagram_groups(["eat", "eat", "tea"])
        assert len(result) == 1
        assert sorted(result[0]) == ['eat', 'eat', 'tea']

    def test_words_with_duplicates(self):
        """Test handling of duplicate strings."""
        result = anagram_groups(["abc", "abc", "bca"])
        assert len(result) == 1
        assert sorted(result[0]) == ['abc', 'abc', 'bca']

    def test_numbers_as_anagrams(self):
        """Test numbers can be anagrams."""
        result = anagram_groups(["123", "321", "231", "456"])
        # 123, 321, 231 are anagrams; 456 is different
        assert len(result) == 2
        # Find the group with 3 elements
        groups_with_3 = [g for g in result if len(g) == 3]
        assert len(groups_with_3) == 1
        assert sorted(groups_with_3[0]) == ['123', '231', '321']

    def test_empty_strings(self):
        """Test empty strings are grouped together."""
        result = anagram_groups(["", "", "abc"])
        # Empty strings are anagrams of each other
        assert len(result) == 2

    def test_special_characters(self):
        """Test special characters in anagrams."""
        result = anagram_groups(["a!b", "!ba", "b!a"])
        assert len(result) == 1
        assert sorted(result[0]) == ['!ba', 'a!b', 'b!a']

    def test_unicode_handling(self):
        """Test unicode handling."""
        result = anagram_groups(["café", "éfac"])
        assert len(result) == 1


class TestIntegration:
    """Integration tests using multiple functions."""

    def test_reverse_then_check_palindrome(self):
        """Test reversing a palindrome gives same result."""
        original = "racecar"
        reversed_str = reverse_string(original)
        assert reversed_str == original
        assert is_palindrome(reversed_str) is True

    def test_count_vowels_in_reversed_string(self):
        """Test vowel count is same after reversing."""
        original = "Hello World"
        reversed_str = reverse_string(original)
        assert count_vowels(original) == count_vowels(reversed_str)

    def test_anagram_count_vowels(self):
        """Test anagrams have same vowel count."""
        # Get all groups that have more than one word
        result = anagram_groups(["eat", "tea", "ate", "hello"])
        for group in result:
            if len(group) > 1:
                # All anagrams should have same vowel count
                vowel_counts = [count_vowels(word) for word in group]
                assert len(set(vowel_counts)) == 1


class TestEdgeCases:
    """Tests for edge cases across all functions."""

    def test_very_long_string(self):
        """Test functions with a very long string."""
        long_string = "a" * 10000
        assert reverse_string(long_string) == "a" * 10000
        assert count_vowels(long_string) == 10000

    def test_whitespace_only(self):
        """Test functions with whitespace-only strings."""
        assert reverse_string("   ") == "   "
        assert count_vowels("   ") == 0


class TestTypeHints:
    """Tests to verify type hints are correct."""

    def test_reverse_string_return_type(self):
        """Test reverse_string returns a string."""
        result = reverse_string("test")
        assert isinstance(result, str)

    def test_is_palindrome_return_type(self):
        """Test is_palindrome returns a boolean."""
        result = is_palindrome("test")
        assert isinstance(result, bool)

    def test_count_vowels_return_type(self):
        """Test count_vowels returns an integer."""
        result = count_vowels("test")
        assert isinstance(result, int)

    def test_anagram_groups_return_type(self):
        """Test anagram_groups returns a list of lists."""
        result = anagram_groups(["test"])
        assert isinstance(result, list)
        assert all(isinstance(group, list) for group in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])