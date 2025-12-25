#!/usr/bin/env python3
"""
Vocabulary Module - Frequency-based word lists with definitions.

Uses COCA (Corpus of Contemporary American English) frequency data
and NLTK WordNet for definitions.
"""

import csv
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
import urllib.request

# NLTK for definitions
try:
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# =============================================================================
# Data Paths
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
COCA_CSV_PATH = DATA_DIR / "COCA_WordFrequency.csv"
VOCAB_CACHE_PATH = DATA_DIR / "vocabulary_cache.json"

COCA_URL = "https://raw.githubusercontent.com/brucewlee/COCA-WordFrequency/main/COCA_WordFrequency.csv"


# =============================================================================
# POS Mapping
# =============================================================================

# COCA POS tags to WordNet POS
COCA_TO_WORDNET = {
    'n': 'n',    # noun
    'v': 'v',    # verb
    'j': 'a',    # adjective (COCA uses 'j', WordNet uses 'a')
    'r': 'r',    # adverb
}

# Content word POS (skip function words)
CONTENT_POS = {'n', 'v', 'j', 'r'}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VocabWord:
    """A vocabulary word with frequency and definition."""
    word: str
    pos: str  # n, v, j, r
    rank: int
    frequency: int
    definition: str
    inflections: List[str]

    def to_tuple(self) -> Tuple[str, str]:
        """Return (word, definition) tuple for prompts."""
        return (self.word, self.definition)


# =============================================================================
# COCA Data Loading
# =============================================================================

def download_coca_data() -> bool:
    """Download COCA frequency data if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if COCA_CSV_PATH.exists():
        return True

    print(f"Downloading COCA data to {COCA_CSV_PATH}...")
    try:
        urllib.request.urlretrieve(COCA_URL, COCA_CSV_PATH)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def load_coca_raw() -> List[Dict]:
    """Load raw COCA CSV data."""
    if not download_coca_data():
        return []

    words = []
    with open(COCA_CSV_PATH, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
        reader = csv.DictReader(f)
        for row in reader:
            words.append({
                'rank': int(row['rank']),
                'word': row['lemma'].lower(),
                'pos': row['PoS'].lower(),
                'freq': int(row['freq']),
            })

    return words


def filter_content_words(words: List[Dict], max_rank: int = 2000) -> List[Dict]:
    """Filter to content words within rank threshold."""
    filtered = []

    for w in words:
        # Skip if beyond rank threshold
        if w['rank'] > max_rank:
            continue

        # Skip function words
        if w['pos'] not in CONTENT_POS:
            continue

        # Skip single letters
        if len(w['word']) < 2:
            continue

        # Skip words with non-alpha characters
        if not w['word'].isalpha():
            continue

        filtered.append(w)

    return filtered


# =============================================================================
# WordNet Definitions
# =============================================================================

def get_wordnet_definition(word: str, pos: str) -> Optional[str]:
    """Get a simple definition from WordNet."""
    if not HAS_NLTK:
        return None

    wn_pos = COCA_TO_WORDNET.get(pos)
    if not wn_pos:
        return None

    synsets = wn.synsets(word, pos=wn_pos)
    if not synsets:
        # Try without POS filter
        synsets = wn.synsets(word)

    if not synsets:
        return None

    # Get the first (most common) definition
    definition = synsets[0].definition()

    # Clean up the definition
    definition = definition.lower()
    if len(definition) > 100:
        # Truncate long definitions
        definition = definition[:100].rsplit(' ', 1)[0] + "..."

    return definition


def get_simple_definition(word: str, pos: str) -> str:
    """Get or generate a simple definition."""
    # Try WordNet first
    defn = get_wordnet_definition(word, pos)
    if defn:
        return defn

    # Fallback to generic definitions by POS
    fallbacks = {
        'n': f"a type of thing or concept",
        'v': f"to do or perform an action",
        'j': f"a quality or characteristic",
        'r': f"in a certain way or manner",
    }
    return fallbacks.get(pos, "a word")


# =============================================================================
# Inflection Generation
# =============================================================================

# Common inflection patterns
VERB_INFLECTIONS = {
    'be': ['be', 'am', 'is', 'are', 'was', 'were', 'been', 'being'],
    'have': ['have', 'has', 'had', 'having'],
    'do': ['do', 'does', 'did', 'done', 'doing'],
    'go': ['go', 'goes', 'went', 'gone', 'going'],
    'say': ['say', 'says', 'said', 'saying'],
    'get': ['get', 'gets', 'got', 'gotten', 'getting'],
    'make': ['make', 'makes', 'made', 'making'],
    'know': ['know', 'knows', 'knew', 'known', 'knowing'],
    'think': ['think', 'thinks', 'thought', 'thinking'],
    'take': ['take', 'takes', 'took', 'taken', 'taking'],
    'see': ['see', 'sees', 'saw', 'seen', 'seeing'],
    'come': ['come', 'comes', 'came', 'coming'],
    'give': ['give', 'gives', 'gave', 'given', 'giving'],
    'find': ['find', 'finds', 'found', 'finding'],
    'tell': ['tell', 'tells', 'told', 'telling'],
    'leave': ['leave', 'leaves', 'left', 'leaving'],
    'feel': ['feel', 'feels', 'felt', 'feeling'],
    'put': ['put', 'puts', 'putting'],
    'bring': ['bring', 'brings', 'brought', 'bringing'],
    'keep': ['keep', 'keeps', 'kept', 'keeping'],
    'run': ['run', 'runs', 'ran', 'running'],
    'write': ['write', 'writes', 'wrote', 'written', 'writing'],
    'read': ['read', 'reads', 'reading'],
    'begin': ['begin', 'begins', 'began', 'begun', 'beginning'],
}

NOUN_PLURALS = {
    'man': ['man', 'men'],
    'woman': ['woman', 'women'],
    'child': ['child', 'children'],
    'person': ['person', 'people', 'persons'],
    'foot': ['foot', 'feet'],
    'tooth': ['tooth', 'teeth'],
    'mouse': ['mouse', 'mice'],
    'life': ['life', 'lives'],
    'wife': ['wife', 'wives'],
    'knife': ['knife', 'knives'],
}


def generate_inflections(word: str, pos: str) -> List[str]:
    """Generate common inflections for a word."""
    inflections = [word]

    if pos == 'v':
        # Check irregular verbs first
        if word in VERB_INFLECTIONS:
            return VERB_INFLECTIONS[word]

        # Regular verb patterns
        if word.endswith('e'):
            inflections.extend([word + 's', word + 'd', word[:-1] + 'ing'])
        elif word.endswith('y') and len(word) > 2 and word[-2] not in 'aeiou':
            inflections.extend([word[:-1] + 'ies', word[:-1] + 'ied', word + 'ing'])
        else:
            inflections.extend([word + 's', word + 'ed', word + 'ing'])

    elif pos == 'n':
        # Check irregular plurals first
        if word in NOUN_PLURALS:
            return NOUN_PLURALS[word]

        # Regular plural patterns
        if word.endswith(('s', 'x', 'z', 'ch', 'sh')):
            inflections.append(word + 'es')
        elif word.endswith('y') and len(word) > 2 and word[-2] not in 'aeiou':
            inflections.append(word[:-1] + 'ies')
        else:
            inflections.append(word + 's')

    elif pos == 'j':
        # Adjective comparatives/superlatives
        if len(word) <= 6:  # Short adjectives
            if word.endswith('e'):
                inflections.extend([word + 'r', word + 'st'])
            elif word.endswith('y'):
                inflections.extend([word[:-1] + 'ier', word[:-1] + 'iest'])
            else:
                inflections.extend([word + 'er', word + 'est'])
        # Adverb form
        if not word.endswith('ly'):
            inflections.append(word + 'ly')

    elif pos == 'r':
        # Adverbs - usually no inflections
        pass

    return list(set(inflections))


# =============================================================================
# Vocabulary Building
# =============================================================================

def build_vocabulary(max_rank: int = 2000, use_cache: bool = True) -> List[VocabWord]:
    """Build vocabulary list from COCA data."""

    # Check cache
    if use_cache and VOCAB_CACHE_PATH.exists():
        try:
            with open(VOCAB_CACHE_PATH, 'r') as f:
                data = json.load(f)
                if data.get('max_rank') >= max_rank:
                    return [VocabWord(**w) for w in data['words'][:max_rank]]
        except Exception:
            pass

    # Load and filter COCA data
    raw = load_coca_raw()
    if not raw:
        print("Warning: Could not load COCA data, using fallback vocabulary")
        return _get_fallback_vocabulary()

    filtered = filter_content_words(raw, max_rank)
    print(f"Filtered to {len(filtered)} content words from top {max_rank}")

    # Build vocabulary entries
    vocab = []
    seen_words = set()

    for w in filtered:
        # Skip duplicates (same word, different POS)
        if w['word'] in seen_words:
            continue
        seen_words.add(w['word'])

        # Get definition
        definition = get_simple_definition(w['word'], w['pos'])

        # Generate inflections
        inflections = generate_inflections(w['word'], w['pos'])

        vocab.append(VocabWord(
            word=w['word'],
            pos=w['pos'],
            rank=w['rank'],
            frequency=w['freq'],
            definition=definition,
            inflections=inflections,
        ))

    # Cache results
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_data = {
        'max_rank': max_rank,
        'words': [
            {
                'word': v.word,
                'pos': v.pos,
                'rank': v.rank,
                'frequency': v.frequency,
                'definition': v.definition,
                'inflections': v.inflections,
            }
            for v in vocab
        ]
    }
    with open(VOCAB_CACHE_PATH, 'w') as f:
        json.dump(cache_data, f)

    return vocab


def _get_fallback_vocabulary() -> List[VocabWord]:
    """Fallback vocabulary if COCA data unavailable."""
    # Basic common words
    fallback = [
        ("walk", "v", "to move by putting one foot in front of the other"),
        ("run", "v", "to move fast using your legs"),
        ("help", "v", "to make something easier for someone"),
        ("want", "v", "to wish for something"),
        ("need", "v", "to require something"),
        ("find", "v", "to discover or locate something"),
        ("give", "v", "to hand something to someone"),
        ("take", "v", "to get hold of something"),
        ("make", "v", "to create or build something"),
        ("know", "v", "to have information in your mind"),
        ("think", "v", "to use your mind to consider something"),
        ("see", "v", "to use your eyes to look at something"),
        ("time", "n", "the measure of when things happen"),
        ("day", "n", "the period when the sun is up"),
        ("place", "n", "a location or area"),
        ("thing", "n", "an object or item"),
        ("person", "n", "a human being"),
        ("man", "n", "an adult male person"),
        ("woman", "n", "an adult female person"),
        ("home", "n", "the place where you live"),
        ("good", "j", "of high quality or pleasant"),
        ("new", "j", "recently made or started"),
        ("old", "j", "having lived for many years"),
        ("big", "j", "large in size"),
        ("small", "j", "little in size"),
    ]

    return [
        VocabWord(
            word=w, pos=p, rank=i+1, frequency=10000-i*100,
            definition=d, inflections=generate_inflections(w, p)
        )
        for i, (w, p, d) in enumerate(fallback)
    ]


# =============================================================================
# Vocabulary Sampling
# =============================================================================

class VocabSampler:
    """Sample vocabulary words with frequency weighting."""

    def __init__(self, max_rank: int = 2000):
        self.vocab = build_vocabulary(max_rank)
        self._build_index()

    def _build_index(self):
        """Build lookup indices."""
        self.by_word = {v.word: v for v in self.vocab}
        self.by_pos = {'n': [], 'v': [], 'j': [], 'r': []}
        for v in self.vocab:
            if v.pos in self.by_pos:
                self.by_pos[v.pos].append(v)

        # Build inflection lookup
        self.inflection_to_word = {}
        for v in self.vocab:
            for infl in v.inflections:
                self.inflection_to_word[infl.lower()] = v.word

    def sample(self, n: int = 12, weighted: bool = True) -> List[VocabWord]:
        """Sample n vocabulary words."""
        import random

        if weighted:
            # Weight by inverse rank (more common = higher weight)
            weights = [1 / v.rank for v in self.vocab]
            total = sum(weights)
            weights = [w / total for w in weights]

            sampled = []
            used = set()
            while len(sampled) < min(n, len(self.vocab)):
                idx = random.choices(range(len(self.vocab)), weights=weights, k=1)[0]
                if idx not in used:
                    used.add(idx)
                    sampled.append(self.vocab[idx])

            return sampled
        else:
            return random.sample(self.vocab, min(n, len(self.vocab)))

    def sample_tuples(self, n: int = 12, weighted: bool = True) -> List[Tuple[str, str]]:
        """Sample and return as (word, definition) tuples."""
        return [v.to_tuple() for v in self.sample(n, weighted)]

    def find_word_in_text(self, text: str) -> Optional[VocabWord]:
        """Find first vocabulary word appearing in text."""
        text_lower = text.lower()
        words_in_text = set(re.findall(r'\b\w+\b', text_lower))

        for word in words_in_text:
            if word in self.inflection_to_word:
                base = self.inflection_to_word[word]
                return self.by_word.get(base)

        return None

    def find_all_words_in_text(self, text: str) -> List[VocabWord]:
        """Find all vocabulary words appearing in text."""
        text_lower = text.lower()
        words_in_text = set(re.findall(r'\b\w+\b', text_lower))

        found = []
        seen = set()
        for word in words_in_text:
            if word in self.inflection_to_word:
                base = self.inflection_to_word[word]
                if base not in seen:
                    seen.add(base)
                    v = self.by_word.get(base)
                    if v:
                        found.append(v)

        return sorted(found, key=lambda v: v.rank)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Test vocabulary module."""
    print("=" * 60)
    print("VOCABULARY MODULE TEST")
    print("=" * 60)

    # Ensure NLTK data is available
    if HAS_NLTK:
        import nltk
        try:
            wn.synsets('test')
        except LookupError:
            print("Downloading WordNet data...")
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

    # Build vocabulary
    print("\n--- Building Vocabulary ---")
    sampler = VocabSampler(max_rank=1000)
    print(f"Loaded {len(sampler.vocab)} vocabulary words")

    # Show samples by POS
    print("\n--- Sample by POS ---")
    for pos, name in [('n', 'Nouns'), ('v', 'Verbs'), ('j', 'Adjectives'), ('r', 'Adverbs')]:
        words = sampler.by_pos.get(pos, [])[:5]
        print(f"{name}: {[w.word for w in words]}")

    # Sample words
    print("\n--- Random Sample (weighted) ---")
    sample = sampler.sample(10)
    for v in sample:
        print(f"  {v.word} ({v.pos}): {v.definition}")
        print(f"    inflections: {v.inflections[:4]}")

    # Test text matching
    print("\n--- Text Matching ---")
    test_texts = [
        "Patrolling the Mojave almost makes you wish for a nuclear winter.",
        "I need to find my way home.",
        "The Hexagon expects compliance.",
    ]
    for text in test_texts:
        found = sampler.find_all_words_in_text(text)
        print(f"'{text[:40]}...'")
        print(f"  Found: {[v.word for v in found]}")


if __name__ == "__main__":
    main()
