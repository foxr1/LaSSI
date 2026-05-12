import collections
import io
import re

import dacite
import yaml

from LaSSI.external_services.web_cralwer.ScraperConfiguration import ScraperConfiguration

import six


_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_yaml_row_sentences(text):
    """Split a YAML row into sub-sentences at end-of-sentence boundaries followed by
    an uppercase-initial token. Single-sentence rows return as a single-element list.
    Datagramdb processes one row at a time, so multi-sentence rows produce a single
    merged graph that conflates nodes across sentence boundaries; splitting first
    lets each sub-sentence be parsed independently."""
    if not isinstance(text, str) or not text.strip():
        return [text] if text else []
    parts = _SENTENCE_BOUNDARY_RE.split(text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def SentenceLoader(arg):
    if isinstance(arg, io.IOBase):
        arg = yaml.safe_load(arg)
    if isinstance(arg, dict):
        arg = dacite.from_dict(data_class=ScraperConfiguration, data=arg)
    if isinstance(arg, ScraperConfiguration):
        from LaSSI.external_services.web_cralwer.Scrape import Scrape
        return Scrape(arg)
    elif isinstance(arg, six.string_types):
        return [str(arg)]
    elif isinstance(arg, collections.abc.Iterable):
        return list(map(str, arg))
