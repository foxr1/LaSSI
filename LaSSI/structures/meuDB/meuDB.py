import dataclasses
from typing import List


@dataclasses.dataclass(frozen=True)
class MeuDBEntry:
    text: str
    type: str
    start_char: int
    end_char: int
    monad: str
    confidence: float
    id: str = None
    source: str = None

    @classmethod
    def _normalise_type(cls, type_val):
        if isinstance(type_val, list):
            return type_val[0] if type_val else "None"
        return type_val

    @classmethod
    def from_dict(cls, data):
        return cls(
            text = data.get('text'),
            type = cls._normalise_type(data.get('type')),
            start_char = data.get('start_char'),
            end_char=data.get('end_char'),
            monad=data.get('monad'),
            confidence=data.get('confidence', 1.0),
            id= data.get("id"),
            source= data.get("source")
        )

    @classmethod
    def from_dict_with_src(cls, data, src):
        return cls(
            text = data.get('text'),
            type = cls._normalise_type(data.get('type')),
            start_char = data.get('start_char'),
            end_char=data.get('end_char'),
            monad=data.get('monad'),
            confidence=data.get('confidence', 1.0),
            id= data.get("id"),
            source= str(src)
        )

@dataclasses.dataclass
class MeuDB:
    first_sentence: str
    multi_entity_unit: List[MeuDBEntry]

    @classmethod
    def from_dict(cls, data):
        return cls(
            first_sentence = data.get('first_sentence'),
            multi_entity_unit = [MeuDBEntry.from_dict(x) for x in data.get('multi_entity_unit')]
        )
