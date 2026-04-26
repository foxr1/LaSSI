from typing import Union, Iterable


class GSMObject:
    """
    Implementing an empty object
    """

    @property
    def ell(self)->list[str]:
        return []

    @property
    def xi(self)->list[str]:
        return []

    def property(self, key:str)->Union[str,float,int]:
        raise KeyError

    @property
    def property_keys(self)->Iterable[str]:
        return []

    @property
    def scores(self)->list[float]:
        return [1.0]

    def containment(self, key:str)->list[tuple[float, 'GSMObject']]:
        raise KeyError

    def containment_keys(self)->Iterable[str]:
        raise KeyError