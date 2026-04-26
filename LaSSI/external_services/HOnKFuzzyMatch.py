__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

import math


class HOnKFuzzyMatch(object):

    def __init__(self, psql, nlp, parmo):
        from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import DBFuzzyStringMatching
        # If we have a remote parmo, use it as the matcher instead of Postgres
        if parmo is not None and hasattr(parmo, 'endpoint_url'):
            self.s = parmo
        else:
            self.s = DBFuzzyStringMatching(psql, "honk")
        self.nlp = nlp  # StanzaService().nlp_token
        self.parmo = parmo

    def resolve_u(self, recallThreshold, precisionThreshold, s):
        from LaSSI.ner.ResolveMultiEntity import ResolveMultiNamedEntity
        ar = ResolveMultiNamedEntity(recallThreshold, precisionThreshold, "HOnK", parmo=self.parmo)
        return ar.start(s, self.s, self, self.nlp, None)
