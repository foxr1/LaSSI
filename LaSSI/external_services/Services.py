import tempfile

from StanfordNLPExtractor.OldWrapper import OldWrapper
from nltk import WordNetLemmatizer

from LaSSI.external_services.Existentials import Existentials
from LaSSI.external_services.utilities.DatabaseConfiguration import load_db_configuration
from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import FuzzyStringMatchDatabase


class Services:
    __instance = None

    @staticmethod
    def getInstance(logger=None)->'Services':
        """ Static access method. """
        if Services.__instance == None:
            Services(logger)
        if hasattr(Services, '__instance'):
            assert Services.__instance is not None
            return Services.__instance
        elif hasattr(Services, '_Services__instance'):
            assert Services._Services__instance is not None
            return Services._Services__instance
        raise RuntimeError("ERROR: cannot found instantiated field")

    def setHOnK(self, honk):
        self.honk = honk
        from LaSSI.external_services.HOnKFuzzyMatch import HOnKFuzzyMatch
        self.fuzzyHOnK = HOnKFuzzyMatch(self.postgres, self.stanza.nlp_token, self.honk)

    def getHOnK(self):
        if self.honk is None:
            # This exists for doing multiprocessing
            from LaSSI.HOnK.HOnK import HOnKSingleton
            fuzzyDBs = load_db_configuration("connection.yaml")

            (FuzzyStringMatchDatabase
             .instance()
             .init(fuzzyDBs.db, fuzzyDBs.uname, fuzzyDBs.pw, fuzzyDBs.host, fuzzyDBs.port))

            HOnKSingleton.instance()
            HOnKSingleton.init("cache", fuzzyDBs.uname, fuzzyDBs.pw,
                                     fuzzyDBs.host, fuzzyDBs.port, False, "LaSSI/HOnK.ttl",
                                     rules_path="raw_data/logical_analysis.json")
            self.setHOnK(HOnKSingleton.get())
            return HOnKSingleton.get()
        return self.honk

    def getFuzzyHOnK(self):
        if self.fuzzyHOnK is None:
            # This path is hit by spawned worker processes which cannot open the
            # RocksDB store (the main process holds the exclusive write lock).
            # The main process already built the "honk" postgres table
            # before spawning workers, so workers only need a live DB connection
            # and a parmo stub for most_specific_type (a pure string function
            # that never touches the RDF store).
            fuzzyDBs = load_db_configuration("connection.yaml")
            db = FuzzyStringMatchDatabase.instance()
            db.init(fuzzyDBs.db, fuzzyDBs.uname, fuzzyDBs.pw, fuzzyDBs.host, fuzzyDBs.port)

            with db.connection.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                    ("honk",),
                )
                table_exists = cur.fetchone()[0]

            if table_exists:
                # Table is ready — skip opening the RocksDB store entirely.
                # parmo=None is safe: ResolveMultiEntity guards every call to
                # parmo.most_specific_type() with `and parmo`.
                from LaSSI.external_services.HOnKFuzzyMatch import HOnKFuzzyMatch
                self.fuzzyHOnK = HOnKFuzzyMatch(db, self.stanza.nlp_token, None)
            else:
                # Table not yet built (should not happen in normal flow, but
                # fall back to the full initialisation path just in case).
                with tempfile.NamedTemporaryFile() as honk_tab:
                    with open(honk_tab.name, 'w') as f:
                        self.getHOnK().dumpTypedObjectsToTAB(f)
                    db.create("honk", honk_tab.name,
                              '(id integer NOT NULL, idx text, t text, type text)')
        return self.fuzzyHOnK

    def getGeoNames(self):
        return self.geonames

    def getStanza(self):
        return self.stanza

    def getStanzaNLPToken(self):
        return self.stanza.nlp_token

    def getStanzaNLP(self):
        return self.stanza.nlp

    def lemmatize_sentence(self, text):
        doc = self.stanza.nlp(text)
        lemmas = set()
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.pos.lower() != 'aux':
                    lemmas.add(word.lemma)
        return lemmas

    def getStanzaSTNLP(self):
        return self.stanza.stNLP

    def resolveTimeUnits(self, sentences):
        if self.old_java_Service is None:
            self.logger("init old java service")
            self.old_java_Service = OldWrapper.getInstance()
        try:
            return self.old_java_Service.getTimeUnits(sentences)
        except Exception as e:
            self.logger(f"getTimeUnits failed ({e}); returning empty time units for all sentences")
            return [[] for _ in sentences]

    def getWTLemmatizer(self):
        return self.lemmatizer

    def getConcepts(self):
        return self.conceptnet

    def getGSMString(self, sentences):
        if self.old_java_Service is None:
            self.logger("init old java service")
            self.old_java_Service = OldWrapper.getInstance()
        return self.old_java_Service.generateGSMDatabase(sentences)

    def getExistentials(self):
        return self.existentials

    def log(self, message):
        self.logger(message)

    def __init__(self, logger=None):
        """ Virtually private constructor. """
        if Services.__instance is not None:
            raise Exception("This class is a singleton!")
        elif logger is None:
            raise Exception("The first initialization should provide a non-None logger!")
        else:
            from LaSSI.external_services.Stanza import StanzaService
            from LaSSI.external_services.GeoNames import GeoNamesService
            from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import FuzzyStringMatchDatabase
            from LaSSI.external_services.ConceptNet5 import ConceptNetService
            from StanfordNLPExtractor.OldWrapper import OldWrapper
            from LaSSI.external_services.HOnKFuzzyMatch import HOnKFuzzyMatch
            self.logger = logger

            self.logger("init honk")
            self.honk = None
            self.logger("retrieving postgres")
            self.postgres = FuzzyStringMatchDatabase.instance()
            self.logger("init stanza")
            self.stanza = StanzaService()
            # self.logger("init geonames wrapper")
            # self.geonames = GeoNamesService(self.postgres, self.stanza.nlp_token)
            # self.logger("init conceptnet wrapper")
            # self.conceptnet = ConceptNetService(self.postgres, self.stanza.nlp_token)
            # self.logger("init fuzzyHOnK wrapper")
            self.fuzzyHOnK = None
            self.logger("init old java service")
            self.old_java_Service = None
            self.logger("init WordNet Lemmatizer")
            self.lemmatizer = WordNetLemmatizer()
            self.logger("init existentials")
            self.existentials = Existentials()
            Services.__instance = self
