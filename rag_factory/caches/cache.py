import json
from pathlib import Path

from peewee import (
    AutoField,
    CharField,
    Model,
    SqliteDatabase,
    TextField,
    SQL,
)

# We don't init the database here.
# It is initialized by the application startup.
db = SqliteDatabase(None)


class _BaseCache(Model):
    class Meta:
        database = db


class _TranslationCache(_BaseCache):
    id = AutoField()
    translate_engine = CharField(max_length=20)
    translate_engine_params = TextField()
    original_text = TextField()
    translation = TextField()

    class Meta:
        constraints = [
            SQL(
                "UNIQUE (translate_engine, translate_engine_params, original_text) ON CONFLICT REPLACE"
            ),
        ]


class _LlmResponseCache(_BaseCache):
    id = AutoField()
    llm_name = CharField(max_length=50)
    llm_params = TextField()
    prompt = TextField()
    response = TextField()

    class Meta:
        constraints = [
            SQL("UNIQUE (llm_name, llm_params, prompt) ON CONFLICT REPLACE"),
        ]


class _EntityInfoCache(_BaseCache):
    id = AutoField()
    entity_name = CharField(max_length=255, index=True, unique=True)
    entity_info = TextField()

class _CommunityInfoCache(_BaseCache):
    id = AutoField()
    community_id = CharField(max_length=255, index=True, unique=True)
    community_info = TextField()

    class Meta:
        constraints = [
            SQL("UNIQUE (community_id) ON CONFLICT REPLACE"),
        ]

class _CommunitySummaryCache(_BaseCache):
    id = AutoField()
    community_id = CharField(max_length=255, index=True, unique=True)
    summary = TextField()


def _sort_dict_recursively(obj):
    if isinstance(obj, dict):
        return {
            k: _sort_dict_recursively(v)
            for k in sorted(obj.keys())
            for v in [obj[k]]
        }
    elif isinstance(obj, list):
        return [_sort_dict_recursively(item) for item in obj]
    return obj


class TranslationCache:
    def __init__(self, translate_engine: str, translate_engine_params: dict = None):
        self.translate_engine = translate_engine
        self.replace_params(translate_engine_params)

    def replace_params(self, params: dict = None):
        if params is None:
            params = {}
        self.params = params
        params = _sort_dict_recursively(params)
        self.translate_engine_params = json.dumps(params)

    def get(self, original_text: str) -> str | None:
        result = _TranslationCache.get_or_none(
            translate_engine=self.translate_engine,
            translate_engine_params=self.translate_engine_params,
            original_text=original_text,
        )
        return result.translation if result else None

    def set(self, original_text: str, translation: str):
        _TranslationCache.create(
            translate_engine=self.translate_engine,
            translate_engine_params=self.translate_engine_params,
            original_text=original_text,
            translation=translation,
        )


class LlmResponseCache:
    def __init__(self, llm_name: str, llm_params: dict = None):
        self.llm_name = llm_name
        self.replace_params(llm_params)

    def replace_params(self, params: dict = None):
        if params is None:
            params = {}
        self.params = params
        params = _sort_dict_recursively(params)
        self.llm_params = json.dumps(params)

    def get(self, prompt: str) -> str | None:
        result = _LlmResponseCache.get_or_none(
            llm_name=self.llm_name,
            llm_params=self.llm_params,
            prompt=prompt,
        )
        return result.response if result else None

    def set(self, prompt: str, response: str):
        _LlmResponseCache.create(
            llm_name=self.llm_name,
            llm_params=self.llm_params,
            prompt=prompt,
            response=response,
        )


class EntityInfoCache:
    def get(self, entity_name: str) -> str | None:
        result = _EntityInfoCache.get_or_none(entity_name=entity_name)
        return result.entity_info if result else None

    def set(self, entity_name: str, entity_info: str):
        _EntityInfoCache.replace(
            entity_name=entity_name, entity_info=entity_info
        ).execute()

class CommunityInfoCache:
    def get(self, community_id: str) -> str | None:
        result = _CommunityInfoCache.get_or_none(community_id=community_id)
        return result.community_info if result else None

    def set(self, community_id: str, community_info: str):
        _CommunityInfoCache.replace(
            community_id=community_id, community_info=community_info
        ).execute()


class CommunitySummaryCache:
    def get(self, community_id: str) -> str | None:
        result = _CommunitySummaryCache.get_or_none(community_id=community_id)
        return result.summary if result else None

    def set(self, community_id: str, summary: str):
        _CommunitySummaryCache.replace(
            community_id=community_id, summary=summary
        ).execute()


def init_db(cache_folder: Path, remove_exists=False):
    cache_folder.mkdir(parents=True, exist_ok=True)
    cache_db_path = cache_folder / "cache.v1.db"
    if remove_exists and cache_db_path.exists():
        cache_db_path.unlink()
    db.init(
        str(cache_db_path),
        pragmas={
            "journal_mode": "wal",
            "busy_timeout": 1000,
        },
    )
    db.create_tables(
        [_TranslationCache, _LlmResponseCache, _EntityInfoCache, _CommunityInfoCache, _CommunitySummaryCache],
        safe=True,
    )


def init_test_db():
    import tempfile

    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    cache_db_path = temp_file.name
    temp_file.close()

    test_db = SqliteDatabase(
        cache_db_path,
        pragmas={
            "journal_mode": "wal",
            "busy_timeout": 1000,
        },
    )
    models = [
        _TranslationCache,
        _LlmResponseCache,
        _EntityInfoCache,
        _CommunityInfoCache,
        _CommunitySummaryCache,
    ]
    test_db.bind(models, bind_refs=False, bind_backrefs=False)
    test_db.connect()
    test_db.create_tables(models, safe=True)
    return test_db


def clean_test_db(test_db):
    models = [
        _TranslationCache,
        _LlmResponseCache,
        _EntityInfoCache,
        _CommunityInfoCache,
        _CommunitySummaryCache,
    ]
    test_db.drop_tables(models)
    test_db.close()
    db_path = Path(test_db.database)
    if db_path.exists():
        db_path.unlink()
    wal_path = Path(str(db_path) + "-wal")
    if wal_path.exists():
        wal_path.unlink()
    shm_path = Path(str(db_path) + "-shm")
    if shm_path.exists():
        shm_path.unlink()

# Default initialization
# In multi-threaded scenarios, ensure this is called only once at startup.
# default_cache_folder = Path.home() / ".cache" / "rag_factory"
# init_db(default_cache_folder)
