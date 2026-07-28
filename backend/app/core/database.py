import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


load_dotenv()


class DatabaseConfigurationError(RuntimeError):
    """資料庫設定不完整。"""


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    建立並重複使用 SQLAlchemy Engine。

    資料庫網址由環境變數 DATABASE_URL 提供，
    不依賴 Streamlit。
    """

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise DatabaseConfigurationError(
            "找不到 DATABASE_URL，請檢查專案根目錄的 .env"
        )

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
    )